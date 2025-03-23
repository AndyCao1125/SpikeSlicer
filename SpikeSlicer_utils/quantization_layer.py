import torch
import tonic
import torch.nn as nn
import numpy as np
from os.path import join, dirname, isfile
from tqdm import tqdm
import torch.nn.functional as F

events_struct = np.dtype([("x", np.int16), ("y", np.int16), ("t", np.int64), ("p", bool)])
def make_structured_array(*args, dtype=events_struct):
    """
    Make a structured array given a variable number of argument values

    Args:
        *args: Values in the form of nested lists or tuples or numpy arrays. Every except the first argument can be of a primitive data type like int or float
    Returns:
        struct_arr: numpy structured array with the shape of the first argument
    """
    assert not isinstance(args[-1], np.dtype), "The `dtype` must be provided as a keyword argument."
    names = dtype.names
    assert len(args) == len(names)
    struct_arr = np.empty_like(args[0], dtype=dtype)
    for arg,name in zip(args,names):
        struct_arr[name] = arg
    return struct_arr

class ValueLayer(nn.Module):
    def __init__(self, mlp_layers, activation=nn.ReLU(), num_channels=9):
        assert mlp_layers[-1] == 1, "Last layer of the mlp must have 1 input channel."
        assert mlp_layers[0] == 1, "First layer of the mlp must have 1 output channel"

        nn.Module.__init__(self)
        self.mlp = nn.ModuleList()
        self.activation = activation

        # create mlp
        in_channels = 1
        for out_channels in mlp_layers[1:]:
            self.mlp.append(nn.Linear(in_channels, out_channels))
            in_channels = out_channels

        # init with trilinear kernel
        path = join(dirname(__file__), "quantization_layer_init", "trilinear_init.pth")
        if isfile(path):
            state_dict = torch.load(path)
            self.load_state_dict(state_dict)
        else:
            self.init_kernel(num_channels)

    def forward(self, x):
        # create sample of batchsize 1 and input channels 1
        x = x[None, ..., None]

        # apply mlp convolution
        for i in range(len(self.mlp[:-1])):
            x = self.activation(self.mlp[i](x))

        x = self.mlp[-1](x)
        x = x.squeeze()

        return x

    def init_kernel(self, num_channels):
        ts = torch.zeros((1, 2000))
        optim = torch.optim.Adam(self.parameters(), lr=1e-2)

        torch.manual_seed(1)

        for _ in tqdm(range(1000)):  # converges in a reasonable time
            optim.zero_grad()
            ts.uniform_(-1, 1)

            # gt
            gt_values = self.trilinear_kernel(ts, num_channels)

            # pred
            values = self.forward(ts)

            # optimize
            loss = (values - gt_values).pow(2).sum()

            loss.backward()
            optim.step()

    def trilinear_kernel(self, ts, num_channels):
        gt_values = torch.zeros_like(ts)

        gt_values[ts > 0] = (1 - (num_channels - 1) * ts)[ts > 0]
        gt_values[ts < 0] = ((num_channels - 1) * ts + 1)[ts < 0]

        gt_values[ts < -1.0 / (num_channels - 1)] = 0
        gt_values[ts > 1.0 / (num_channels - 1)] = 0

        return gt_values


class EST(nn.Module):
    def __init__(self, dim,
                 mlp_layers=[1, 30, 30, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1)):
        nn.Module.__init__(self)
        self.value_layer = ValueLayer(mlp_layers,
                                      activation=activation,
                                      num_channels=dim[0])
        self.dim = dim

    def forward(self, events):
        # points is a list, since events can have any size
        B = int((1 + events[-1, -1]).item())
        num_voxels = int(2 * np.prod(self.dim) * B)
        vox = events[0].new_full([num_voxels, ], fill_value=0).to(torch.float32)
        C, H, W = self.dim

        # get values for each channel
        x, y, t, p, b = events.t()
        x = x.to(torch.int64)
        y = y.to(torch.int64)
        p = p.to(torch.int64)
        b = b.to(torch.int64)
        t = t.to(torch.float64)
        # normalizing timestamps
        scale_max = []
        for bi in range(B):
            t[events[:, -1] == bi] -= t[events[:, -1] == bi].min()
            scale_max.append(t[events[:, -1] == bi].max())
            t[events[:, -1] == bi] /= (t[events[:, -1] == bi].max() + 1e-8)
        # p = (p + 1) / 2  # maps polarity to 0, 1
        scale_max = torch.stack(scale_max)
        t = t.to(torch.float32)
        idx_before_bins = x \
                          + W * y \
                          + 0 \
                          + W * H * C * p \
                          + W * H * C * 2 * b

        for i_bin in range(C):
            with torch.no_grad():
                values = t * self.value_layer.forward(t - 1 / 2) # adapt to channel 1 condition

            # draw in voxel grid
            idx = idx_before_bins + W * H * i_bin
            vox.put_(idx.long(), values, accumulate=True)

        vox = vox.view(-1, 2, C, H, W)
        vox = vox * scale_max[:, None, None, None, None]
        vox = torch.cat([vox[:, 0, ...], vox[:, 1, ...]], 1)

        return vox
    

class ToFrame(nn.Module):
    def __init__(self, dim):
        nn.Module.__init__(self)
        self.dim = dim

    def forward(self, events):
        epsilon = 10e-3
        B = int(1+events[-1, -1].item())
        # tqdm.write(str(B))
        num_voxels = int(2 * np.prod(self.dim) * B)
        C, H, W = self.dim
        vox = events[0].new_full([num_voxels, ], fill_value=0)
        # get values for each channel
        x, y, t, p, b = events.T
        x = x.to(torch.int64)
        y = y.to(torch.int64)
        p = p.to(torch.int64)
        b = b.to(torch.int64)
        t = t.to(torch.float64)

        # print(f' t start: {t.min()}, t end: {t.max()}')
        # p = (p + 1) / 2  # maps polarity to 0, 1
        # normalizing timestamps
        # tqdm.write("-------------bi shape----------------")
        for bi in range(B):
            # tqdm.write(str(t[events[:, -1] == bi].shape))
            t[events[:, -1] == bi] -= (t[events[:, -1] == bi].min() - 1e-8)
            t[events[:, -1] == bi] /= t[events[:, -1] == bi].max()

        idx_before_bins = x \
                          + W * y \
                          + 0 \
                          + W * H * C * p \
                          + W * H * C * 2 * b
        for i_bin in range(C):
            values = torch.zeros_like(t).to(vox.dtype)
            values[(t > i_bin/C) & (t <= (i_bin+1)/C)] = 1

            # draw in voxel grid
            idx = idx_before_bins + W * H * i_bin
            vox.put_(idx.long(), values, accumulate=True)

        vox = vox.view(-1, 2, C, H, W)
        vox = torch.cat([vox[:, 0, ...], vox[:, 1, ...]], 1)
        vox = vox.to(torch.float32)
        return vox

class ToTimeSurface(nn.Module):
    def __init__(self, resolution):
        nn.Module.__init__(self)
        self.resolution = resolution

    def forward(self, events):
        epsilon = 10e-3
        B = int(1+events[-1, -1].item())
        # tqdm.write(str(B))
        H, W = self.resolution

        # Unpack the events
        x, y, t, p, batch_idx = events.T
        x = x.to(torch.int64)
        y = y.to(torch.int64)
        batch_idx = batch_idx.to(torch.int64)
        p = p.to(torch.int64)
        
        # Initialize the time surface tensor for all batches and polarities
        time_surfaces = torch.full((B, 2, H, W), 0, device=events.device).to(t.dtype)

        # Create a 1D index for the flattened time surface
        # Adjust the index to account for the polarity dimension
        idx = batch_idx * 2 * H * W + p * H * W + y * W + x

        # Use scatter to update the time surfaces
        time_surfaces.view(-1).put_(idx, t, accumulate=False)
        time_surfaces = time_surfaces.view(B, 2, H, W)
        # for i in range(B):
        #     save_img(time_surfaces[i, 0], "before"+str(i))
        second_min = torch.stack([time_surface.view(-1).unique().topk(2, largest=False)[0][1] for time_surface in time_surfaces]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        time_surfaces = (time_surfaces - second_min).clamp(min=0) # no norm
        # time_surfaces = ((time_surfaces - second_min) / (torch.stack([time_surface.max() for time_surface in time_surfaces]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) - second_min)).clamp(min=0, max=1)
        # for i in range(B):
        #     save_img(time_surfaces[i, 0], "after"+str(i))
        return time_surfaces.to(torch.float32)
    
def QuantizationLayer(representation, resolution):
    if representation == 'frame':
        quant_layer = ToFrame((1, *resolution))
    elif representation == 'timesurface':
        quant_layer = ToTimeSurface(resolution)
    elif representation == "EST":
        quant_layer = EST((1, *resolution))
    else:
        raise Exception("Wrong representation type")
    return quant_layer

def save_img(img, name, path):
    import matplotlib.pyplot as plt
    img = img.cpu()
    plt.imshow(img)
    plt.savefig(path+name+".png")
    
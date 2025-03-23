import os
import torch
from ltr.dataset import Lasot, MSCOCOSeq, Got10k, TrackingNet, FE108
from ltr.data import processing, sampler, LTRLoader
import ltr.models.tracking.transt as transt_models
from ltr import actors
from ltr.trainers import TransT_SpikeSlicerTrainer
import ltr.data.transforms as tfm
from ltr import MultiGPU
import time
import wandb
from SpikeSlicer_utils.SpikeSliceNet import SpikeSlicerNet_S_IF, MembraneLoss, SpikeSlicerNet_B_LIF, SpikeSlicerNet_B_IF
from SpikeSlicer_utils.quantization_layer import QuantizationLayer
import tonic



def run(settings):
    # Most common settings are assigned in the settings struct
    settings.device = 'cuda'
    settings.description = 'TransT with default settings.'
    settings.batch_size = 10
    settings.num_workers = 6
    settings.multi_gpu = False
    settings.print_interval = 1
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]
    settings.search_area_factor = 4.0
    settings.template_area_factor = 2.0
    settings.search_feature_sz = 32
    settings.template_feature_sz = 16
    settings.search_sz = settings.search_feature_sz * 8
    settings.temp_sz = settings.template_feature_sz * 8
    settings.center_jitter_factor = {'search': 3, 'template': 0}
    settings.scale_jitter_factor = {'search': 0.25, 'template': 0}
    settings.trained_tracker_checkpoint = ""
    settings.event_resolution = (260, 346)
    settings.extend_step = 2
    settings.extend_width = 1
    settings.group_num = 5
    settings.snn_max_epoch = 50
    settings.max_epoch = 50
    settings.alpha = 0.5
    settings.alpha_lr = 0.1
    settings.split_template_epoch = 10
    settings.extend_width_epoch = 25
    # Transformer
    settings.position_embedding = 'sine'
    settings.hidden_dim = 256
    settings.dropout = 0.1
    settings.nheads = 8
    settings.dim_feedforward = 2048
    settings.featurefusion_layers = 4
    settings.event_representation = "frame"
    settings.log_name = f"transt_train"
    settings.project_path = os.path.join(settings.project_path, settings.log_name)
    wandb.init(project="SpikeSlicer", name=settings.log_name, config=settings)
    # Train datasets

    fe108_train_snn = FE108(settings.env.fe108_dir, subset='val', groupnum=settings.group_num, txt_suffix=".txt")
    fe108_train_ann = FE108(settings.env.fe108_dir, subset='train', groupnum=settings.group_num, txt_suffix=".txt")
    fe108_train_all = FE108(settings.env.fe108_dir, subset='train', groupnum=settings.group_num, txt_suffix="_all.txt")
    # The joint augmentation transform, that is applied to the pairs jointly

    # The augmentation transform applied to the training set (individually to each image in the pair)

    # Data processing to do on the training pairs
    data_processing_train = processing.TransTProcessing(search_area_factor=settings.search_area_factor,
                                                      template_area_factor = settings.template_area_factor,
                                                      search_sz=settings.search_sz,
                                                      temp_sz=settings.temp_sz,
                                                      center_jitter_factor=settings.center_jitter_factor,
                                                      scale_jitter_factor=settings.scale_jitter_factor,
                                                      mode='sequence',
                                                      transform=None,
                                                      joint_transform=None,
                                                      train_mode="spikeslicer")

    # The sampler for training
    dataset_train_snn = sampler.TransTSampler([fe108_train_snn], [1], samples_per_epoch=400*settings.batch_size, max_gap=200,
                                processing=None, num_search_frames=20, num_template_frames=20, frame_sample_mode='continual')
    dataset_train_ann = sampler.TransTSampler([fe108_train_ann], [1], samples_per_epoch=400 * settings.batch_size,
                                              max_gap=200,
                                              processing=None, num_search_frames=20, num_template_frames=20,
                                              frame_sample_mode='continual')
    dataset_train_all = sampler.TransTSampler([fe108_train_all], [1], samples_per_epoch=400 * settings.batch_size,
                                              max_gap=200,
                                              processing=None, num_search_frames=20, num_template_frames=20,
                                              frame_sample_mode='continual')
    # The loader for training
    loader_train_snn = LTRLoader('train', dataset_train_snn, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=True, drop_last=True, stack_dim=0)
    loader_train_ann = LTRLoader('train', dataset_train_ann, training=True, batch_size=settings.batch_size,
                                 num_workers=settings.num_workers,
                                 shuffle=True, drop_last=True, stack_dim=0)
    loader_train_all = LTRLoader('train', dataset_train_all, training=True, batch_size=settings.batch_size,
                                 num_workers=settings.num_workers,
                                 shuffle=True, drop_last=True, stack_dim=0)
    # Create network and actor
    model = transt_models.transt_resnet50(settings)
    model.load_state_dict(torch.load(settings.trained_tracker_checkpoint)['net'], strict=False)
    snn_net = SpikeSlicerNet_S_IF(settings.event_resolution, output_num=1).to(settings.device)

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        model = MultiGPU(model, dim=0)
        snn_net = MultiGPU(snn_net, dim=0)


    objective = transt_models.spikeslicer_transt_loss(settings)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    actor = actors.TranstActor(net=model, objective=objective)

    # Optimizer
    ann_param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": 1e-6,
        },
    ]

    ann_optimizer = torch.optim.AdamW(ann_param_dicts, lr=1e-5, weight_decay=1e-5)
    ann_scheduler = torch.optim.lr_scheduler.StepLR(ann_optimizer, (settings.max_epoch - settings.split_template_epoch) // 2, gamma=0.1)
    snn_optimizer = torch.optim.SGD(snn_net.parameters(), lr=1e-4)
    snn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(snn_optimizer, T_max=settings.snn_max_epoch, eta_min=0.)
    if isinstance(snn_net, SpikeSlicerNet_B_LIF):
        mem_loss = MembraneLoss(v_decay=0.5, i_decay=0.5, alpha=settings.alpha, alpha_lr=settings.alpha_lr)
    elif isinstance(snn_net, SpikeSlicerNet_S_IF) or isinstance(snn_net, SpikeSlicerNet_B_IF):
        mem_loss = MembraneLoss(alpha=settings.alpha, alpha_lr=settings.alpha_lr)
    else:
        raise Exception("Wrong SNN type")
    
    # Create trainer
    quantization_layer = QuantizationLayer(settings.event_representation, settings.event_resolution).to(settings.device)
    trainer = TransT_SpikeSlicerTrainer(actor, snn_net, [loader_train_snn], [loader_train_ann, loader_train_all], snn_optimizer, snn_scheduler, ann_optimizer, ann_scheduler, settings, data_processing_train, mem_loss, quantization_layer)

    # Run training (set fail_safe=False if you are debugging)
    trainer.train(settings.max_epoch, load_latest=False, fail_safe=True)

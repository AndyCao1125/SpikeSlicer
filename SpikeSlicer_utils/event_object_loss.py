from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# iou_seq = torch.FloatTensor([[[0]],[[0]],[[1]],[[0]],[[3]],[[2]]])
# spk_seq = torch.FloatTensor([[[0]],[[0]],[[0]],[[1]]])
# print(int((spk_seq.shape[0]+1)/2))
# buchong = torch.zeros((int(spk_seq.shape[0]/2),1,1))
# print(buchong)
# print(torch.cat((spk_seq,buchong),0))
# print(a.shape,spk_seq.shape)
# b = a.view(4)
# c = spk_seq.view(4)
# print(b.shape,c.shape)

def maxIOU_cn_loss(spk_seq, IOU_seq, punish=False):  ## spk_seq:[k] IOU_seq:[k+int(k/2)]
    # iou = IOU_seq.view(IOU_seq.shape[0])
    max_idx = IOU_seq.argmax(dim=0)  ## 找到IOU序列中的最大值
    len_of_spk_seq = len(spk_seq)

    buchong = torch.zeros(2, 1, device=spk_seq.device)  ## 对spk_seq进行补充，长度由k变成k+2，向下取整

    # half_size = torch.ceil(len(spk_seq) / 2)  ## 计算(k/2,k)的长度
    # target_idx = max_idx + half_size  ## 当长度为k的向量时的目标index位置

    # label = F.one_hot(max_idx,spk_seq.shape[0])
    spk_seq = torch.cat((spk_seq, buchong), 0).view(-1)
    print(f'spk_seq is :{spk_seq}')
    # label = torch.tensor([max_idx])
    print(f'label for this spkseq is :{label}')
    celoss = nn.CrossEntropyLoss()

    if punish == False:
        loss = celoss(spk_seq.view(1, spk_seq.shape[0]), label.long())
    if punish == True:
        loss = torch.log(5) * celoss(spk_seq.view(1, spk_seq.shape[0]), label.long())
    return loss




def maxIOU_event_loss(spk_seq, IOU_seq, normalised=False):  ## 找出最值点，做elementwise的mse
    '''
    spk_seq.shape:{N,1}  可能是{1,1}{2,1}{3,1}
    iou_seq.shape:{N,}   可能是{3,}{4,}{5,}
    '''
    # diff_len = len(IOU_seq) - len(spk_seq)
    buchong = torch.zeros(2, 1, device=spk_seq.device)  ## 对spk_seq进行补充，长度由k变成k+2，向下取整

    spk_seq = torch.cat((spk_seq, buchong), 0).view(-1)  ## spk_seq --> [5]

    max_idx = torch.tensor([IOU_seq.argmax()])  ## shape:{1}
    a = max_idx.unsqueeze(1)
    label = torch.zeros(1, len(IOU_seq)).scatter_(1, a, 1).view(-1)  ## label --> [5]

    # print(spk_seq.shape,label.shape)

    # print(f'spk:{spk_seq}, label:{label}')

    # print(f'label for this spkseq is :{label}')

    # mseloss = nn.MSELoss(reduction='sum')
    mseloss = nn.MSELoss(reduction='mean')

    label = label.to(torch.float32).to(spk_seq.device)
    if normalised:
        loss = mseloss(spk_seq, label) / (spk_seq.size(0) ** 2)
        # loss = mseloss(spk_seq, label)/(spk_seq.size(0))
    else:
        loss = mseloss(spk_seq, label)
    return loss


def mse_event_loss_softmax(spk_seq, IOU_seq):
    # print(f'IOU_seq:{IOU_seq}')
    ## IOU_seq now the smaller the better

    idx_ls = [0, 1, 2]
    spk_idx = spk_seq.max(dim=0)[1]  ## remove the spk_idx from the idx_ls
    idx_ls.pop(spk_idx)

    spk_one = IOU_seq[spk_seq.max(dim=0)[1]]
    compared1 = IOU_seq[idx_ls[0]]
    compared2 = IOU_seq[idx_ls[1]]

    ## if the difference between spkone and the compared one is less than 10%, then the desired one is still the spkone.
    if (torch.abs(compared1 - spk_one) / (spk_one + 1e-6)) < 0.1 and (
            torch.abs(compared2 - spk_one) / (spk_one + 1e-6)) < 0.1:
        max_idx = torch.tensor([spk_idx])  ## shape:{1}
        a = max_idx.unsqueeze(1)
        label = torch.zeros(1, 3).scatter_(1, a, 1).view(-1)  ## label --> one hot
    else:
        max_idx = torch.tensor([IOU_seq.argmax()])  ## shape:{1}
        a = max_idx.unsqueeze(1)
        label = torch.zeros(1, 3).scatter_(1, a, 1).view(-1)  ## label --> one hot

    ## normalize the loss, the smaller the better
    # IOU_seq = (IOU_seq.max(dim=0)[0] - IOU_seq) / (IOU_seq.max(dim=0)[0] - IOU_seq.min(dim=0)[0] + 1e-6)
    # IOU_seq = F.softmax(IOU_seq, dim=0)
    # print(f'difference:{torch.abs(compared1-spk_one)/(spk_one+1e-6)} and {torch.abs(compared2-spk_one)/(spk_one+1e-6)}')
    # max_idx = torch.tensor([IOU_seq.argmax()])    ## shape:{1}
    # a = max_idx.unsqueeze(1)
    # label = torch.zeros(1,len(IOU_seq)).scatter_(1,a,1).view(-1)   ## label --> [5]\

    label = label.to(spk_seq.device)

    loss = F.mse_loss(spk_seq, label)
    # loss = torch.mean(torch.sub(label,spk_seq)**2,dim=0)
    # print(f'spk_seq:{spk_seq}')
    # print(f'IOU_seq:{IOU_seq}')
    # print(f'desired label:{label}')
    # print(f'loss:{loss}')
    return loss, max_idx


def space_event_loss_softmax(spk_seq, IOU_seq):
    # print(f'IOU_seq:{IOU_seq}')
    ## IOU_seq now the smaller the better

    idx_ls = [0, 1, 2]
    spk_idx = spk_seq.max(dim=0)[1]  ## remove the spk_idx from the idx_ls
    idx_ls.pop(spk_idx)

    spk_one = IOU_seq[spk_seq.max(dim=0)[1]]
    compared1 = IOU_seq[idx_ls[0]]
    compared2 = IOU_seq[idx_ls[1]]

    ## if the difference between spkone and the compared one is less than 10%, then the desired one is still the spkone.
    if (torch.abs(compared1 - spk_one) / (spk_one + 1e-6)) < 0.1 and (
            torch.abs(compared2 - spk_one) / (spk_one + 1e-6)) < 0.1:
        max_idx = torch.tensor([spk_idx])  ## shape:{1}
        a = max_idx.unsqueeze(1)
        label = torch.zeros(1, 3).scatter_(1, a, 1).view(-1)  ## label --> one hot
    else:
        max_idx = torch.tensor([IOU_seq.argmax()])  ## shape:{1}
        a = max_idx.unsqueeze(1)
        label = torch.zeros(1, 3).scatter_(1, a, 1).view(-1)  ## label --> one hot

    ## normalize the loss, the smaller the better
    # IOU_seq = (IOU_seq.max(dim=0)[0] - IOU_seq) / (IOU_seq.max(dim=0)[0] - IOU_seq.min(dim=0)[0] + 1e-6)
    # IOU_seq = F.softmax(IOU_seq, dim=0)
    # print(f'difference:{torch.abs(compared1-spk_one)/(spk_one+1e-6)} and {torch.abs(compared2-spk_one)/(spk_one+1e-6)}')
    # max_idx = torch.tensor([IOU_seq.argmax()])    ## shape:{1}
    # a = max_idx.unsqueeze(1)
    # label = torch.zeros(1,len(IOU_seq)).scatter_(1,a,1).view(-1)   ## label --> [5]\

    label = label.to(spk_seq.device)

    spk_float_label = F.softmax(torch.abs(spk_seq - IOU_seq), dim=0).to(spk_seq.device)
    float_label = F.softmax(torch.abs(label - IOU_seq), dim=0).to(spk_seq.device)

    loss = -torch.sum(spk_float_label.mul(torch.log(float_label + 1e-6)), dim=0)

    desired_locak_spk_move = max_idx - torch.tensor([spk_idx])

    # loss = torch.mean(torch.sub(label,spk_seq)**2,dim=0)
    # print(f'spk_seq:{spk_seq}')
    # print(f'IOU_seq:{IOU_seq}')
    # print(f'desired label:{label}')
    # print(f'loss:{loss}')
    return loss, desired_locak_spk_move


def qiangce_event_loss(spk_seq, IOU_seq):
    IOU_seq = torch.FloatTensor(IOU_seq).to(spk_seq.device)
    buchong = torch.zeros(2, 1, device=spk_seq.device)  ## 对spk_seq进行补充，长度由k变成k+2，向下取整
    spk_seq = torch.cat((spk_seq, buchong), 0).view(-1)  ## spk_seq --> [5]
    spk_seq = torch.abs(spk_seq - IOU_seq).to(spk_seq.device)

    max_idx = torch.tensor([IOU_seq.argmax()])  ## shape:{1}
    a = max_idx.unsqueeze(1)
    label = torch.zeros(1, len(IOU_seq)).scatter_(1, a, 1).view(-1)  ## label --> [5]
    label = label.to(spk_seq.device)

    label_seq = torch.abs(label - IOU_seq)

    # print(spk_seq,label_seq)

    celoss = -torch.sum(label_seq.mul(torch.log(spk_seq + 1e-3)), dim=0)

    return celoss


def CE_temp_loss(spk_seq, IOU_seq, punish=False):  ## spk_seq:[k] IOU_seq:[k+2]
    # print(f'spk:{spk_seq},shape:{spk_seq.shape}')
    # max_idx = IOU_seq.argmax(dim=0)  ## 找到IOU序列中的最大值
    # print(f'max_index:{max_idx}')
    # obvious = torch.ones((len(IOU_seq)))
    # obvious[max_idx] = 5
    # punish_val = torch.linspace(1.5,1,steps=len(IOU_seq))
    soft_iou = F.log_softmax(torch.FloatTensor(IOU_seq), dim=0)
    # soft_iou = soft_iou*obvious
    soft_iou = soft_iou.to(spk_seq.device)
    # print(f'soft_iou:{soft_iou}')
    # soft_iou = punish_val*soft_iou
    # print(f'iou_weight:{soft_iou}')
    # len_of_spk_seq=len(spk_seq)
    buchong = torch.zeros(2, 1, device=spk_seq.device)  ## 对spk_seq进行补充，长度由k变成k+2，向下取整
    spk_seq = torch.cat((spk_seq, buchong), 0).view(-1)

    print(f'spk_seq is :{spk_seq}')
    # print(f'label for this spkseq is :{label}')
    celoss = nn.CrossEntropyLoss()

    if punish == False:
        # loss = celoss(soft_iou.view(1,soft_iou.shape[0]), label.long())
        loss = -torch.sum(soft_iou.mul(spk_seq + 0.01), dim=0)
        print(f'loss for this seq:{loss}')
    if punish == True:
        loss = torch.log(5) * celoss(spk_seq.view(1, spk_seq.shape[0]), label.long())
    return loss


##----------------new version 7/5 -----------------


def qiangce_event_loss(spk_seq, IOU_seq):
    IOU_seq = torch.FloatTensor(IOU_seq).to(spk_seq.device)
    buchong = torch.zeros(2, 1, device=spk_seq.device)  ## 对spk_seq进行补充，长度由k变成k+2，向下取整
    spk_seq = torch.cat((spk_seq, buchong), 0).view(-1)  ## spk_seq --> [5]
    spk_seq = torch.abs(spk_seq - IOU_seq).to(spk_seq.device)

    max_idx = torch.tensor([IOU_seq.argmax()])  ## shape:{1}
    a = max_idx.unsqueeze(1)
    label = torch.zeros(1, len(IOU_seq)).scatter_(1, a, 1).view(-1)  ## label --> [5]
    label = label.to(spk_seq.device)

    label_seq = torch.abs(label - IOU_seq)

    # print(spk_seq,label_seq)

    celoss = -torch.sum(label_seq.mul(torch.log(spk_seq + 1e-3)), dim=0)

    return celoss


def generate_penalty_rates(extend_step, penalty):
    # Find the index of the center element
    center = extend_step // 2

    # Generate the penalty rates
    penalty_rates = [1.0 + (center - i) * penalty for i in range(extend_step)]

    # Convert to a tensor
    penalty_rate_tensor = torch.tensor(penalty_rates)

    return penalty_rate_tensor


def mse_event_loss_softmax(IOU_seq, spk_seq, mem_seq, cur_seq, spk_idx, penalty=0.0, extend_step=3,
                           mse=torch.nn.MSELoss()):
    # print(f'IOU_seq:{IOU_seq}')
    ## IOU_seq now the smaller the better

    # idx_ls = [0,1,2]
    # spk_idx = spk_seq.max(dim=0)[1] ## remove the spk_idx from the idx_ls
    # idx_ls.pop(spk_idx)
    # if spk_seq.sum() > 0:
    #     spk_idx = spk_seq.max(dim=0)[1] ## remove the spk_idx from the idx_ls
    # elif spk_seq.sum() ==0:
    #     spk_idx = torch.tensor([len(spk_seq)-1])  ## if no spk, then spk_idx is the last one

    # spk_one = IOU_seq[spk_seq.max(dim=0)[1]]
    # compared1 = IOU_seq[idx_ls[0]]
    # compared2 = IOU_seq[idx_ls[1]]

    ## if the difference between spkone and the compared one is less than 10%, then the desired one is still the spkone.
    # if (torch.abs(compared1-spk_one)/(torch.abs(spk_one)+1e-6)) < 0.1 and  (torch.abs(compared2-spk_one)/(torch.abs(spk_one)+1e-6)) < 0.1:
    #     max_idx = torch.tensor([spk_idx])  ## shape:{1}
    #     a = max_idx.unsqueeze(1)
    #     label = torch.zeros(1,3).scatter_(1,a,1).view(-1)   ## label --> one hot
    # else:
    #     max_idx = torch.tensor([IOU_seq.argmax()])  ## shape:{1}
    #     a = max_idx.unsqueeze(1)
    #     label = torch.zeros(1, 3).scatter_(1, a, 1).view(-1)  ## label --> one hot

    # penalty_rate = generate_penalty_rates(extend_step=len(spk_seq), penalty=penalty)
    # # print(IOU_seq)
    # IOU_seq = torch.where(IOU_seq < 0, IOU_seq / penalty_rate, IOU_seq * penalty_rate)

    max_idx = torch.tensor([IOU_seq.argmax()])  ## shape:[1]
    a = max_idx.unsqueeze(1)
    label = torch.zeros(1, len(spk_seq)).scatter_(1, a, 1).view(-1)  ## label --> one hot
    print(f'label:{label}')
    ## normalize the loss, the smaller the better
    # IOU_seq = (IOU_seq.max(dim=0)[0] - IOU_seq) / (IOU_seq.max(dim=0)[0] - IOU_seq.min(dim=0)[0] + 1e-6)
    # IOU_seq = F.softmax(IOU_seq, dim=0)
    # print(f'difference:{torch.abs(compared1-spk_one)/(torch.abs(spk_one)+1e-6)} and {torch.abs(compared2-spk_one)/(torch.abs(spk_one)+1e-6)}')
    # max_idx = torch.tensor([IOU_seq.argmax()])    ## shape:{1}
    # a = max_idx.unsqueeze(1)
    # label = torch.zeros(1,len(IOU_seq)).scatter_(1,a,1).view(-1)   ## label --> [5]\

    label = label.to(spk_seq.device)
    # print(f'spk_seq:{spk_seq}')
    loss = mse(spk_seq.squeeze(-1).squeeze(-1), label)
    # loss = F.mse_loss(spk_seq,label)

    desired_idx_move = max_idx.to(spk_idx.device) - torch.tensor([spk_idx]).to(spk_idx.device)
    print(f'spkidx:{spk_idx},max_idx:{max_idx}')
    print(f'IOU_seq:{IOU_seq}')
    print(f'mem_seq:{mem_seq.squeeze().detach().cpu()}')
    print(f'cur_seq:{cur_seq.squeeze().detach().cpu()}')
    print(f'spk_seq:{spk_seq.squeeze().detach().cpu()}')
    print(f'desired move:{desired_idx_move.squeeze().detach().cpu()}\n')
    # print(f'desired_idx_move:{desired_idx_move}')

    # loss = torch.mean(torch.sub(label,spk_seq)**2,dim=0)

    # if penalty:
    #     if desired_idx_move < 0.0:
    #         loss = 0.8*loss
    #     elif desired_idx_move > 0.0:
    #         loss = 1.2*loss
    # else:
    #     loss = 1.0*loss
    # print(f'spk_seq:{spk_seq}')
    # print(f'IOU_seq:{IOU_seq}')
    # print(f'desired label:{label}, desired_idx_move:{desired_idx_move}')
    # print(f'loss:{loss}')
    return loss, desired_idx_move


def mse_event_loss(IOU_seq, spk_seq, spk_idx, penalty=0.0, extend_step=3, mse=torch.nn.MSELoss()):
    max_idx = torch.tensor([IOU_seq.argmax()])  ## shape:[1]
    a = max_idx.unsqueeze(1)
    label = torch.zeros(1, len(spk_seq)).scatter_(1, a, 1).view(-1)  ## label --> one hot
    print(f'label:{label}')
    ## normalize the loss, the smaller the better
    # IOU_seq = (IOU_seq.max(dim=0)[0] - IOU_seq) / (IOU_seq.max(dim=0)[0] - IOU_seq.min(dim=0)[0] + 1e-6)
    # IOU_seq = F.softmax(IOU_seq, dim=0)
    # print(f'difference:{torch.abs(compared1-spk_one)/(torch.abs(spk_one)+1e-6)} and {torch.abs(compared2-spk_one)/(torch.abs(spk_one)+1e-6)}')
    # max_idx = torch.tensor([IOU_seq.argmax()])    ## shape:{1}
    # a = max_idx.unsqueeze(1)
    # label = torch.zeros(1,len(IOU_seq)).scatter_(1,a,1).view(-1)   ## label --> [5]\

    label = label.to(spk_seq.device)
    # print(f'spk_seq:{spk_seq}')
    loss = mse(spk_seq.squeeze(-1).squeeze(-1), label)
    # loss = F.mse_loss(spk_seq,label)

    desired_idx_move = max_idx.to(spk_idx.device) - torch.tensor([spk_idx]).to(spk_idx.device)
    print(f'spkidx:{spk_idx},max_idx:{max_idx}')
    print(f'IOU_seq:{IOU_seq}')
    print(f'spk_seq:{spk_seq.squeeze().detach().cpu()}')
    print(f'desired move:{desired_idx_move.squeeze().detach().cpu()}\n')
    # print(f'desired_idx_move:{desired_idx_move}')

    # loss = torch.mean(torch.sub(label,spk_seq)**2,dim=0)

    # if penalty:
    #     if desired_idx_move < 0.0:
    #         loss = 0.8*loss
    #     elif desired_idx_move > 0.0:
    #         loss = 1.2*loss
    # else:
    #     loss = 1.0*loss
    # print(f'spk_seq:{spk_seq}')
    # print(f'IOU_seq:{IOU_seq}')
    # print(f'desired label:{label}, desired_idx_move:{desired_idx_move}')
    # print(f'loss:{loss}')
    return loss, desired_idx_move


def mem_loss(IOU_seq, spk_seq, mem_seq, cur_seq, spk_idx, Vth=1.0, mse=torch.nn.MSELoss(), tau=1, alpha=0.6,
             tolerance=0.2):
    """
    :param mem_seq: membrane potential sequence (with gradient)
    :param I: current sequence (with gradient)
    :param batch_idx: global index of batch
    :param spike_idx: global index of spike
    :param max_idx: global index of max reward (negative loss) value
    :param Vth: threshold of membrane potential
    :param mse: loss function
    :param tau: time decreasing constant of the neuron model
    :param alpha: weight of upper bound
    """
    # print(f'IOU_seq:{IOU_seq.shape},spk_seq:{spk_seq.shape}, mem_seq:{mem_seq.shape}, cur_seq:{cur_seq.shape}')
    # print(spk_seq,mem_seq,cur_seq)
    spk_seq = spk_seq.squeeze(-1)  ## shape:{[X,1]}
    mem_seq = mem_seq.squeeze(-1)
    cur_seq = cur_seq.squeeze(-1)

    # if spk_seq.sum() > 0:
    #     spk_idx = spk_seq.max(dim=0)[1] ## remove the spk_idx from the idx_ls
    # elif spk_seq.sum() ==0:
    #     spk_idx = torch.tensor([len(spk_seq)-1])  ## if no spk, then spk_idx is the last one
    max_iou = np.max(IOU_seq)
    snn_iou = IOU_seq[spk_idx]
    # print(max_iou,snn_iou)
    tolerance_flag = False
    if snn_iou < 0.1:
        print(IOU_seq)
        print(spk_seq.squeeze())
        print(mem_seq.squeeze())
    if np.abs((max_iou - snn_iou) / snn_iou) < tolerance:
        max_idx = spk_idx
        tolerance_flag = True
    else:
        max_idx = torch.tensor([IOU_seq.argmax()]).to(spk_idx.device)  ## shape:[1]
        tolerance_flag = False
    # print(max_idx)

    mem_v = mem_seq[max_idx]
    # print(f'max_idx:{max_idx},shape:{max_idx.shape}, spk_idx:{spk_idx},shape:{spk_idx.shape}')
    # print(f'mem_seq:{mem_seq},shape:{mem_seq.shape}')
    # print(f'mem_v:{mem_v},shape:{mem_v.shape}')

    pre_mem_v = 0
    if max_idx > spk_idx:
        pre_mem_v = mem_seq[spk_idx]
        for i in range(spk_idx + 1, max_idx + 1):
            pre_mem_v = pre_mem_v * tau + cur_seq[i].clamp(min=0)
        mem_v = pre_mem_v
    up_bound_target = torch.tensor(Vth) * tau + cur_seq[max_idx].clamp(min=0).detach()
    low_bound_target = torch.tensor(Vth)
    target = alpha * up_bound_target + (1 - alpha) * low_bound_target

    # print(f'target:{target},shape:{target.shape}')
    # print(f'mem_v:{mem_v},shape:{mem_v.shape}')

    cat_I = cur_seq.clamp(max=0)  ## supervise the current to be positive
    I_loss = mse(cat_I, torch.zeros_like(cat_I))
    # print(mem_v.shape,target.shape)
    mem_loss = mse(mem_v.squeeze(), target.squeeze())
    loss = I_loss + mem_loss
    desired_idx_move = max_idx - torch.tensor([spk_idx]).to(spk_idx.device)
    # print(f'spkidx:{spk_idx},max_idx:{max_idx}')
    # print(f'IOU_seq:{IOU_seq.detach().cpu()}')
    # print(f'mem_seq:{mem_seq.squeeze().detach().cpu()}')
    # print(f'cur_seq:{cur_seq.squeeze().detach().cpu()}')
    # print(f'mem_v:{mem_v.squeeze().detach().cpu()}')
    # print(f'target:{target.squeeze().detach().cpu()}')
    # print(f'spk_seq:{spk_seq.squeeze().detach().cpu()}')
    # print(f'desired move:{desired_idx_move.squeeze().detach().cpu()}\n')

    return loss, max_idx, desired_idx_move, tolerance_flag


def main():
    y = torch.tensor([0, 0, 0, 0, 1.])
    y_for_time = y.view(y.shape[0], 1, 1)
    iou_test = torch.tensor([5, 6, 7, 8, 9, 10, 11.])
    loss = CE_temp_loss(y, iou_test)
    print(loss)


if __name__ == '__main__':
    main()

# loss = maxIOU_cn_loss(spk_seq,iou_seq)
# print(loss)
#






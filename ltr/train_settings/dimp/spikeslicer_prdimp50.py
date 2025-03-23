import torch.nn as nn
import torch.optim as optim
from ltr.dataset import Lasot, Got10k, TrackingNet, MSCOCOSeq, FE108
from ltr.data import processing, sampler, LTRLoader
from ltr.models.tracking import dimpnet
import ltr.models.loss as ltr_losses
from ltr import actors
import ltr.models.loss.kl_regression as klreg_losses
import ltr.actors.tracking as tracking_actors
from ltr.trainers import DIMP_SpikeSlicerTrainer
import ltr.data.transforms as tfm
from ltr import MultiGPU
import wandb
import os 
import torch
from SpikeSlicer_utils.SpikeSliceNet import SpikeSlicerNet_S_IF, MembraneLoss


def run(settings):
    # Most common settings are assigned in the settings struct
    settings.device = 'cuda'
    settings.description = 'TransT with default settings.'
    settings.batch_size = 10
    settings.num_workers = 10
    settings.multi_gpu = False
    settings.print_interval = 1
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]
    settings.search_area_factor = 5.0
    settings.output_sigma_factor = 1/4
    settings.target_filter_sz = 4
    settings.feature_sz = 18
    settings.train_model_name = 'prdimp50'
    settings.output_sz = settings.feature_sz * 16
    settings.center_jitter_factor = {'train': 3, 'test': 4.5}
    settings.scale_jitter_factor = {'train': 0.25, 'test': 0.5}
    settings.hinge_threshold = 0.05
    settings.print_stats = ['Loss/total', 'Loss/bb_ce', 'ClfTrain/clf_ce']

    settings.trained_tracker_checkpoint = ""
    settings.event_resolution = (260, 346)
    settings.extend_step = 2
    settings.extend_width = 1
    settings.group_num = 5
    settings.max_epoch = 50
    settings.alpha = 0.5
    settings.alpha_lr = 0.1
    settings.split_train_epoch = 10
    settings.extend_width_epoch = 25
    settings.log_name = f"prdimp50_train"
    settings.project_path = os.path.join(settings.project_path, settings.log_name)
    # Transformer
    settings.position_embedding = 'sine'
    settings.hidden_dim = 256
    settings.dropout = 0.1
    settings.nheads = 8
    settings.dim_feedforward = 2048
    settings.featurefusion_layers = 4
    wandb.init(project="SpikeSlicer", name=settings.log_name, config=settings)
    # Train datasets
    fe108_train_snn = FE108(settings.env.fe108_dir, subset='val', groupnum=settings.group_num, txt_suffix=".txt")
    fe108_train_ann = FE108(settings.env.fe108_dir, subset='train', groupnum=settings.group_num, txt_suffix=".txt")
    # The joint augmentation transform, that is applied to the pairs jointly

    # The tracking pairs processing module
    output_sigma = settings.output_sigma_factor / settings.search_area_factor
    proposal_params = {'boxes_per_frame': 128, 'gt_sigma': (0.05, 0.05), 'proposal_sigma': [(0.05, 0.05), (0.5, 0.5)]}
    label_params = {'feature_sz': settings.feature_sz, 'sigma_factor': output_sigma, 'kernel_sz': settings.target_filter_sz}
    label_density_params = {'feature_sz': settings.feature_sz, 'sigma_factor': output_sigma, 'kernel_sz': settings.target_filter_sz, 'normalize': True}
    # Data processing to do on the training pairs
    data_processing_train = processing.EKLDiMPProcessing(search_area_factor=settings.search_area_factor,
                                                        output_sz=settings.output_sz,
                                                        center_jitter_factor=settings.center_jitter_factor,
                                                        scale_jitter_factor=settings.scale_jitter_factor,
                                                        mode='sequence',
                                                        proposal_params=proposal_params,
                                                        label_function_params=label_params,
                                                        label_density_params=label_density_params,
                                                        train_mode='prdimp',
                                                        transform=None,
                                                        joint_transform=None)

    # The sampler for training
    dataset_snn = sampler.DiMPSampler([fe108_train_snn], [1],
                                        samples_per_epoch=5000, max_gap=200, num_test_frames=20, num_train_frames=20,
                                        processing=None, frame_sample_mode='continual')
    dataset_ann = sampler.DiMPSampler([fe108_train_ann], [1], samples_per_epoch=5000, max_gap=200,
                                      num_test_frames=60, num_train_frames=60, processing=None, frame_sample_mode='continual')

    # The loader for training
    loader_train_snn = LTRLoader('train', dataset_snn, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=True, drop_last=True, stack_dim=0)
    loader_train_ann = LTRLoader('train', dataset_ann, training=True, batch_size=settings.batch_size,
                                 num_workers=settings.num_workers, shuffle=True, drop_last=True, stack_dim=0)

    # a = dataset_train[0]
    # Create network and actor
    net = dimpnet.klcedimpnet50(filter_size=settings.target_filter_sz, backbone_pretrained=True, optim_iter=5,
                                clf_feat_norm=True, clf_feat_blocks=0, final_conv=True, out_feature_dim=512,
                                optim_init_step=1.0, optim_init_reg=0.05, optim_min_reg=0.05,
                                gauss_sigma=output_sigma * settings.feature_sz, alpha_eps=0.05, normalize_label=True, init_initializer='zero')
    net.load_state_dict(torch.load(settings.trained_tracker_checkpoint)['net'])
    ## IF
    snn_net = SpikeSlicerNet_S_IF(settings.event_resolution, output_num=1).to(settings.device)

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        net = MultiGPU(net, dim=0)
        snn_net = MultiGPU(snn_net, dim=0)


    objective = {'bb_ce': klreg_losses.KLRegression(reduction='none'), 'clf_ce': klreg_losses.KLRegressionGrid()}

    loss_weight = {'bb_ce': 0.0025, 'clf_ce': 0.25, 'clf_ce_init': 0.25, 'clf_ce_iter': 1.0}

    actor = tracking_actors.KLDiMPActor(net=net, objective=objective, loss_weight=loss_weight)

    # Optimizer
    ann_optimizer = optim.Adam([{'params': actor.net.classifier.parameters(), 'lr': 1e-4},
                            {'params': actor.net.bb_regressor.parameters(), 'lr': 1e-4},
                            {'params': actor.net.feature_extractor.parameters(), 'lr': 2e-6}],
                           lr=2e-5)

    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.2)
    ann_scheduler = optim.lr_scheduler.StepLR(ann_optimizer, (settings.max_epoch - settings.split_train_epoch) // 2, gamma=0.1)


    snn_optimizer = torch.optim.SGD(snn_net.parameters(), lr=1e-4)
    snn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(snn_optimizer, T_max=settings.max_epoch, eta_min=0.)
    mem_loss = MembraneLoss(alpha=settings.alpha, alpha_lr=settings.alpha_lr)

    # Create trainer
    trainer = DIMP_SpikeSlicerTrainer(actor, snn_net, [loader_train_snn], [loader_train_ann], snn_optimizer, snn_scheduler, ann_optimizer, ann_scheduler, settings, data_processing_train, mem_loss)

    # Run training (set fail_safe=False if you are debugging)
    trainer.train(settings.max_epoch, load_latest=False, fail_safe=True)

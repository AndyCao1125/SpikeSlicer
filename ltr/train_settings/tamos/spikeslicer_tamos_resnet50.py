import torch
import torch.optim as optim
from ltr.data.loader import MultiEpochLTRLoader
from ltr.dataset import Got10k, Lasot, TrackingNet, FE108
from ltr.data import processing, sampler, LTRLoader
from ltr.models.tracking import tamosnet
import ltr.models.loss as ltr_losses
import ltr.actors.tracking as actors
from ltr.trainers import LTRTrainer, TAMOS_SpikeSlicerTrainer
import ltr.data.transforms as tfm
from ltr import MultiGPU
from ltr.models.loss.bbr_loss import GIoULoss
import numpy as np
import wandb
from SpikeSlicer_utils.SpikeSliceNet import SpikeSlicerNet_S_IF, MembraneLoss, SpikeSlicerNet_B_IF
import os

def run(settings):
    settings.description = 'TaMOs-Resnet50'
    settings.multi_gpu = False
    settings.batch_size = 8
    settings.num_workers = 10
    fail_safe = True
    load_latest = False

    settings.print_interval = 10
    settings.save_checkpoint_freq = 10
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]
    settings.search_area_factor = 5.0
    settings.output_sigma_factor = 1 / 4
    settings.target_filter_sz = 1
    settings.feature_sz = (22, 17)
    settings.output_sz = (16 * settings.feature_sz[0], 16 * settings.feature_sz[1])  # w, h
    settings.center_jitter_factor = {'train': 0., 'test': 4.5}
    settings.scale_jitter_factor = {'train': 0., 'test': 0.5}
    settings.hinge_threshold = 0.05
    settings.num_encoder_layers = 6
    settings.num_decoder_layers = 6
    settings.frozen_backbone_layers = 'none'
    settings.freeze_backbone_bn_layers = True

    settings.crop_type = 'inside_major'
    settings.max_scale_change = 1.5
    settings.max_gap = 200
    settings.train_samples_per_epoch = 40000
    settings.val_samples_per_epoch = 10000
    settings.val_epoch_interval = 1
    settings.num_epochs = 50

    settings.weight_giou = 1.0
    settings.weight_clf = 100.0
    settings.normalized_bbreg_coords = True
    settings.center_sampling_radius = 1.0
    settings.use_test_frame_encoding = False  # Set to True to use the same as in the paper but is less stable to train.

    settings.event_resolution = (260, 346)
    settings.extend_step = 2
    settings.extend_width = 1
    settings.group_num = 5
    settings.alpha = 0.5
    settings.alpha_lr = 0.1
    settings.split_train_epoch = 10
    settings.extend_width_epoch = 25
    settings.train_model_name = 'tamos_resnet50'

    settings.grad_clip_max_norm = 0.1
    settings.max_num_objects = 1
    settings.move_data_to_gpu = True
    settings.log_name = f"tamos_train"
    settings.project_path = os.path.join(settings.project_path, settings.log_name)
    settings.trained_tracker_checkpoint = "/home/jiahang/jiahang/BEEF_train_tracker/pytracking/checkpoints/ltr/tamos/tamos_resnet50/TaMOsNet_ep0081.pth.tar"
    settings.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wandb.init(project="SpikeSlicer", name=settings.log_name, config=settings)
    # Train datasets
    fe108_train = FE108(settings.env.fe108_dir, subset='train', txt_suffix=".txt")

    # Validation datasets
    fe108_val = FE108(settings.env.fe108_dir, subset='val', txt_suffix=".txt")


    # Data transform


    # The tracking pairs processing module
    output_sigma = settings.output_sigma_factor / settings.search_area_factor
    label_params = {'feature_sz': settings.feature_sz, 'sigma_factor': output_sigma,
                    'kernel_sz': settings.target_filter_sz}

    data_processing_train = processing.TaMOsProcessing(max_num_objects=settings.max_num_objects,
                                                       search_area_factor=settings.search_area_factor,
                                                       output_sz=settings.output_sz,
                                                       center_jitter_factor=settings.center_jitter_factor,
                                                       scale_jitter_factor=settings.scale_jitter_factor,
                                                       crop_type=settings.crop_type,
                                                       max_scale_change=settings.max_scale_change,
                                                       mode='sequence',
                                                       label_function_params=label_params,
                                                       transform=None,
                                                       joint_transform=None,
                                                       use_normalized_coords=settings.normalized_bbreg_coords,
                                                       center_sampling_radius=settings.center_sampling_radius,
                                                       include_high_res_labels=True,
                                                       enforce_one_sample_region_per_object=True)

    data_processing_val = processing.TaMOsProcessing(max_num_objects=settings.max_num_objects,
                                                     search_area_factor=settings.search_area_factor,
                                                     output_sz=settings.output_sz,
                                                     center_jitter_factor=settings.center_jitter_factor,
                                                     scale_jitter_factor=settings.scale_jitter_factor,
                                                     crop_type=settings.crop_type,
                                                     max_scale_change=settings.max_scale_change,
                                                     mode='sequence',
                                                     label_function_params=label_params,
                                                     transform=None,
                                                     joint_transform=None,
                                                     use_normalized_coords=settings.normalized_bbreg_coords,
                                                     center_sampling_radius=settings.center_sampling_radius,
                                                     include_high_res_labels=True,
                                                     enforce_one_sample_region_per_object=True)

    # Train sampler and loader
    dataset_ann = sampler.TaMOsDatasetSampler([fe108_train], [1],
        samples_per_epoch=5000, max_gap=settings.max_gap,
        num_test_frames=60, num_train_frames=60, frame_sample_mode='continual')

    loader_train_ann = MultiEpochLTRLoader('train', dataset_ann, training=True, batch_size=settings.batch_size,
                                       num_workers=settings.num_workers,
                                       shuffle=True, drop_last=True, stack_dim=0)

    # Validation samplers and loaders
    dataset_snn = sampler.TaMOsDatasetSampler([fe108_val], [1], samples_per_epoch=settings.val_samples_per_epoch,
                                                  max_gap=settings.max_gap, num_test_frames=20,
                                                  num_train_frames=20, frame_sample_mode='continual')

    loader_train_snn = LTRLoader('train', dataset_snn, training=True, batch_size=settings.batch_size,
                               num_workers=settings.num_workers, 
                               shuffle=False, drop_last=True, epoch_interval=1, stack_dim=0)

    # a = dataset_ann[0]
    # Create network and actor
    net = tamosnet.tamosnet_resnet50(filter_size=settings.target_filter_sz, backbone_pretrained=True,
                                     head_feat_blocks=0,
                                     head_feat_norm=True, final_conv=True, out_feature_dim=256,
                                     feature_sz=settings.feature_sz,
                                     frozen_backbone_layers=settings.frozen_backbone_layers,
                                     num_encoder_layers=settings.num_encoder_layers,
                                     num_decoder_layers=settings.num_decoder_layers,
                                     head_layer=['layer2', 'layer3'],
                                     num_tokens=settings.max_num_objects, label_enc='gaussian', box_enc='ltrb_token',
                                     fpn_head_cls_output_mode=['low', 'high', 'trafo'],
                                     fpn_head_bbreg_output_mode=['low', 'high', 'trafo'])
    net.load_state_dict(torch.load(settings.trained_tracker_checkpoint)['net'])
    snn_net = SpikeSlicerNet_B_IF(settings.event_resolution, output_num=1).to(settings.device)
    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        net = MultiGPU(net, dim=1)
        snn_net = MultiGPU(snn_net, dim=0)

    objective = {'giou': GIoULoss(), 'test_clf': ltr_losses.FocalLoss()}

    loss_weight = {'giou': settings.weight_giou, 'test_clf': settings.weight_clf}

    actor = actors.TaMOsActor(net=net, objective=objective, loss_weight=loss_weight, prob=True)

    # Optimizer


    ann_optimizer = optim.AdamW([
        {'params': actor.net.head.parameters(), 'lr': 1e-4},
        {'params': actor.net.feature_extractor.layer3.parameters(), 'lr': 2e-5}
    ], lr=2e-4, weight_decay=0.0001)

    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.2)
    sche_epoch = (settings.num_epochs - settings.split_train_epoch)
    ann_scheduler = optim.lr_scheduler.MultiStepLR(ann_optimizer, milestones=[sche_epoch//3, sche_epoch * 2 // 3], gamma=0.2)



    snn_optimizer = torch.optim.SGD(snn_net.parameters(), lr=1e-4)
    snn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(snn_optimizer, T_max=settings.num_epochs, eta_min=0.)
    mem_loss = MembraneLoss(alpha=settings.alpha, alpha_lr=settings.alpha_lr)

    trainer = TAMOS_SpikeSlicerTrainer(actor, snn_net, [loader_train_snn], [loader_train_ann], snn_optimizer, snn_scheduler, ann_optimizer, ann_scheduler, settings, data_processing_train, mem_loss)
    trainer.train(settings.num_epochs, load_latest=load_latest, fail_safe=fail_safe)

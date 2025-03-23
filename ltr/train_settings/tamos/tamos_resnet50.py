import torch
import torch.optim as optim
from ltr.data.loader import MultiEpochLTRLoader
from ltr.dataset import Got10k, Lasot, TrackingNet, FE108
from ltr.data import processing, sampler, LTRLoader
from ltr.models.tracking import tamosnet
import ltr.models.loss as ltr_losses
import ltr.actors.tracking as actors
from ltr.trainers import LTRTrainer
import ltr.data.transforms as tfm
from ltr import MultiGPU
from ltr.models.loss.bbr_loss import GIoULoss
import numpy as np
import wandb

def run(settings):
    settings.description = 'TaMOs-Resnet50'
    settings.multi_gpu = False
    settings.batch_size = 10 * torch.cuda.device_count()
    settings.num_workers = 10 * torch.cuda.device_count()
    fail_safe = True
    load_latest = True

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
    settings.num_train_frames = 1
    settings.num_test_frames = 1
    settings.num_encoder_layers = 6
    settings.num_decoder_layers = 6
    settings.frozen_backbone_layers = 'none'
    settings.freeze_backbone_bn_layers = True

    settings.crop_type = 'inside_major'
    settings.max_scale_change = 1.5
    settings.max_gap = 200
    settings.train_samples_per_epoch = 40000
    settings.val_samples_per_epoch = 10000
    settings.val_epoch_interval = 5
    settings.num_epochs = 300

    settings.weight_giou = 1.0
    settings.weight_clf = 100.0
    settings.normalized_bbreg_coords = True
    settings.center_sampling_radius = 1.0
    settings.use_test_frame_encoding = False  # Set to True to use the same as in the paper but is less stable to train.

    settings.grad_clip_max_norm = 0.1
    settings.max_num_objects = 1
    settings.move_data_to_gpu = True
    settings.log_name = f"tamos_pretrain"
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
    dataset_train = sampler.TaMOsDatasetSampler(
        [fe108_train],
        [1],
        samples_per_epoch=settings.train_samples_per_epoch, max_gap=settings.max_gap,
        num_test_frames=settings.num_test_frames, num_train_frames=settings.num_train_frames,
        processing=data_processing_train)

    loader_train = MultiEpochLTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size,
                                       num_workers=settings.num_workers,
                                       shuffle=True, drop_last=True, stack_dim=1)

    # Validation samplers and loaders
    dataset_sot_val = sampler.TaMOsDatasetSampler([fe108_val], [1], samples_per_epoch=settings.val_samples_per_epoch,
                                                  max_gap=settings.max_gap, num_test_frames=settings.num_test_frames,
                                                  num_train_frames=settings.num_train_frames,
                                                  processing=data_processing_val)

    loader_sot_val = LTRLoader('val_sot', dataset_sot_val, training=False, batch_size=settings.batch_size,
                               num_workers=settings.num_workers,
                               shuffle=False, drop_last=True, epoch_interval=settings.val_epoch_interval, stack_dim=1)

    a = dataset_train[0]
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

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        net = MultiGPU(net, dim=1)

    objective = {'giou': GIoULoss(), 'test_clf': ltr_losses.FocalLoss()}

    loss_weight = {'giou': settings.weight_giou, 'test_clf': settings.weight_clf}

    actor = actors.TaMOsActor(net=net, objective=objective, loss_weight=loss_weight, prob=True)

    # Optimizer
    optimizer = optim.AdamW([
        {'params': actor.net.head.parameters(), 'lr': 1e-4},
        {'params': actor.net.feature_extractor.layer3.parameters(), 'lr': 2e-5}
    ], lr=2e-4, weight_decay=0.0001)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.2)

    trainer = LTRTrainer(actor, [loader_train, loader_sot_val], optimizer, settings, lr_scheduler,
                         freeze_backbone_bn_layers=settings.freeze_backbone_bn_layers)

    trainer.train(settings.num_epochs, load_latest=load_latest, fail_safe=fail_safe)

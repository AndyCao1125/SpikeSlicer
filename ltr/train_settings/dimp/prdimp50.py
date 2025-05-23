import torch.optim as optim
from ltr.dataset import Lasot, Got10k, TrackingNet, MSCOCOSeq, FE108
from ltr.data import processing, sampler, LTRLoader
from ltr.models.tracking import dimpnet
import ltr.models.loss as ltr_losses
import ltr.models.loss.kl_regression as klreg_losses
import ltr.actors.tracking as tracking_actors
from ltr.trainers import LTRTrainer
import ltr.data.transforms as tfm
from ltr import MultiGPU
import wandb

def run(settings):
    settings.description = 'Default train settings for PrDiMP with ResNet50 as backbone.'
    settings.batch_size = 10
    settings.num_workers = 8
    settings.multi_gpu = False
    settings.print_interval = 1
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]
    settings.search_area_factor = 5.0
    settings.output_sigma_factor = 1/4
    settings.target_filter_sz = 4
    settings.feature_sz = 18
    settings.output_sz = settings.feature_sz * 16
    settings.center_jitter_factor = {'train': 3, 'test': 4.5}
    settings.scale_jitter_factor = {'train': 0.25, 'test': 0.5}
    settings.hinge_threshold = 0.05
    settings.print_stats = ['Loss/total', 'Loss/bb_ce', 'ClfTrain/clf_ce']


    settings.log_name = "prdimp_pretrain"
    settings.project_path = settings.project_path + '/' + settings.log_name

    wandb.init(project="SpikeSlicer", name=settings.log_name, config=settings)

    # Train datasets
    fe108_train = FE108(settings.env.fe108_dir, subset='train')

    # Validation datasets
    fe108_val = FE108(settings.env.fe108_dir, subset='val')
    
    


    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    # The tracking pairs processing module
    output_sigma = settings.output_sigma_factor / settings.search_area_factor
    proposal_params = {'boxes_per_frame': 128, 'gt_sigma': (0.05, 0.05), 'proposal_sigma': [(0.05, 0.05), (0.5, 0.5)]}
    label_params = {'feature_sz': settings.feature_sz, 'sigma_factor': output_sigma, 'kernel_sz': settings.target_filter_sz}
    label_density_params = {'feature_sz': settings.feature_sz, 'sigma_factor': output_sigma, 'kernel_sz': settings.target_filter_sz, 'normalize': True}

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

    data_processing_val = processing.EKLDiMPProcessing(search_area_factor=settings.search_area_factor,
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

    # Train sampler and loader
    dataset_train = sampler.DiMPSampler([fe108_train], [1],
                                        samples_per_epoch=26000, max_gap=200, num_test_frames=3, num_train_frames=3,
                                        processing=data_processing_train)

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=True, drop_last=True, stack_dim=1)

    # Validation samplers and loaders
    dataset_val = sampler.DiMPSampler([fe108_val], [1], samples_per_epoch=5000, max_gap=200,
                                      num_test_frames=3, num_train_frames=3,
                                      processing=data_processing_val)

    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=settings.batch_size, num_workers=settings.num_workers,
                           shuffle=False, drop_last=True, epoch_interval=5, stack_dim=1)

    # Create network and actor
    net = dimpnet.klcedimpnet50(filter_size=settings.target_filter_sz, backbone_pretrained=True, optim_iter=5,
                                clf_feat_norm=True, clf_feat_blocks=0, final_conv=True, out_feature_dim=512,
                                optim_init_step=1.0, optim_init_reg=0.05, optim_min_reg=0.05,
                                gauss_sigma=output_sigma * settings.feature_sz, alpha_eps=0.05, normalize_label=True, init_initializer='zero')

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        net = MultiGPU(net, dim=1)

    objective = {'bb_ce': klreg_losses.KLRegression(), 'clf_ce': klreg_losses.KLRegressionGrid()}

    loss_weight = {'bb_ce': 0.0025, 'clf_ce': 0.25, 'clf_ce_init': 0.25, 'clf_ce_iter': 1.0}

    actor = tracking_actors.KLDiMPActor(net=net, objective=objective, loss_weight=loss_weight)

    # Optimizer
    optimizer = optim.Adam([{'params': actor.net.classifier.parameters(), 'lr': 1e-3},
                            {'params': actor.net.bb_regressor.parameters(), 'lr': 1e-3},
                            {'params': actor.net.feature_extractor.parameters(), 'lr': 2e-5}],
                           lr=2e-4)

    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.2)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.1)

    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    # trainer.train(50, load_latest=True, fail_safe=True)
    trainer.train(100, load_latest=True, fail_safe=True)

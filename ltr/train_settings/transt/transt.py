import torch
from ltr.dataset import Lasot, MSCOCOSeq, Got10k, TrackingNet, FE108
from ltr.data import processing, sampler, LTRLoader
import ltr.models.tracking.transt as transt_models
from ltr import actors
from ltr.trainers import LTRTrainer
import ltr.data.transforms as tfm
from ltr import MultiGPU
from SpikeSlicer_utils.quantization_layer import QuantizationLayer
import time
import wandb

def run(settings):
    # Most common settings are assigned in the settings struct
    settings.device = 'cuda'
    settings.description = 'TransT with default settings.'
    settings.batch_size = 20
    settings.num_workers = 6
    settings.multi_gpu = False
    settings.print_interval = 1
    settings.event_resolution = (260, 346)
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
    settings.log_name = "transt_pretrain"
    settings.event_representation = "frame"
    settings.project_path = settings.project_path + '/' + settings.log_name
    # Transformer
    settings.position_embedding = 'sine'
    settings.hidden_dim = 256
    settings.dropout = 0.1
    settings.nheads = 8
    settings.dim_feedforward = 2048
    settings.featurefusion_layers = 4
    wandb.init(project="SpikeSlicer", name=settings.log_name, config=settings)
    # Train datasets
    fe108_train = FE108(settings.env.fe108_dir, txt_suffix="_all.txt")
    # The joint augmentation transform, that is applied to the pairs jointly

    # The augmentation transform applied to the training set (individually to each image in the pair)
    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2))

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
                                                      train_mode='transt')

    # The sampler for training
    quantization_layer = QuantizationLayer(settings.event_representation, settings.event_resolution)#.to(settings.device)
    quantization_layer = None
    dataset_train = sampler.TransTSampler([fe108_train], [1],
                                samples_per_epoch=400*settings.batch_size, max_gap=100, processing=data_processing_train, quantization_layer=quantization_layer)
    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=True, drop_last=True, stack_dim=0)

    # Create network and actor
    model = transt_models.transt_resnet50(settings)

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        model = MultiGPU(model, dim=0)

    objective = transt_models.transt_loss(settings)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    actor = actors.TranstActor(net=model, objective=objective)

    # Optimizer
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": 1e-5,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=1e-4,
                                  weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.1)

    # Create trainer
    trainer = LTRTrainer(actor, [loader_train], optimizer, settings, lr_scheduler)

    # Run training (set fail_safe=False if you are debugging)
    trainer.train(100, load_latest=False, fail_safe=True)

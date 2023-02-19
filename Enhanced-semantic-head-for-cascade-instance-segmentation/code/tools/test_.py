from mmdet.models.builder import build_head
from mmdet.datasets import build_dataset, build_dataloader
from mmcv.runner import build_runner,build_optimizer,DistSamplerSeedHook,OptimizerHook,EvalHook
from mmcv.parallel import MMDataParallel
from mmdet.utils import get_root_logger
from mmcv.parallel.scatter_gather import scatter_kwargs
from mmcv import Config
from mmdet.models import build_detector

import torch


if __name__ == '__main__':
    dataset_type = 'CocoDataset'
    data_root = '/home/hxr/new_work/data/coco/'
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
    ]
    test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
    ]
    samples_per_gpu = 1
    feat_channels = 8
    fusion_level =2
    
    with_dyn_params = False
    in_channels = 256
    
    data_cfg = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=2,    
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
    
    num_stages = 6
    num_proposals = 100
    
    
    checkpoint_config = dict(interval=1)
    log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
    runner_cfg = dict(type='EpochBasedRunner', max_epochs=12)
    
    
    # optimizer
    optimizer = dict(type='AdamW', lr=0.000025, weight_decay=0.0001)
    optimizer_config = dict( grad_clip=dict(max_norm=1, norm_type=2))
    # learning policy
    lr_config = dict(policy='step', step=[8, 11], warmup_iters=1000)

    
    data_set = build_dataset(data_cfg['train'])
    data_loader = build_dataloader(
            data_set,
            2,
            2,
            # `num_gpus` will be ignored if distributed
            dist=True,
            num_gpus=1,
            seed=None,)
    config_file = '/home/hxr/fusion6/mmdetection-2.18.1/configs/solo/solo_r50_fpn_1x_coco.py'
    cfg = Config.fromfile(config_file)
    # model0 = build_head(cfg.model)
    
    model0 = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model0.init_weights() 
    model = MMDataParallel(model0.cuda(0), device_ids=[0])
    
    logger = get_root_logger(log_level='INFO')
    optimizer = build_optimizer(model, optimizer)
    runner_default_args=dict(
        model=model,
        optimizer=optimizer,
        logger=logger,
        work_dir = './work_dirs_',
        meta=None)
    runner = build_runner(
        runner_cfg,
        default_args=runner_default_args)
    
    
    
    val_dataset = build_dataset(data_cfg['val'], dict(test_mode=True))
    val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=2,
            dist=True,
            num_gpus=1,
            shuffle=False)
    eval_cfg = dict(metric=['bbox', 'segm'])
    
        
    # runner.register_hook(DistSamplerSeedHook())
    # register hooks
    runner.register_training_hooks(lr_config, OptimizerHook(**optimizer_config),
                                   checkpoint_config,log_config,
                                   None)
    runner.register_hook(EvalHook(val_dataloader, **eval_cfg))
    # runner.run([data_loader],[('train', 1)])
    # runner.call_hook('after_train_epoch')
    optim = torch.optim.Adam(model.parameters(), lr = 0.5e-2)
    for i, data_batch in enumerate(data_loader):
        # print(data_batch.keys())
        input, _ = scatter_kwargs(data_batch,None,[0],0)  #(inputs, kwargs, device_ids, dim=self.dim)
        # out = model.train_step(input[0],None)
        # optim.zero_grad()
        
        # out['loss'].backward()
        # optim.step()
        # print(i, out)
        out = model0.simple_test(input[0]['img'], input[0]['img_metas'])
    # for i, data_batch in enumerate(val_dataloader):
    #     print(data_batch.keys())
    #     input, kwargs = scatter_kwargs(data_batch,None,[0],0)  #(inputs, kwargs, device_ids, dim=self.dim)
    #     print(input[0].keys())
    #     print(len(input[0]))
    #     out = model0.simple_test(**input[0])
    #     break
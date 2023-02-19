from mmdet.models.builder import build_head
from mmdet.datasets import build_dataset, build_dataloader
from mmcv.runner import build_runner,build_optimizer,DistSamplerSeedHook,OptimizerHook,EvalHook
from mmcv.parallel import MMDataParallel
from mmdet.utils import get_root_logger
from mmcv.parallel.scatter_gather import scatter_kwargs
from mmcv import Config
from mmdet.models import build_detector
import torch
from thop import profile,clever_format
from tqdm import tqdm

if __name__ == '__main__':
    dataset_type = 'CocoDataset'
    data_root = '/home/hxr/QueryInst/data/coco/'
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

    
    
    # optimizer = dict(type='Adam', lr=0.0005, weight_decay=0.0001)
    # optimizer_config = dict(grad_clip=None)
    # lr_config = dict(
    #     policy='step',
    #     warmup='linear',
    #     warmup_iters=500,
    #     warmup_ratio=0.001,
    #     step=[8, 11])   
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
    config_file = '/home/hxr/fusion6/mmdetection-2.18.1/configs/htc/htc_r50_fpn_1x_coco.py'
    cfg = Config.fromfile(config_file)
    # model0 = build_head(cfg.model)
    
    model0 = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model0.init_weights() 
    model0 = model0.cuda(6)
    
    logger = get_root_logger(log_level='INFO')

    
    
    val_dataset = build_dataset(data_cfg['val'], dict(test_mode=True))
    val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=2,
            workers_per_gpu=2,
            dist=True,
            num_gpus=1,
            shuffle=False)
    eval_cfg = dict(metric=['bbox', 'segm'])
    
        
    # runner.register_hook(DistSamplerSeedHook())
    # register hooks
    
    
    import time
    n = 2000
    total = 0
    model0.eval()
    with torch.no_grad():
        for i, data_batch in enumerate(tqdm(data_loader)):
            # print(data_batch.keys())
            input, _ = scatter_kwargs(data_batch,None,[6],0)  #(inputs, kwargs, device_ids, dim=self.dim)
            # flops,params = profile(model0, inputs = ([input[0]['img']], [input[0]['img_metas']],False))
            # macs, params = clever_format([flops, params], "%.3f")
            # print(macs, params)
            # break
            
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            # torch.cuda.synchronize()
            out = model0.simple_test(input[0]['img'], input[0]['img_metas'])
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)/1000
            total += curr_time
        # for i, data_batch in enumerate(val_dataloader):
        #     print(data_batch.keys())
        #     input, kwargs = scatter_kwargs(data_batch,None,[0],0)  #(inputs, kwargs, device_ids, dim=self.dim)
        #     print(input[0].keys())
        #     print(len(input[0]))
            # out = model0.simple_test(**input[0])
        #     break
            if i == n-1:
                break
    
    print(n*2/(total))
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=8,
        dataset=dict(
            type='CityscapesDataset',
            ann_file=
            'data/cityscapes/annotations/instancesonly_filtered_gtFine_train.json',
            img_prefix='data/cityscapes/leftImg8bit/train/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
                dict(
                    type='Resize',
                    img_scale=[(2048, 800), (2048, 1024)],
                    keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(
                    type='Collect',
                    keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
            ]),
        seg_prefix='data/cityscapes/stuffthingmaps/train/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True,
                with_seg=True),
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='SegRescale', scale_factor=0.125),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=[
                    'img', 'gt_bboxes', 'gt_labels', 'gt_masks',
                    'gt_semantic_seg'
                ])
        ]),
    val=dict(
        type='CityscapesDataset',
        ann_file=
        'data/cityscapes/annotations/instancesonly_filtered_gtFine_val.json',
        img_prefix='data/cityscapes/leftImg8bit/val/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip', flip_ratio=0.5),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CityscapesDataset',
        ann_file=
        'data/cityscapes/annotations/instancesonly_filtered_gtFine_test.json',
        img_prefix='data/cityscapes/leftImg8bit/test/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip', flip_ratio=0.5),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))

train_cfg = dict(
        type='RepeatDataset',
        times=8,
        dataset=dict(
            type='CityscapesDataset',
            ann_file=
            'data/cityscapes/annotations/instancesonly_filtered_gtFine_train.json',
            img_prefix='data/cityscapes/leftImg8bit/train/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
                dict(
                    type='Resize',
                    img_scale=[(2048, 800), (2048, 1024)],
                    keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(
                    type='Collect',
                    keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
            ]),
        seg_prefix='data/cityscapes/stuffthingmaps/train/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True,
                with_seg=True),
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='SegRescale', scale_factor=0.125),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=[
                    'img', 'gt_bboxes', 'gt_labels', 'gt_masks',
                    'gt_semantic_seg'
                ])
        ])

data = build_dataset(train_cfg)
print(data[1])
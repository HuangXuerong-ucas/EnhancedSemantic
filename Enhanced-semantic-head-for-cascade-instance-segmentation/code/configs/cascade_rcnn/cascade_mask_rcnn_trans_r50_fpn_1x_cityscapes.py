model = dict(
    type='CascadeRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='SCascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=8,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=8,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=8,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ],
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=8,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
        semantic_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[8]),
        semantic_head=dict(
            type='FusedTransformerSemanticHead',
            fusion_head=dict(
                type='FusionHead',
                num_ins=5,
                fusion_level=1,
                in_channels=256,
                conv_out_channels=256),
            transformer_head=dict(
                type='TransformerSemanticHead',
                inchannels=256,
                outchannels=256),
            fcn_head=dict(
                type='FCNHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256),
            cls_class=8,
            cls_loss_weight=3.0,
            loss_seg=dict(
                type='CrossEntropyLoss', ignore_index=255, loss_weight=0.2),
            conv_channels=256,
            num_classes=19)),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))
# dataset_type = 'CityscapesDataset'
# data_root = 'data/cityscapes/'
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=True),
#     dict(type='Resize', img_scale=(2048, 1024), keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(
#         type='Normalize',
#         mean=[123.675, 116.28, 103.53],
#         std=[58.395, 57.12, 57.375],
#         to_rgb=True),
#     dict(type='Pad', size_divisor=32),
#     dict(type='SegRescale', scale_factor=0.125),
#     dict(type='DefaultFormatBundle'),
#     dict(
#         type='Collect',
#         keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg'])
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(2048, 1024),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip', flip_ratio=0.5),
#             dict(
#                 type='Normalize',
#                 mean=[123.675, 116.28, 103.53],
#                 std=[58.395, 57.12, 57.375],
#                 to_rgb=True),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img'])
#         ])
# ]
# data = dict(
#     samples_per_gpu=1,
#     workers_per_gpu=2,
#     train=dict(
#         type='RepeatDataset',
#         times=8,
#         dataset=dict(
#             type='CityscapesDataset',
#             ann_file=
#             'data/cityscapes/annotations/instancesonly_filtered_gtFine_train.json',
#             img_prefix='data/cityscapes/leftImg8bit/train/',
#             seg_prefix='data/cityscapes/stuffthingmaps/train/',
#             pipeline=[
#                 dict(type='LoadImageFromFile'),
#                 dict(
#                     type='LoadAnnotations',
#                     with_bbox=True,
#                     with_mask=True,
#                     with_seg=True),
#                 dict(
#                     type='Resize', img_scale=[(2048, 800), (2048, 1024)], keep_ratio=True),
#                 dict(type='RandomFlip', flip_ratio=0.5),
#                 dict(
#                     type='Normalize',
#                     mean=[123.675, 116.28, 103.53],
#                     std=[58.395, 57.12, 57.375],
#                     to_rgb=True),
#                 dict(type='Pad', size_divisor=32),
#                 dict(type='SegRescale', scale_factor=0.125),
#                 dict(type='DefaultFormatBundle'),
#                 dict(
#                     type='Collect',
#                     keys=[
#                         'img', 'gt_bboxes', 'gt_labels', 'gt_masks',
#                         'gt_semantic_seg'
#                     ])
#             ]),
#         ),
#     val=dict(
#         type='CityscapesDataset',
#         ann_file=
#         'data/cityscapes/annotations/instancesonly_filtered_gtFine_val.json',
#         img_prefix='data/cityscapes/leftImg8bit/val/',
#         pipeline=[
#             dict(type='LoadImageFromFile'),
#             dict(
#                 type='MultiScaleFlipAug',
#                 img_scale=(2048, 1024),
#                 flip=False,
#                 transforms=[
#                     dict(type='Resize', keep_ratio=True),
#                     dict(type='RandomFlip', flip_ratio=0.5),
#                     dict(
#                         type='Normalize',
#                         mean=[123.675, 116.28, 103.53],
#                         std=[58.395, 57.12, 57.375],
#                         to_rgb=True),
#                     dict(type='Pad', size_divisor=32),
#                     dict(type='ImageToTensor', keys=['img']),
#                     dict(type='Collect', keys=['img'])
#                 ])
#         ]),
#     test=dict(
#         type='CityscapesDataset',
#         ann_file=
#         'data/cityscapes/annotations/instancesonly_filtered_gtFine_test.json',
#         img_prefix='data/cityscapes/leftImg8bit/test/',
#         pipeline=[
#             dict(type='LoadImageFromFile'),
#             dict(
#                 type='MultiScaleFlipAug',
#                 img_scale=(2048, 1024),
#                 flip=False,
#                 transforms=[
#                     dict(type='Resize', keep_ratio=True),
#                     dict(type='RandomFlip', flip_ratio=0.5),
#                     dict(
#                         type='Normalize',
#                         mean=[123.675, 116.28, 103.53],
#                         std=[58.395, 57.12, 57.375],
#                         to_rgb=True),
#                     dict(type='Pad', size_divisor=32),
#                     dict(type='ImageToTensor', keys=['img']),
#                     dict(type='Collect', keys=['img'])
#                 ])
#         ]))
# evaluation = dict(metric=['bbox', 'segm'])
dataset_type = 'CityscapesDataset'
data_root = 'data/cityscapes/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True,with_seg=True),
    dict(
        type='Resize', img_scale=[(2048, 800), (2048, 1024)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='SegRescale', scale_factor=0.125),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks','gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
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
data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=8,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root +
            'annotations/instancesonly_filtered_gtFine_train.json',
            img_prefix=data_root + 'leftImg8bit/train/',
            
            seg_prefix='data/cityscapes/stuffthingmaps/train/',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root +
        'annotations/instancesonly_filtered_gtFine_val.json',
        img_prefix=data_root + 'leftImg8bit/val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root +
        'annotations/instancesonly_filtered_gtFine_val.json',
        img_prefix=data_root + 'leftImg8bit/val/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric=['bbox', 'segm'])

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = './work_dirs/cascade_mask_rcnn_r50_fpn_1x_cityscapes_new2'


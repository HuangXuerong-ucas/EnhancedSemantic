import torch.nn as nn
from mmdet.datasets import build_dataset, build_dataloader
from mmcv.parallel.scatter_gather import scatter_kwargs

class Model(nn.Module):
    # Our model

    def __init__(self):
        super(Model, self).__init__()
        # self.fc = nn.Linear(input_size, output_size)
        self.count = dict(coco=[0,0,0], coco_val = [0,0,0], cityscape=[0,0,0])

    def forward(self, data_batch, name):
        img_metas = data_batch['img_metas'][0]
        # size = 
        # print(img_metas['ori_shape'], img_metas['scale_factor'])

        for gt_masks in data_batch['gt_masks']:
            areas = gt_masks.areas
            for area in areas:
                if area < 32**2:
                    self.count[name][0] = self.count[name][0] + 1
                elif area < 96**2:
                    self.count[name][1] = self.count[name][1] + 1
                elif area < 1e5**2:
                    self.count[name][2] = self.count[name][2] + 1
        return


coco_cfg = dict(
        type='CocoDataset',
        ann_file='/home/hxr/fusion6/mmdetection-2.18.1/data/coco/annotations/instances_train2017.json',
        img_prefix='/home/hxr/fusion6/mmdetection-2.18.1/data/coco/train2017/',
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
        ],
        seg_prefix='/home/hxr/fusion6/mmdetection-2.18.1/data/coco/stuffthingmaps/train2017/')

coco_val_cfg = dict(
        type='CocoDataset',
        ann_file='/home/hxr/fusion6/mmdetection-2.18.1/data/coco/annotations/instances_val2017.json',
        img_prefix='/home/hxr/fusion6/mmdetection-2.18.1/data/coco/val2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True,),
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
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
                keys=[
                    'img', 'gt_bboxes', 'gt_labels', 'gt_masks',
                ])
        ],)


cityscapes_cfg = dict(
        type='RepeatDataset',
        times=8,
        dataset=dict(
            type='CityscapesDataset',
            ann_file=
            '/home/hxr/fusion6/mmdetection-2.18.1/data/cityscapes/annotations/instancesonly_filtered_gtFine_train.json',
            img_prefix='/home/hxr/fusion6/mmdetection-2.18.1/data/cityscapes/leftImg8bit/train/',
            seg_prefix='/home/hxr/fusion6/mmdetection-2.18.1/data/cityscapes/stuffthingmaps/train/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(
                    type='LoadAnnotations',
                    with_bbox=True,
                    with_mask=True,
                    with_seg=True),
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
                dict(type='SegRescale', scale_factor=0.125),
                dict(type='DefaultFormatBundle'),
                dict(
                    type='Collect',
                    keys=[
                        'img', 'gt_bboxes', 'gt_labels', 'gt_masks',
                        'gt_semantic_seg'
                    ])
            ])
)

coco_data_loader = build_dataloader(
    build_dataset(coco_cfg,),
    samples_per_gpu=30,
    workers_per_gpu=30,
    dist=True,
    num_gpus=1,
    shuffle=False
)
coco_val_data_loader = build_dataloader(
    build_dataset(coco_val_cfg),
    samples_per_gpu=30,
    workers_per_gpu=30,
    dist=True,
    num_gpus=1,
    shuffle=False
)

# cityscapes_data_loader = build_dataloader(
#     build_dataset(cityscapes_cfg),
#     samples_per_gpu=30,
#     workers_per_gpu=30,
#     dist=True,
#     num_gpus=1,
#     shuffle=False
# )


model = Model()
model = nn.DataParallel(model)
model = model.to(device='cuda')

names = ['coco_val']
data_loder = [coco_val_data_loader]
for name, loader in zip(names, data_loder):
    for i, data_batch in enumerate(loader):
        # self.areaRng = [[0**2, 1e5**2], [0**2, 32**2], [32**2, 96**2],[96**2, 1e5**2]]
        # img_metas = data_batch['img_metas'].data[0][0]
        # # size = 
        # # print(img_metas['ori_shape'], img_metas['scale_factor'])
        # gt_masks = data_batch['gt_masks'].data[0][0]
        # areas = gt_masks.areas
        # for area in areas:
        #     if area < 32**2:
        #         count[name][0] = count[name][0]+1
        #     elif area < 96**2:
        #         count[name][1] = count[name][1] + 1
        #     else:
        #         count[name][2] = count[name][2] + 1
        input, _ = scatter_kwargs(data_batch,None,[0],0)
        model(input[0],name)
        print(i, model.module.count)
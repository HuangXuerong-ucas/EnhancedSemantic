_base_ = './htc_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
# learning policy
lr_config = dict(step=[16, 19])
runner = dict(type='EpochBasedRunner', max_epochs=20)
data_root = 'data/coco/'
data = dict(
    test=dict(
        ann_file=data_root + 'annotations/image_info_test2017.json',
        img_prefix=data_root + 'test2017/',))

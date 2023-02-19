_base_ = './scnet_r50_fpn_20e_coco.py'
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                       checkpoint='torchvision://resnet101')),
    roi_head=dict(
        semantic_head=dict(
            _delete_ = True,
            type='FusedTransformerSemanticHead',
            fusion_head = dict(
                type = 'FusionHead',
                num_ins = 5,
                fusion_level = 1,
                in_channels=256,
                conv_out_channels=256,
            ),
            transformer_head=dict(
                type='TransformerSemanticHead',
                inchannels=256,
                outchannels=256,
                ),
            fcn_head=dict(
                type='FCNHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                ),
            cls_class = 80,
            cls_loss_weight = 0.0,
            loss_seg=dict(
                type='CrossEntropyLoss', ignore_index=255, loss_weight=0.2),
            conv_channels=256,
            num_classes=183),
        ))
runner = dict(type='EpochBasedRunner', max_epochs=48)

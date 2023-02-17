_base_ = './htc_r50_fpn_1x_cityscapes.py'
model = dict(
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
                in_channels=256,
                out_channels=256,
                ),
            fcn_head=dict(
                type='FCNHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                ),
            cls_class = 8,
            cls_loss_weight = 3.0,
            loss_seg=dict(
                type='CrossEntropyLoss', ignore_index=255, loss_weight=0.2),
            conv_channels=256,
            num_classes=33)
            ),
    )


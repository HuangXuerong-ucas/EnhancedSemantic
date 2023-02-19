
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmdet.models.builder import HEADS, build_loss

@HEADS.register_module()
class FusionHead(BaseModule):
    def __init__(self, num_ins, fusion_level=1, in_channels = 256, conv_out_channels = 256):
        super(FusionHead, self).__init__()
        self.num_ins = num_ins
        self.fusion_level = fusion_level
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels

        self.lateral_convs = nn.ModuleList()
        for i in range(self.num_ins):
            self.lateral_convs.append(
                ConvModule(
                    self.in_channels,
                    self.in_channels,
                    1,
                    inplace=False))
    @auto_fp16()
    def forward(self, feats):
        x = self.lateral_convs[self.fusion_level](feats[self.fusion_level])
        fused_size = tuple(x.shape[-2:])
        for i, feat in enumerate(feats):
            if i != self.fusion_level:
                feat = F.interpolate(
                    feat, size=fused_size, mode='bilinear', align_corners=True)
                x += self.lateral_convs[i](feat)
        return x


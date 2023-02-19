
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.utils import ResLayer, SimplifiedBasicBlock

@HEADS.register_module()
class FCNHead(BaseModule):
    def __init__(self, num_convs=4, in_channels = 256, conv_out_channels = 256):
        super(FCNHead, self).__init__()
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels

        self.convs = nn.ModuleList()
        for _ in range(self.num_convs-1):
            self.convs.append(
                ConvModule(
                    self.in_channels,
                    self.in_channels,
                    3,
                    padding = 1,
                ))
        self.convs.append(
                ConvModule(
                    self.in_channels,
                    self.conv_out_channels,
                    3,
                    padding = 1
                ))
        # self.num_res_blocks = self.num_convs // 2
        # self.convs = ResLayer(
        #     SimplifiedBasicBlock,
        #     self.in_channels,
        #     self.conv_out_channels,
        #     self.num_res_blocks)
    @auto_fp16()
    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


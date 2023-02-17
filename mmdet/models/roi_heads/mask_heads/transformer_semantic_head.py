import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# from mmdet.models import  build_head, build_backbone,build_neck
from mmdet.models.builder import HEADS, build_loss,build_head
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmcv.cnn import ConvModule

@HEADS.register_module()
class PositionEmbeddingXY(BaseModule):
    """
        Class for position embedding. Given lengths of cordinates x and y, return the
        embeddings.
        The process of computing position embeddings is:
            
        where 
    """
    def __init__(self):
        super().__init__()
    
    def get_position_angle_vec(self, temperature, dim):
        '''
        This function computes the position angle vector, which is a vector of
        size dim that is used to compute the position embeddings. The formula
        for the position angle vector is:
        
        [position / np.power(temperature, 2 * (hid_j // 2) / dim) for hid_j in range(dim)]
        
        Args:
            temperature (float): Temperature parameter for the position angle vector
            dim (int): Dimensionality of the position angle vector
            
        Returns:
            torch.FloatTensor: The position angle vector
        '''
        return torch.FloatTensor([1 / np.power(temperature, 2 * (i // 2) / dim) for i in range(dim)])

    def forward(self, b, x_len = 32 , y_len = 32, embed_dim = 256 , temperature=10000):
        '''
        This function computes the position embeddings for a batch of images. The position
        embeddings are computed using a sine/cosine function that takes the cumulative sum
        of a binary mask along the x and y axes. The resulting embeddings are concatenated
        along the channel dimension and returned.

        Args:
            b (int): Batch size
            x_len (int): Length of the x-axis
            y_len (int): Length of the y-axis
            embed_dim (int): Dimensionality of the position embeddings
            temperature (float): Temperature parameter for the position angle vector

        Returns:
            torch.Tensor: Position embeddings of size [b, embed_dim, x_len, y_len]
        '''
        mask = torch.ones(b, x_len, y_len)
        x_embed = mask.cumsum(0,dtype = torch.float32) 
        y_embed = mask.cumsum(1,dtype = torch.float32)
        pos_x = x_embed[:,:,:,None] / self.get_position_angle_vec(temperature,embed_dim // 2)  #N,h,w,128
        pos_y = y_embed[:,:,:,None] / self.get_position_angle_vec(temperature,embed_dim // 2)
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)  #N,H,W,128
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2).contiguous() #N, 256, H, W
        return pos
    
@HEADS.register_module()
class TransformerEncoderLayer(BaseModule):
    """
    A single layer of the Transformer encoder.
    
    Args:
        embed_dim (int): The number of expected features in the input.
        num_heads (int): The number of heads in the multiheadattention models.
        inner_dim (int): The dimensionality of the "inner" (often smaller) layer in the 
        feedforward network.
        dropout (float, optional): The dropout probability. Default: 0.1.
    """
    def __init__(self,embed_dim, num_heads, inner_dim, dropout = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.inner_dim = inner_dim
        self.dropout = dropout
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        self.linear1 = nn.Linear(self.embed_dim, self.inner_dim)
        self.linear2 = nn.Linear(self.inner_dim, self.embed_dim)
        self.activation = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.attns = nn.MultiheadAttention(self.embed_dim, self.num_heads, self.dropout)
    

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, channels, height, width).
        Returns:
            torch.Tensor: The output tensor of the same shape as the input tensor.
        """
        b, c, h, w = x.size()
        x = x.view(b,c,-1).permute(2, 0, 1).contiguous() # b, c, hw -> hw, b, c:len, b, dim

        q = k = v = x
        out, _ = self.attns(q, k, v)
        out = x + self.dropout1(out)  
        x = self.norm1(out)
        
        out = self.linear2(self.activation(self.linear1(x)))
        out = x + self.dropout2(out)
        out = self.norm2(out)
        out = out.permute(1, 2, 0).contiguous().view(b, c, h, w) 

        return out

@HEADS.register_module()
class TransformerEncoder(BaseModule):
    """
    The Transformer encoder, which consists of multiple TransformerEncoderLayers.
    
    Args:
        embed_dim (int, optional): The dimension of the position embedding. Default: 256.
        num_heads (int, optional): The number of heads in the multiheadattention models. Default: 8.
        inner_dim (int, optional): The dimensionality of the "inner" (often smaller) layer in the 
        feedforward network. Default: 1024.
        num_layers (int, optional): The number of TransformerEncoderLayers in the encoder. Default: 6.
        dropout (float, optional): The dropout probability. Default: 0.1.
    """

    def __init__(self,embed_dim = 256, num_heads=8,inner_dim=1024, num_layers=6, dropout = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.inner_dim = inner_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.postion_embedding = PositionEmbeddingXY()  #add position embedding to each multihead layer
        for _ in range(self.num_layers):
            self.layers.append(TransformerEncoderLayer(self.embed_dim,self.num_heads,self.inner_dim,self.dropout))

    def forward(self,x):
        """
        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The output tensor of the same shape as the input tensor.
        """
        b, c, h, w = x.size()
        pos = self.postion_embedding(b, h, w, self.embed_dim).to(x.device)
        x = x + pos
        for layer in self.layers:
            x = layer(x) 
        return x
        
@HEADS.register_module()
class TransformerSemanticHead(BaseModule):
    """
    The semantic head of the transformer-based semantic segmentation detection model.
    
    Args:
        in_channels (int): The number of channels in the input feature map.
        out_channels (int, optional): The number of output channels. Default: 256.
        num_grids (int, optional): The number of grids in the feature map. Default: 36.
        num_layers (int, optional): The number of TransformerEncoderLayers in the encoder. 
        Default: 6.
    """
    def __init__(self,in_channels, out_channels = 256, num_grids = 36, num_layers = 6):
        super(TransformerSemanticHead,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.transformer = TransformerEncoder(self.out_channels, 8, 784, num_layers) 
        self.num_grids = num_grids
        
    @auto_fp16()
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, num_grids^2, out_channels).
        """
        w,h = x.shape[-2:]
        x = F.interpolate(x, size=(self.num_grids,self.num_grids), mode='bilinear', align_corners=True)
        x = self.transformer(x)
        return x
        
@HEADS.register_module()
class FusedTransformerSemanticHead(BaseModule):
    """
    The fused semantic head of the enhanced semantic segmentation detection model, witch combines the
    transformer head and FCN head to extract the enhanced semantic information.
    
    Args:
        fusion_head (FusionHead): The fusion module of the input multi-scale images features.
        transformer_head (TransformerSemanticHead, optional): The Transformer head. Default: None.
        fcn_head (FCNHead, optional): The FCN head. Default: None.
        loss_seg (dict, optional): The loss function of the semantic segmentation. Default:
        None. 
        with_fcn (bool, optional): Whether to use FCN head. Defaule: True.
        with_trans (bool, optional): Whether to use transformer head. Default: True.
        with_cls (bool, optional): Whether to use multi-classification on semantic segmentation head. 
        Default: None.
        num_grids (int, optional): The number of grids in the feature map. Default:
        36. 
        cls_loss_weight (FusionHead): The loss weight of the multi-classification. Default: 2.0.
        conv_channels (int, optional): The number of channels of channel-reduction input. Default: 256.
        num_classes (int, optional): The number of semantic segmentation categories. Default: 183.
        cls_class (int, optional): The number of categories for multi-categorization. Default: 80.
        None. 
    """
    
    def __init__(self, fusion_head, transformer_head=None, fcn_head=None, loss_seg=None, with_fcn=True, with_trans=True, with_cls=True, num_grids = 36, cls_loss_weight = 2.0, conv_channels=256, num_classes = 183, cls_class = 80, init_cfg=dict(
                     type='Kaiming', override=dict(name='conv_logits'))):
        super(FusedTransformerSemanticHead,self).__init__(init_cfg)
        self.fusion_head = build_head(fusion_head)
        self.with_fcn = with_fcn
        self.with_trans = with_trans
        self.with_cls = with_cls
        self.num_classe = num_classes
        self.cls_class = cls_class
        self.conv_channels = conv_channels
        self.cls_loss_weight = cls_loss_weight
        self.num_grids = num_grids
        
        if self.with_trans:
            self.transformer_head = build_head(transformer_head)
        if self.with_fcn:
            self.fcn_head = build_head(fcn_head)
            
        if self.with_cls:
            self.reduce_conv = nn.Sequential(
                ConvModule(
                    self.conv_channels,
                    int(self.conv_channels/2),
                    3,
                    padding = 1
                ),
                ConvModule(
                    int(self.conv_channels/2),
                    1,
                    1
                ),
            )
            self.fc = nn.Linear(self.num_grids * self.num_grids, self.cls_class)
            self.criterion_cls = nn.BCEWithLogitsLoss()
            
        if self.with_fcn and self.with_trans:
            self.fused_embedding = ConvModule(
                    self.conv_channels,
                    self.conv_channels,
                    3,
                    padding = 1,
                )
        self.criterion = build_loss(loss_seg)

        self.conv_logits = nn.Conv2d(self.conv_channels, self.num_classe, 1)

    @auto_fp16()
    def forward(self, x):
        """
        Args:
            x (List(torch.Tensor)): The input tensor list, each with shape (batch_size, in_channels, height, width).
        """
        x = self.fusion_head(x)
        if self.with_trans:
            trm_feat = self.transformer_head(x)
        if self.with_fcn:
            fcn_feat = self.fcn_head(x)

        if self.with_trans and self.with_cls:
            class_scores = self.fc(self.reduce_conv(trm_feat).view(x.shape[0], -1))

        if self.with_trans and self.with_fcn:
            # upsample the trm_feat to the same size as the input x, and then add it with fcn_feat to get add_feat
            trm_feat = F.interpolate(trm_feat, size=tuple(x.shape[-2:]), mode='bilinear', align_corners=True)  #upsample
            add_feat = torch.add(trm_feat, fcn_feat)
            fused_feat = self.fused_embedding(add_feat)
            mask_pred = self.conv_logits(add_feat)
            if self.with_cls:
                return mask_pred, fused_feat, class_scores
            else:
                return mask_pred, fused_feat
        elif self.with_trans:
            trm_feat = F.interpolate(trm_feat, size=tuple(x.shape[-2:]), mode='bilinear', align_corners=True)  #upsample
            mask_pred = self.conv_logits(trm_feat)
            if self.with_cls:
                return mask_pred, trm_feat, class_scores
            else:
                return mask_pred, trm_feat
        elif self.with_fcn:
            mask_pred = self.conv_logits(fcn_feat)
            return mask_pred, fcn_feat
     
    @force_fp32(apply_to=('mask_pred', 'class_scores'))
    def loss(self, mask_pred, labels, class_scores=None, gt_labels=None):
        # (semantic_pred, gt_semantic_seg,scores,gt_labels)
        labels = labels.squeeze(1).long()
        mask_pred = mask_pred.squeeze(1)
        loss_semantic_seg = self.criterion(mask_pred, labels)

        if self.with_cls and self.with_trans:
        
            gt_labels = [lbl.unique() for lbl in gt_labels]
            targets = class_scores.new_zeros(class_scores.size())
            for i, label in enumerate(gt_labels):
                targets[i, label] = 1.0

            loss_semantic_cls = self.cls_loss_weight * self.criterion_cls(class_scores, targets)

            return loss_semantic_seg, loss_semantic_cls
        else:
            return loss_semantic_seg

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM,EigenGradCAM,LayerCAM	
from pytorch_grad_cam.utils.image import show_cam_on_image
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import init_detector, inference_detector
from mmcv import Config
import torch
from matplotlib import pyplot as plt
import os
from shutil import copyfile
import numpy as np
import torch.nn  as nn

data_train = dict(
    type='CocoDataset',
    ann_file='/home/hxr/fusion5/mmdetection-2.18.1/data/coco/annotations/instances_val2017.json',
    img_prefix='/home/hxr/fusion5/mmdetection-2.18.1/data/coco/val2017/',
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
                'img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg'
            ])
    ],
    seg_prefix='/home/hxr/fusion5/mmdetection-2.18.1/data/coco/stuffthingmaps/val2017/')
data = build_dataset(data_train)
model = build_detector(
    Config.fromfile(
        '/home/hxr/fusion5/mmdetection-2.18.1/configs/htc/transformer_htc_r101_fpn_20e_coco.py').
    model)
print(torch.load("/home/hxr/fusion5/mmdetection-2.18.1/htc_ours.pth").keys())
model.load_state_dict(torch.load("/home/hxr/fusion5/mmdetection-2.18.1/htc_ours.pth")["state_dict"])
# print(data[0])
# print(model)

layer1 = model.roi_head.semantic_head.transformer_head.transformer.layers[-3:]
layer2 = model.roi_head.semantic_head.fcn_head.convs[-1:]
layer3 = nn.ModuleList([model.roi_head.semantic_head.fused_embedding])
print(layer1)
print(layer2)
print(layer3)


name = ['/transformer_branch.jpg','/fcn_branch.jpg','/fusion_embbeding.jpg']
file_path = './data/coco/val2017'

# def reshape_transform(tensor, height=36, width=36):
#     result = tensor[:, 1 :  , :].reshape(tensor.size(0),
#         height, width, tensor.size(2))

#     # Bring the channels to the first dimension,
#     # like in CNNs.
#     result = result.transpose(2, 3).transpose(1, 2)
#     return result


# test = ['000000201072','000000331075','000000194716','000000492110','000000505573','000000476415','000000458992','000000155179']
cam1 = GradCAMPlusPlus(model, layer1)
cam2 = GradCAMPlusPlus(model, layer2)
cam3 = GradCAMPlusPlus(model, layer3)
cam = [cam1, cam2, cam3]


    
for file_name in os.listdir(data_train['img_prefix']):
    # input_tensor = data.load_from_filename('000000024243.jpg')
    plt.xticks([])  #去掉x轴
    plt.yticks([])  #去掉y轴
    plt.axis('off')  #去掉坐标轴
    print('***************')
    # file_name = '000000575500.jpg'
    input_tensor = data.load_from_filename(file_name)
    save_path = 'grad_cam_1/' + file_name
    # if input_tensor and not os.path.exists(save_path + '/' + file_name):
    input_tensor={k:[v.data] for k,v in input_tensor.items()}
    input_tensor["img"] = input_tensor["img"][0].unsqueeze(0)
    input_tensor["gt_labels"]=[torch.tensor(v) for v in input_tensor["gt_labels"]]
    input_tensor["gt_semantic_seg"]=input_tensor["gt_semantic_seg"][0]
    # result = model(return_loss=False, rescale=True,**input_tensor)
    # ms_bbox_result, ms_segm_result = result
    # if isinstance(ms_bbox_result, dict):
    #     result = (ms_bbox_result['ensemble'],ms_segm_result['ensemble'])        
    for i in range(len(cam)):
        target_category = 80
        grayscale_cam = cam[i](input_tensor=input_tensor, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]
        if os.path.exists(save_path) != True:
            os.makedirs(save_path)
        plt.imshow(grayscale_cam)
        plt.savefig(save_path + name[i],bbox_inches = 'tight', pad_inches = 0)
        # print(input_tensor['img_metas'])
        # plt.savefig(save_path + name[i],bbox_inches = 'tight', pad_inches = 0)
        copyfile(data_train['img_prefix'] + file_name ,save_path + '/' + file_name)
        print('save to ',save_path + name[i])
        print('################')
    break


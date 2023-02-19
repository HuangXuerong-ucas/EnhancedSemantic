
from mmdet.apis import init_detector, inference_detector
from mmdet.datasets import build_dataset
import mmcv
import os

config_file = '/home/hxr/fusion6/mmdetection-2.18.1/configs/htc/transformer_htc_r50_fpn_1x_coco.py'
checkpoint_file = '/home/hxr/fusion6/mmdetection-2.18.1/work_dirs/transformer_htc_r101_fpn_20e_coco/epoch_12.pth'


# config_file = '/home/hxr/fusion3/mmdetection/configs/htc/htc_r101_fpn_20e_coco.py'
# checkpoint_file = '/home/hxr/mmdetection_multiview/work_dirs2/epoch_20.pth'
# save_path = '/home/hxr/fusion3/mmdetection/test_predict_htc/'



# image_path = '/home/hxr/fusion3/mmdetection/data/coco/test2017/'

# coco_annotations = '/home/hxr/fusion3/mmdetection/data/coco/annotations/image_info_test-dev2017.json'

#val prediction
image_path = '/home/hxr/fusion6/mmdetection-2.18.1/data/coco/val2017/'

coco_annotations = '/home/hxr/fusion6/mmdetection-2.18.1/data/coco/annotations/instances_val2017.json'

config = mmcv.Config.fromfile(config_file)
datasets = build_dataset(config.data.val)
img_infos = datasets.load_annotations(coco_annotations)
print(len(img_infos))

model = init_detector(config_file, checkpoint_file, device='cuda:1')
print(model)

for img_info in img_infos:
    img_id = img_info['id']
    name = img_info['file_name']
    print(img_info.keys())
    ann_ids = datasets.coco.get_ann_ids(img_ids=[img_id])
    ann_infos = datasets.coco.load_anns(ann_ids)#一张图像有多少ann
    img_file = image_path + name
    
    is_large = False
    for ann in ann_infos:
        if ann['area'] > 96*96:
            save_path = '/home/hxr/fusion6/mmdetection-2.18.1/htc+es_train_large_val_predict_large/'
            is_large = True
            break
    if not is_large:
        save_path = '/home/hxr/fusion6/mmdetection-2.18.1/htc+es_train_large_val_predict_htc_small_midlle/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = save_path + name
    if not os.path.exists(save_file):
        result = inference_detector(model, img_file)
        model.show_result(img_file, result,  out_file=save_file,bbox_color=(0,255,0))
        print('prediction {0}'.format(save_file))
            # break
                
        # print(img_info.keys())
        # ann_ids = datasets.coco.get_ann_ids(img_ids=[img_id])
        # ann_infos = datasets.coco.load_anns(ann_ids)#一张图像有多少ann
        # for ann_info in ann_infos: #对于每一个标注
        #     # print(ann_info.keys())
        #     if ann_info['iscrowd'] == 1:  # total 446
        #         img_name = img_info['file_name']
        #         img_file = image_path + img_name
        #         save_file = save_path + img_name
        #         result = inference_detector(model, img_file)
        #         model.show_result(img_file, result,  out_file=save_file)
        #         print('prediction {0}'.format(save_file))

# img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once

# result = inference_detector(model, img)
# model.show_result(img, result, out_file='result.jpg')

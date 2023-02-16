from matplotlib import pyplot as plt
import json
path = '/home/hxr/fusion6/mmdetection-2.18.1/work_dirs/htc_r50_fpn_1x_coco/20220621_230625.log.json'
# result = open(path,"r",encoding="utf-8")
path2 = '/home/hxr/fusion6/mmdetection-2.18.1/work_dirs/transformer_htc_r101_fpn_20e_coco/20220616_152543.log.json'

path3 = '/home/hxr/fusion6/mmdetection-2.18.1/work_dirs/htc_r50_fpn_1x_coco_large/20220527_212656.log.json'

path4 = '/home/hxr/fusion6/mmdetection-2.18.1/work_dirs/transformer_htc_r101_fpn_20e_coco_large/20220530_011013.log.json'

path5 = '/home/hxr/fusion6/mmdetection-2.18.1/work_dirs/htc_r50_fpn_1x_coco_middle_small/20220605_010512.log.json'

path6 = '/home/hxr/fusion6/mmdetection-2.18.1/work_dirs/transformer_htc_r101_fpn_20e_coco_middle_small/20220606_014110.log.json'

path7 = '/home/hxr/fusion6/mmdetection-2.18.1/tools/htc_r50_fpn_1x_coco_20200317_070435.log.json'
path8 = '/home/hxr/fusion6/mmdetection-2.18.1/tools/20211219_214915.log.json'
def load_json(path, label):
    
    # 由于文件中有多行，直接读取会出现错误，因此一行一行读取
    file = open(path, 'r', encoding='utf-8')
    result = []
    for line in file.readlines():
        result.append(json.loads(line))

    losses = [loss['loss'] for loss in result[1:] if 'loss' in loss]
# print(losses)
# figure = plt.figure(figsize=(20,40))
    
    plt.plot(range(len(losses)), losses, label = label)

if __name__ == '__main__':
    # plt.figure(figsize=(40,20), dpi= 80)
    load_json(path, 'htc-small')
    load_json(path2, 'htc+es-small')
    load_json(path3, 'htc-middle')
    load_json(path4, 'htc+es-middle') 
    load_json(path5, 'htc-large')
    load_json(path6, 'htc+es-large')
    load_json(path7, 'htc-all')
    load_json(path8, 'htc+es-all')
    plt.legend()
    plt.show()
    plt.savefig('/home/hxr/fusion6/mmdetection-2.18.1/tools/r.png')
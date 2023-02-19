# Enhanced semantic head for cascade instance segmentation
## 环境依赖
所有实验都在一台配备了ubuntu 20.04操作系统、Intel(R) Xeon(R) Gold 5218R处理器和GeForce RTX 3090 显卡的服务器上进行，相关依赖如下（见code/requirements.txt）：

- python
- cityscapesScripts
- matplotli
- mmcv_full
- opencv_python
- pycocotools
- scikit_learn
- pytorch
- tqdm


## 数据集
论文中主要涉及到以下两个数据集，所有数据均已公开，具体如下：
- MSCOCO 2017: 可通过[链接](https://cocodataset.org/#download)进行下载，数据保存的根目录为code/data/coco, MSCOCO数据集的训练集、验证集及测试集的命名及保存路径
        
    code/data/coco:

        /annotations
        /train2017
        /val2017
        /test2017
        /stuffthingmaps
        
- CityScapes:可通过[链接](https://www.cityscapes-dataset.com/)进行下载，数据保存的根目录为code/data/，CityScapes数据集的训练集、验证集及测试集的命名及保存路径

    code/data/cityscapes:
        
        /annotations
        /leftImg8bit
            /train
            /val
        /gtFine
            /train
            /val


## 代码运行
1. 参考[get_started.md](./code/docs/get_started.md)配置环境依赖
2. 下载并按格式保存数据集
3. 运行脚本
        
    tools/dist_train.sh configs/htc/transformer_htc_r50_fpn_1x_coco.py 8

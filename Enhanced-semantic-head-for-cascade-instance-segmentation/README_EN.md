# EnhancedSemantic
# Enhanced semantic head for cascade instance segmentation
## Dependencies
All experiments are performed on a server equipped with ubuntu 20.04 operating system, Intel(R) Xeon(R) Gold 5218R processor and GeForce RTX 3090 graphics card. In addition, our experiments require:

- python
- cityscapesScripts
- matplotli
- mmcv_full
- opencv_python
- pycocotools
- scikit_learn
- pytorch
- tqdm


## Datasets
In our paper, we evaluate our proposed method on two benchmark datasets, including:
- MSCOCO 2017: [Download here](https://cocodataset.org/#download),The root directory of data storage is code/data/coco, the naming and saving path of the training set, verification set and test set of MSCOCO dataset
        
    code/data/coco:

        /annotations
        /train2017
        /val2017
        /test2017
        /stuffthingmaps

- CityScapes: [Download here](https://www.cityscapes-dataset.com/),The root directory of data storage is code/data/cityscapes, the naming and saving path of the training set, verification set and test set of the CityScapes dataset

    code/data/cityscapes:
        
        /annotations
        /leftImg8bit
            /train
            /val
        /gtFine
            /train
            /val

## How to Run
1. Refer to [get_started.md](./code/docs/get_started.md) to download and install all environment dependencies.
2. Download and save the dataset in format
3. Run tools/dist_train.sh
            
    tools/dist_train.sh configs/htc/transformer_htc_r50_fpn_1x_coco.py 8

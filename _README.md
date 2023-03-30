# MMYOLO

## Install

> Refer: https://github.com/open-mmlab/mmyolo#%EF%B8%8F-installation-

```bash
conda create -n mmyolo python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate mmyolo
pip install openmim
mim install "mmengine>=0.6.0"
mim install "mmcv>=2.0.0rc4,<2.1.0"
mim install "mmdet>=3.0.0rc6,<3.1.0"
git clone https://github.com/XavierJiezou/mmyolo.git
cd mmyolo
# Install albumentations
pip install -r requirements/albu.txt
# Install MMYOLO
mim install -v -e .
```

## Example

### Common Objective Detection

> Refer: https://mmyolo.readthedocs.io/zh_CN/latest/get_started/15_minutes_object_detection.html#id9

1. Download a dataset of cats, including 144 images and their corresponding annotations.

```bash
python tools/misc/download_dataset.py --dataset-name cat --save-dir ./data/cat --unzip --delete
```

2. Create a configuration file named `yolov5_s-v61_fast_1xb12-40e_cat.py` in the `configs/yolov5` folder and write the following:

```python
# 基于该配置进行继承并重写部分配置
_base_ = 'yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'

data_root = './data/cat/' # 数据集根路径
class_name = ('cat', ) # 数据集类别名称
num_classes = len(class_name) # 数据集类别数
# metainfo 必须要传给后面的 dataloader 配置，否则无效
# palette 是可视化时候对应类别的显示颜色
# palette 长度必须大于或等于 classes 长度
metainfo = dict(classes=class_name, palette=[(20, 220, 60)])

# 基于 tools/analysis_tools/optimize_anchors.py 自适应计算的 anchor
anchors = [
    [(68, 69), (154, 91), (143, 162)],  # P3/8
    [(242, 160), (189, 287), (391, 207)],  # P4/16
    [(353, 337), (539, 341), (443, 432)]  # P5/32
]
# 最大训练 40 epoch
max_epochs = 40
# bs 为 12
train_batch_size_per_gpu = 12
# dataloader 加载进程数
train_num_workers = 4

# 加载 COCO 预训练权重
load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth'  # noqa

model = dict(
    # 固定整个 backbone 权重，不进行训练
    backbone=dict(frozen_stages=4),
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
        prior_generator=dict(base_sizes=anchors)
    ))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        # 数据集标注文件 json 路径
        ann_file='annotations/trainval.json',
        # 数据集前缀
        data_prefix=dict(img='images/')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/test.json',
        data_prefix=dict(img='images/')))

test_dataloader = val_dataloader

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

val_evaluator = dict(ann_file=data_root + 'annotations/test.json')
test_evaluator = val_evaluator

default_hooks = dict(
    # 每隔 10 个 epoch 保存一次权重，并且最多保存 2 个权重
    # 模型评估时候自动保存最佳模型
    checkpoint=dict(interval=10, max_keep_ckpts=2, save_best='auto'),
    # warmup_mim_iter 参数非常关键，因为 cat 数据集非常小，默认的最小 warmup_mim_iter 是 1000，导致训练过程学习率偏小
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=10),
    # 日志打印间隔为 5
    logger=dict(type='LoggerHook', interval=5))
# 评估间隔为 10
train_cfg = dict(max_epochs=max_epochs, val_interval=10)
```

3. Train the model with pretrained backbone

```bash
python tools/train.py configs/yolov5/yolov5_s-v61_fast_1xb12-40e_cat.py
```

4. Evaluated performance on [data/cat/annotations/test.json](data/cat/annotations/test.json) with COCO API as follows:

```bash
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.744
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.971
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.854
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.744
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.673
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.807
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.807
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.807
```

5. To test the performance of the trained model and obtain the visulization resutls in `work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/{timestamp}/show_results`

```bash
python tools/test.py configs/yolov5/yolov5_s-v61_fast_1xb12-40e_cat.py \
                     work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/epoch_40.pth \
                     --show-dir show_results
```

### Rotated Objective Detection

TODO

## Experiments

### Datasets

#### [DOTA-v1.0](https://mmyolo.readthedocs.io/zh_CN/latest/recommended_topics/dataset_preparation.html)

1. Download the dataset from [OpenDataLab](https://opendatalab.org.cn/DOTA_V1.0)

2. Unzip

```bash
tar -xzf DOTA_V1.0.tar.gz
```

3. Split (Run `pip install shapely` first)

- single-scale split

```bash
python tools/dataset_converters/dota/dota_split.py \
--split_config "tools/dataset_converters/dota/split_config/single_scale.json" \
--data_root "data/DOTA_V1.0" \
--out_dir "data/split_ss_dota"
```

- multi-scale split

TODO

#### DOTAv1.5

TODO

### Methods

`pip install mmrotate==1.0.0rc1` must be run first for rotated objective detection.

#### RTMDet

```bash
python tools/train.py configs/rtmdet/rotated/rtmdet-r_s_fast_1xb8-36e_dota.py
python tools/test.py configs/rtmdet/rotated/rtmdet-r_s_fast_1xb8-36e_dota.py work_dirs/rtmdet-r_s_fast_1xb8-36e_dota/epoch_36.pth --show-dir show_results
python tools/train.py configs/rtmdet/rotated/rtmdet-r_l_syncbn_fast_2xb4-36e_dota.py
```

#### YOLOv8

- [ ] Support YOLOv8-Rotated

## Evaluation

> Online evaluation: https://captain-whu.github.io/DOTA/evaluation.html

![1679041304663](1679041304663.png)

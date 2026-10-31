_base_ = 'mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

# 默认相对项目根；tools/common.py 将 SSDD_DATA_ROOT 同步到数据集及评测器。
data_root = 'data/SSDD/'
metainfo = dict(classes=('ship',))  # 单类船舶；COCO 类别 ID 为 1
model = dict(
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    roi_head=dict(bbox_head=dict(num_classes=1)))  # 仅 ImageNet 骨干预训练

# 尺度顺序是 (宽, 高)，保持比例，短边 800 / 长边不超过 1333。
# SAR 灰度按默认加载器读为三通道；禁止颜色类增强。
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),  # 唯一随机增强：水平翻转
    dict(type='PackDetInputs')]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=(
        'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))]

train_dataloader = dict(
    batch_size=2, num_workers=4, persistent_workers=False,  # 8GB 显存 / Windows
    dataset=dict(data_root=data_root, metainfo=metainfo,
                 ann_file='annotations/train_norm.json',
                 data_prefix=dict(img='images/'),
                 filter_cfg=dict(filter_empty_gt=False, min_size=0),
                 pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1, num_workers=4, persistent_workers=False,
    dataset=dict(data_root=data_root, metainfo=metainfo,
                 ann_file='annotations/val_norm.json',
                 data_prefix=dict(img='images/'), pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1, num_workers=4, persistent_workers=False,
    dataset=dict(data_root=data_root, metainfo=metainfo,
                 ann_file='annotations/test_norm.json',
                 data_prefix=dict(img='images/'), pipeline=test_pipeline))
val_evaluator = dict(ann_file=data_root + 'annotations/val_norm.json', metric='bbox')
test_evaluator = dict(ann_file=data_root + 'annotations/test_norm.json', metric='bbox')

# 单卡 batch=2 的有效 lr=0.02*(2/16)=0.0025；不可再将基础 lr 写成 0.0025。
optim_wrapper = dict(type='OptimWrapper', optimizer=dict(
    type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))
auto_scale_lr = dict(enable=True, base_batch_size=16)  # batch=1 自动得到 0.00125
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=12, by_epoch=True, milestones=[8, 11], gamma=0.1)]
randomness = dict(seed=0, deterministic=True)  # 固定种子；不同硬件不承诺逐位一致
env_cfg = dict(cudnn_benchmark=False, mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0))
default_hooks = dict(
    logger=dict(interval=50),
    checkpoint=dict(interval=1, max_keep_ckpts=2, save_best='coco/bbox_mAP', rule='greater'))
work_dir = 'work_dirs/faster_rcnn_r50_fpn_ssdd'

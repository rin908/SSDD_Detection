_base_ = './faster_rcnn_r50_fpn_ssdd.py'
# 可选方形输入：保持比例缩放后补零至 1024×1024，不拉伸船舶。
train_pipeline = [
    dict(type='LoadImageFromFile'), dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(1024, 1024), pad_val=dict(img=(0, 0, 0))),
    dict(type='PackDetInputs')]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='Pad', size=(1024, 1024), pad_val=dict(img=(0, 0, 0))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=(
        'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))]
train_dataloader = dict(batch_size=1, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
# 8GB 可选训练变体启用动态 loss scale；FPS 脚本仍强制 FP32。
optim_wrapper = dict(type='AmpOptimWrapper', loss_scale='dynamic')
work_dir = 'work_dirs/faster_rcnn_r50_fpn_ssdd_1024'

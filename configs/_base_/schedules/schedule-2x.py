# optimizer
# This schedule is mainly used by models on nuScenes dataset
lr = 0.001 # Original is 0.001
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    # max_norm=10 is better for SECOND
    clip_grad=dict(max_norm=35, norm_type=2))

# training schedule for 2x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=100)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='ValLoop') # Previously TestLoop

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 1000,
        by_epoch=False,
        begin=0,
        end=100),
    dict(
        type='MultiStepLR',
        begin=0,
        end=200,
        by_epoch=True,
        milestones=[20, 23],
        gamma=0.1)
]

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (4 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=32)

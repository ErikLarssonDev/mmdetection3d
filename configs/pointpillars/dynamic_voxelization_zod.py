_base_ = [
    '../_base_/models/pointpillars_dynamic_voxelization_250r.py',
    '../_base_/datasets/zod_restruct.py',
    '../_base_/schedules/schedule-2x.py', '../_base_/default_runtime.py'
]
# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=1)

experiment_name = 'dynamic_voxelization_20e'
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=1)

work_dir = './work_dirs/' + experiment_name

auto_scale_lr = dict(enable=False, base_batch_size=1)
val_evaluator = dict(
    metric_save_dir='./work_dirs/' + experiment_name,
)

model = dict(
    voxel_encoder=dict(
        in_channels=5, # Change this when adding more point features
    )
)
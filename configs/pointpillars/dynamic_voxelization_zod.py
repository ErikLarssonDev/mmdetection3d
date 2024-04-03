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

experiment_name = 'dynamic_voxelization_20e_b1_time_feature_minizod'
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=1)

work_dir = './work_dirs/' + experiment_name

auto_scale_lr = dict(enable=False, base_batch_size=1)
val_evaluator = dict(
    metric_save_dir='./work_dirs/' + experiment_name,
)
bonus_dataset_options = dict(
    use_frame_time_feature=True,
    frames_before=1,
    frames_after=1,
    num_previous_frames_on_main_path=1,
    secondary_data_path='data/minizod/mini_zod',
    filter_empty_gt=True
)
model = dict(
    voxel_encoder=dict(
        in_channels=5, # Change this when adding more point features
    )
)

train_dataloader = dict(
    dataset=dict(
        dataset = bonus_dataset_options
    )
)

test_dataloader = dict(
    dataset = bonus_dataset_options
)

val_dataloader = dict(
    dataset = bonus_dataset_options
)
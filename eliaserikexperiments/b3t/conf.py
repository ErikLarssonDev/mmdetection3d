_base_ = [
    '../../configs/_base_/models/pointpillars_dynamic_voxelization_250r.py',
    '../../configs/_base_/datasets/zod_restruct.py',
    '../../configs/_base_/schedules/schedule-2x.py', '../../configs/_base_/default_runtime.py'
]
# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=1)

experiment_name = 'dynamic_voxelization_20e_b3T'
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=1)
data_path = "bigzod/"
work_dir = './work_dirs/' + experiment_name

auto_scale_lr = dict(enable=False, base_batch_size=1)
val_evaluator = dict(
    metric_save_dir='./work_dirs/' + experiment_name,
)
bonus_dataset_options = dict(
    use_frame_time_feature=True,
    frames_before=3,
    frames_after=0,
    num_previous_frames_on_main_path=3,
    secondary_data_path='/media/erila/KINGSTON/zod_mmdet3d/points',
    filter_empty_gt=True,
    num_before_frames_bounds=[], # Needs to have one list with a range for each frames before. If left empty you get all frames before for all ranges.
    data_root=data_path,
)

model = dict(
    voxel_encoder=dict(
        in_channels=5 if bonus_dataset_options["use_frame_time_feature"] else 4, # Change this when adding more point features
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
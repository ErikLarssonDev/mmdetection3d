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
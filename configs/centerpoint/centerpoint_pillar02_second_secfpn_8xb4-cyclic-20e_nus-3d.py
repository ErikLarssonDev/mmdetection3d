_base_ = [
    '../_base_/datasets/zod_restruct.py',
    '../_base_/models/centerpoint_pillar02_second_secfpn_nus.py',
    '../_base_/schedules/cyclic-20e.py', '../_base_/default_runtime.py'
]
point_cloud_range = [-25, 0, -5, 25, 250, 3]
class_names = ['Vehicle', 'VulnerableVehicle', 'Pedestrian', 'Animal', 'StaticObject'] 
voxel_size = [0.2, 0.2, 8]
# Using calibration info convert the Lidar-coordinate point cloud range to the
# ego-coordinate point cloud range could bring a little promotion in nuScenes.
# point_cloud_range = [-51.2, -52, -5.0, 51.2, 50.4, 3.0]
# For nuScenes we usually do 10-class detection


model = dict(
    data_preprocessor=dict(
        voxel_layer=dict(point_cloud_range=point_cloud_range, voxel_size=voxel_size)
    ),
    pts_voxel_encoder=dict(point_cloud_range=point_cloud_range, in_channels=4, voxel_size=voxel_size),
    pts_middle_encoder = dict(
        output_shape=(
            int(2048), int(512)
        )
    ),
    pts_bbox_head=dict(bbox_coder=dict(pc_range=point_cloud_range[:2], voxel_size=voxel_size[:2])),
    
    # model training and testing settings
    train_cfg=dict(pts=dict(point_cloud_range=point_cloud_range, voxel_size=voxel_size)),
    test_cfg=dict(pts=dict(pc_range=point_cloud_range[:2], voxel_size=voxel_size))

)

dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
backend_args = None

# train_pipeline = [
#     dict(
#         type='LoadPointsFromFile',
#         coord_type='LIDAR',
#         load_dim=4,
#         use_dim=4,
#         backend_args=backend_args),
#     dict(
#         type='LoadPointsFromMultiSweeps',
#         sweeps_num=9,
#         use_dim=[0, 1, 2, 3, 4],
#         pad_empty_sweeps=True,
#         remove_close=True,
#         backend_args=backend_args),
#     dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
#     dict(
#         type='GlobalRotScaleTrans',
#         rot_range=[-0.3925, 0.3925],
#         scale_ratio_range=[0.95, 1.05],
#         translation_std=[0, 0, 0]),
#     dict(
#         type='RandomFlip3D',
#         sync_2d=False,
#         flip_ratio_bev_horizontal=0.5,
#         flip_ratio_bev_vertical=0.5),
#     dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
#     dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
#     dict(type='PointShuffle'),
#     dict(
#         type='Pack3DDetInputs',
#         keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
# ]
# test_pipeline = [
#     dict(
#         type='LoadPointsFromFile',
#         coord_type='LIDAR',
#         load_dim=4,
#         use_dim=4,
#         backend_args=backend_args),
#     dict(
#         type='LoadPointsFromMultiSweeps',
#         sweeps_num=9,
#         use_dim=[0, 1, 2, 3, 4],
#         pad_empty_sweeps=True,
#         remove_close=True,
#         backend_args=backend_args),
#     dict(
#         type='MultiScaleFlipAug3D',
#         img_scale=(1333, 800),
#         pts_scale_ratio=1,
#         flip=False,
#         transforms=[
#             dict(
#                 type='GlobalRotScaleTrans',
#                 rot_range=[0, 0],
#                 scale_ratio_range=[1., 1.],
#                 translation_std=[0, 0, 0]),
#             dict(type='RandomFlip3D')
#         ]),
#     dict(type='Pack3DDetInputs', keys=['points'])
# ]


# test_dataloader = dict(
#     dataset=dict(pipeline=test_pipeline, metainfo=dict(classes=class_names)))
# val_dataloader = dict(
#     dataset=dict(pipeline=test_pipeline, metainfo=dict(classes=class_names)))

# train_cfg = dict(val_interval=20)

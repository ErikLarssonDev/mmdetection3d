voxel_size = [0.25, 0.25, 8]
pcr_range = [-25, 0, -5, 25, 20, 3]

model = dict(
    # with_cp = True, #Tip to reduce GPU memory
    type='MVXFasterRCNN',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            voxel_size=voxel_size,
            max_num_points=20,      
            point_cloud_range=pcr_range,
            max_voxels=(60000, 60000))),

    pts_voxel_encoder=dict(
        type='HardVFE',
        in_channels=4,
        feat_channels=[64], 
        point_cloud_range=pcr_range,

        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01)),

    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[800, 800]),

    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),


    pts_neck=dict(
        type='mmdet.FPN',
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='ReLU'),
        in_channels=[64, 128, 256],
        out_channels=256,
        start_level=0,
        num_outs=3),

    pts_bbox_head=dict(
        type='Anchor3DHead',
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[pcr_range], custom_values=[],
            scales=[1, 2, 4],
            sizes=[
                [2.5981,    0.8660, 1.],  # 1.5 / sqrt(3)
                [1.7321,    0.5774, 1.],  # 1 / sqrt(3)
                [1.,        1.,     1.],
                [0.4,       0.4,    1],
                [4.016,     1.693,  1.563],
                [1.072,     0.414,  1.222],
                [0.1,       0.4,    0.436],
                [0.075,     0.074,  0.484],
                [0.1,       0.253,  0.435],
            ],
            rotations=[0, 1.57],
            reshape_out=True),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=-0.7854,  # -pi / 4
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False,
            loss_weight=0.2),
            
        num_classes=11,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=7)),

    train_cfg=dict(
        pts=dict(
            assigner=dict(
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1,
                gpu_assign_thr=-1), # memory reduce tip, should be as high as GPU can handle -1 is infinite
            allowed_border=0,
            # code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], #pcr_range
            pos_weight=-1,
            debug=False)
    ),
    test_cfg=dict(
        pts=dict(
            use_rotate_nms=True, # use_rotate_nms=True, # TODO add this after testing is done
            nms_across_levels=False,
            nms_pre=1000,
            nms_thr = 0.2, # nms_thr=0.2,
            score_thr = 0.05, # score_thr=0.05,
            min_bbox_size=0,
            max_num=500
        ),
        assigner=dict(
            type='Max3DIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.6,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1,
            gpu_assign_thr=0),
    ),
    # model training settings (based on nuScenes model settings)
    # train_cfg=dict(pts=dict(code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
)

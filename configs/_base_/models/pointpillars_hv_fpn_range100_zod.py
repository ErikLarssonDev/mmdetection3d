# from math import ceil
# [tensor(25.0000), tensor(242.1388), tensor(3.0000)] [tensor(-24.9999), tensor(6.2734e-05), tensor(-4.9970)]
voxel_size = [0.16, 0.16, 8] # Originally 0.25, 0.25, 8 0.16 0.16
pcr_range = [-25, 0, -5, 25, 245, 3]

model = dict(
    # with_cp = True, #Tip to reduce GPU memory
    type='MVXFasterRCNN',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            voxel_size=voxel_size,
            max_num_points=100,      
            point_cloud_range=pcr_range,
            max_voxels=(60000, 60000))),

    pts_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        legacy=False),

    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, 
        output_shape=[
            int(245 / 0.16 + 1),
            int(50 / 0.16 + 1) ,
            #int((pcr_range[4]-pcr_range[1])/voxel_size[0] + 1), # Y-range / x-size??! # Originally 400
            # int((pcr_range[3]-pcr_range[0])/voxel_size[2] + 1)  # X-range / Z-size??! # Originally 400
            ] 
    ),

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
        assign_per_class=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                [-25, 0, -3, 25, 250, 1], # Vehicle
                [-25, 0, -3, 25, 250, 1], # VulnerableVehicle
                [-25, 0, -3, 25, 250, 1], # Pedestrian
                [-25, 0, -3, 25, 250, 1], # Animal
                [-25, 0, -3, 25, 250, 1], # StaticObject
                # [-25, 0, 0, 25, 250, 0],  # PoleObject
                # [-25, 0, -3, 25, 250, 0], # TrafficBeacon 
                # [-25, 0, -0.1, 25, 250, 0.1], # TrafficSign
                # [-25, 0, -3, 25, 250, 0], # TrafficSignal
                # [-25, 0, -3, 25, 250, 0], # TrafficGuide
                # [-25, 0, -3, 25, 250, 0], # DynamicBarrier
                # [-25, 0, -3, 25, 250, 0], # Unclear
            ], 
            custom_values=[],
            scales=[1, 2, 4],
            sizes=[ # length, width, height
                [ 4.67,   1.96, 1.74], # Vehicle
                [  1.68,  0.61, 1.34],  # VulnerableVehicle
                [ 0.62,   0.64, 1.68], # Pedestrian
                [  .89,   .37,  .64], # Animal
                [ 0.17,   0.55, 2.38], # StaticObject
                # [0.19,  0.19,   5.14], # PoleObject
                # [.1,    .1,       .1], # TrafficBeacon
                # [0.11,  0.75,   0.59], # TrafficSign
                # [0.30,  0.45,   0.98], # TrafficSignal
                # [0.14,  0.20,   0.83], # TrafficGuide
                # [.1,    .1,       .1], # DynamicBarrier
                # [1.89,  1.19,   1.57], # Unclear
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
            
        num_classes=5,
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
            use_rotate_nms=True, # use_rotate_nms=True
            nms_across_levels=False,
            nms_pre=1000,
            nms_thr = 0.02, # nms_thr=0.2,
            score_thr = 0.05, # score_thr=0.05,
            min_bbox_size=0,
            max_num=500,
            assigner=dict(
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1,
                gpu_assign_thr=0
            ),
        ),
    ),
    # model training settings (based on nuScenes model settings)
    # train_cfg=dict(pts=dict(code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
)

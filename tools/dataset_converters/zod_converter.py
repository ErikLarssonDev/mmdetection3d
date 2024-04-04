# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
from tqdm import tqdm
import mmengine
import numpy as np
import os
from .kitti_data_utils import  get_kitti_image_info
from mmdet3d.structures.ops import box_np_ops
from multiprocessing import Pool

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [line.replace('\n','') for line in lines]

def get_zod_image_info(path,
                       image_ids,
                        label_info=True,):
    """
    KITTI annotation format version 2:
    {
        [optional]points: [N, 3+] point cloud
        [optional, for kitti]image: {
            image_idx: ...
            image_path: ...
            image_shape: ...
        }
        point_cloud: {
            num_features: 4
            velodyne_path: ...
        }
        [optional, for kitti]calib: {
            R0_rect: ...
            Tr_velo_to_cam: ...
            P2: ...
        }
        annos: {
            location: [num_gt, 3] array
            dimensions: [num_gt, 3] array
            rotation_y: [num_gt] angle array
            name: [num_gt] ground truth name array
            [optional]difficulty: kitti difficulty
            [optional]group_ids: used for multi-part object
        }
    }
    """
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))
    pool = Pool(processes=8)
    def map_func(idx):
        info = {}
        pc_info = {'num_features': 4}
        annotations = None
        pc_info['velodyne_path'] = os.path.join(path, 'points', f'{idx}.bin')
        if label_info:
            label_path = os.path.join(path, 'labels', f'{idx}.txt')
            annotations = get_label_anno(label_path)
        info['point_cloud'] = pc_info

        if annotations is not None:
            info['annos'] = annotations
        return info


    image_infos = [map_func(im_id) for im_id in tqdm(image_ids)]


    return list(image_infos)


def _calculate_num_points_in_gt(data_path,
                                infos,
                                relative_path,
                                remove_outside=True,
                                num_features=4,
                                num_previous_frames=0):
    for info in mmengine.track_iter_progress(infos):
        pc_info = info['point_cloud']
        if relative_path:
            v_path = str(Path(data_path) / pc_info['velodyne_path'])
        else:
            v_path = pc_info['velodyne_path']
        points_v = np.fromfile(
            v_path, dtype=np.float32, count=-1).reshape([-1, num_features])
        for i in range(num_previous_frames):
            prev_v_path = v_path.replace('.bin', f'_b{i+1}.bin')
            if os.path.exists(prev_v_path):
                prev_points_v = np.fromfile(
                    prev_v_path, dtype=np.float32, count=-1).reshape([-1, num_features])
                points_v = np.concatenate([prev_points_v, points_v], axis=0)

        # points_v = points_v[points_v[:, 0] > 0]
        annos = info['annos']
        num_obj = len([n for n in annos['name'] if n != 'DontCare'])
        # annos = kitti.filter_kitti_anno(annos, ['DontCare'])
        dims = np.array(annos['dimensions'][:num_obj])
        loc = np.array(annos['location'][:num_obj])
        rots = np.array(annos['rotation_y'][:num_obj])

        gt_boxes_lidar = np.hstack((loc, dims, rots.reshape(rots.shape[0],1)))

        indices = box_np_ops.points_in_rbbox(points_v[:, :3], gt_boxes_lidar)
        num_points_in_gt = indices.sum(0)
        num_ignored = len(annos['dimensions']) - num_obj
        num_points_in_gt = np.concatenate(
            [num_points_in_gt, -np.ones([num_ignored])])
        annos['num_points_in_gt'] = num_points_in_gt.astype(np.int32)

def create_zod_info_file(data_path,
                           pkl_prefix='zod',
                           save_path=None,
                           relative_path=True,
                           num_prev_frames=0):
    """Create info file of custom dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        data_path (str): Path of the data root.
        pkl_prefix (str, optional): Prefix of the info file to be generated.
            Default: 'kitti'.
        with_plane (bool, optional): Whether to use plane information.
            Default: False.
        save_path (str, optional): Path to save the info file.
            Default: None.
        relative_path (bool, optional): Whether to use relative path.
            Default: True.
    """
    imageset_folder = Path(data_path) / 'ImageSets'
    train_img_ids = _read_imageset_file(str(imageset_folder / 'train.txt'))
    val_img_ids = _read_imageset_file(str(imageset_folder / 'val.txt'))
    test_img_ids = _read_imageset_file(str(imageset_folder / 'test.txt'))

    print('Generate info. this may take several minutes.')
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)
    kitti_infos_train = get_zod_image_info(
        data_path,
        image_ids=train_img_ids,
        label_info=True)
    _calculate_num_points_in_gt(data_path, kitti_infos_train, relative_path, num_previous_frames=num_prev_frames)
    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'Train file is saved to {filename}')
    mmengine.dump(kitti_infos_train, filename)

    kitti_infos_val = get_zod_image_info(
        data_path,
        image_ids=val_img_ids,
        label_info=True)
    _calculate_num_points_in_gt(data_path, kitti_infos_val, relative_path)
    filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    print(f'Val info file is saved to {filename}')
    mmengine.dump(kitti_infos_val, filename)
    filename = save_path / f'{pkl_prefix}_infos_trainval.pkl'
    print(f'Trainval info file is saved to {filename}')
    mmengine.dump(kitti_infos_train + kitti_infos_val, filename)
    kitti_infos_test = get_zod_image_info(
        data_path,
        image_ids=test_img_ids,
        label_info=True)
    _calculate_num_points_in_gt(data_path, kitti_infos_test, relative_path)
    filename = save_path / f'{pkl_prefix}_infos_test.pkl'
    print(f'Test info file is saved to {filename}')
    mmengine.dump(kitti_infos_test, filename)

def get_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
    # if len(lines) == 0 or len(lines[0]) < 15:
    #     content = []
    # else:
    content = [line.strip().split(' ') for line in lines]
    num_objects = len([x[7] for x in content if x[7] != 'DontCare'])
    annotations['name'] = np.array([x[7] for x in content])
    num_gt = len(annotations['name'])
    annotations['dimensions'] = np.array([[float(info) for info in x[3:6]]
                                    for x in content]).reshape(-1, 3)
    annotations['location'] = np.array([[float(info) for info in x[0:3]]
                                        for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array([float(x[6])
                                          for x in content]).reshape(-1)
    annotations['bbox'] = np.array([[float(info) for info in x[0:7]]
                                    for x in content]).reshape(-1, 7)
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    return annotations
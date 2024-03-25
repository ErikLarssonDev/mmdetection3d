import mmengine

from mmdet3d.registry import DATASETS
from mmdet3d.structures.bbox_3d.lidar_box3d import LiDARInstance3DBoxes
from .det3d_dataset import Det3DDataset
import numpy as np
import copy
import torch


currentmaxpoint = [0, 0, 0]
currentminpoint = [100, 100, 100]


class_translation_map = { 
    "Vehicle": "Vehicle",
    "VulnerableVehicle": "VulnerableVehicle",
    "Pedestrian": "Pedestrian",
    "Animal": "Animal",
    "PoleObject": "StaticObject",
    "TrafficBeacon": "StaticObject",
    "TrafficSign": "StaticObject",
    "TrafficSignal": "StaticObject",
    "TrafficGuide": "StaticObject",
    "DynamicBarrier": "StaticObject",
    "Unclear": "DontCare"
}

@DATASETS.register_module()
class ZodDatasetRestruct(Det3DDataset):
    def __init__(self, frames_before=2, frames_after=0, *args, **kwargs): # Number of frames to include before and after the current frame
        super().__init__(*args, **kwargs)
        self.frames_before = frames_before
        self.frames_after = frames_after

    METAINFO = {
        'classes': ['Vehicle', 'VulnerableVehicle', 'Pedestrian', 'Animal', 'StaticObject'],
        'palette': [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 165, 0), (128, 0, 128)],
        'allClasses': ['Vehicle', 'VulnerableVehicle', 'Pedestrian', 'Animal', 'PoleObject', 'TrafficBeacon', 'TrafficSign', 'TrafficSignal', 'TrafficGuide', 'DynamicBarrier', 'Unclear'],
        'object_range': [-25.04, 0, -5, 25.04, 245.12, 3] 
    }

    def parse_ann_info(self, info):

        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                  3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
        """
        # Translate the class names to the ones used in training according to class_translation map
        for annotation in info['instances']:
            if class_translation_map[self.METAINFO['allClasses'][annotation['bbox_label_3d']]] == "DontCare":
                new_label = -1
            else:
                new_label = self.METAINFO['classes'].index(class_translation_map[self.METAINFO['allClasses'][annotation['bbox_label_3d']]])
            annotation['bbox_label_3d'] = new_label
            annotation['bbox_label'] = new_label

        # After class translation, let super class do what it wants with annotation info
        ann_info = super().parse_ann_info(info)

        if ann_info is None:
            print(f"WARNING: Got empty instance from parse_ann_info {info['lidar_points']['lidar_path']}")
            ann_info = dict()
            # empty instance
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)


        # filter the gt classes not used in training
        ann_info = self._remove_dontcare(ann_info)
        ann_info = self.filter_annotations_on_range(ann_info)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            ann_info['gt_bboxes_3d'],
            origin=(0.5, 0.5, 0.5)
        ).convert_to(self.box_mode_3d)

        
        ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        return ann_info

    def parse_data_info(self, info):
        """
        Parse raw data from dataset
        """
        info = super().parse_data_info(info)
        return info

    def prepare_data(self, index: int):
        """Data preparation for both training and testing stage.

        Called by `__getitem__`  of dataset.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict or None: Data dict of the corresponding index.
        """
        ori_input_dict = self.get_data_info(index)

        # deepcopy here to avoid inplace modification in pipeline.
        input_dict = copy.deepcopy(ori_input_dict)

        # box_type_3d (str): 3D box type.
        input_dict['box_type_3d'] = self.box_type_3d
        # box_mode_3d (str): 3D box mode.
        input_dict['box_mode_3d'] = self.box_mode_3d
        # pre-pipline return None to random another in `__getitem__`
        if not self.test_mode and self.filter_empty_gt:
            if len(input_dict['ann_info']['gt_labels_3d']) == 0:
                return None

        example = self.pipeline(input_dict)
        
        saved_lidar_path = input_dict['lidar_points']["lidar_path"]
        for frame_before_index in range(self.frames_before):
            input_dict['lidar_points']["lidar_path"] = saved_lidar_path.replace(".bin", f"_b{frame_before_index+1}.bin")
            example['inputs']['points'] = torch.cat((example['inputs']['points'], self.pipeline(input_dict)['inputs']['points']),0) # Add points from previous frames
        for frame_after_index in range(self.frames_after):
            input_dict['lidar_points']["lidar_path"] = saved_lidar_path.replace(".bin", f"_a{frame_after_index+1}.bin")
            example['inputs']['points'] = torch.cat((example['inputs']['points'], self.pipeline(input_dict)['inputs']['points']),0) # Add points from future frames
        if not self.test_mode and self.filter_empty_gt:
            # after pipeline drop the example with empty annotations
            # return None to random another in `__getitem__`
            if example is None or len(
                    example['data_samples'].gt_instances_3d.labels_3d) == 0:
                return None

        if self.show_ins_var:
            if 'ann_info' in ori_input_dict:
                self._show_ins_var(
                    ori_input_dict['ann_info']['gt_labels_3d'],
                    example['data_samples'].gt_instances_3d.labels_3d)

        return example

    
    def filter_annotations_on_range(self, ann_info):
        # Remove all object annotations that have center point outside of self.METAINFO.object_range
        filtered_annotations = {}
        filter_mask = np.all([ann_info['gt_bboxes_3d'][:,0] > self.METAINFO["object_range"][0], 
                        ann_info['gt_bboxes_3d'][:,0] < self.METAINFO["object_range"][3],
                        ann_info['gt_bboxes_3d'][:,1] > self.METAINFO["object_range"][1],
                        ann_info['gt_bboxes_3d'][:,1] < self.METAINFO["object_range"][4],
                        ann_info['gt_bboxes_3d'][:,2] > self.METAINFO["object_range"][2],
                        ann_info['gt_bboxes_3d'][:,2] < self.METAINFO["object_range"][5]], axis=0)
        for key in ann_info.keys():
            if key != 'instances':
                filtered_annotations[key] = (ann_info[key][filter_mask])
            else:
                filtered_annotations[key] = ann_info[key]

        return filtered_annotations
        
import mmengine

from mmdet3d.registry import DATASETS
from mmdet3d.structures.bbox_3d.lidar_box3d import LiDARInstance3DBoxes
from .det3d_dataset import Det3DDataset
import numpy as np
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    # replace with all the classes in customized pkl info file
        self.printed = False
    METAINFO = {
        'classes': ['Vehicle', 'VulnerableVehicle', 'Pedestrian', 'Animal', 'StaticObject'],
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192), (60, 255, 60)],
        'allClasses': ['Vehicle', 'VulnerableVehicle', 'Pedestrian', 'Animal', 'PoleObject', 'TrafficBeacon', 'TrafficSign', 'TrafficSignal', 'TrafficGuide', 'DynamicBarrier', 'Unclear'],
        'object_range': [-25, 0, -5, 25, 250, 3]
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
            print("WARNING: Got empty instance from parse_ann_info")
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

    def prepare_data(self, idx):
        data = super().prepare_data(idx)
        return data

    
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
        
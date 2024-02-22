import mmengine

from mmdet3d.registry import DATASETS
from mmdet3d.structures.bbox_3d.lidar_box3d import LiDARInstance3DBoxes
from .det3d_dataset import Det3DDataset
import numpy as np



@DATASETS.register_module()
class ZodDatasetRestruct(Det3DDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    # replace with all the classes in customized pkl info file
    METAINFO = {
        'classes': ['Vehicle', 'VulnerableVehicle', 'Pedestrian', 'Animal', 'PoleObject', 'TrafficBeacon', 'TrafficSign', 'TrafficSignal', 'TrafficGuide', 'DynamicBarrier', 'Unclear'],
        'object_range': [-25, 0, -5, 25, 20, 3]
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
        gt_bboxes_3d = LiDARInstance3DBoxes(ann_info['gt_bboxes_3d'])


        ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        return ann_info

    def parse_data_info(self, info):
        """
        Parse raw data from dataset
        """
        for instance in info['instances']:
            instance['bbox'][2] = instance['bbox'][2] - instance['bbox'][5] / 2 
            instance['bbox_3d'][2] = instance['bbox_3d'][2] - instance['bbox_3d'][5] / 2 
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
        
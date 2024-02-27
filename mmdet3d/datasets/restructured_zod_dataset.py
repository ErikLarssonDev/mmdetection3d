import mmengine

from mmdet3d.registry import DATASETS
from mmdet3d.structures.bbox_3d.lidar_box3d import LiDARInstance3DBoxes
from .det3d_dataset import Det3DDataset
import numpy as np
import torch

currentmaxpoint = [0, 0, 0]
currentminpoint = [100, 100, 100]

@DATASETS.register_module()
class ZodDatasetRestruct(Det3DDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    # replace with all the classes in customized pkl info file
        self.printed = False
    METAINFO = {
        'classes': ['Vehicle', 'VulnerableVehicle', 'Pedestrian', 'Animal', 'PoleObject', 'TrafficBeacon', 'TrafficSign', 'TrafficSignal', 'TrafficGuide', 'DynamicBarrier', 'Unclear'],
        'object_range': [-25, 0, -5, 25, 245, 3]
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
        # ann_info = self._remove_dontcare(ann_info)
        ann_info = self.filter_annotations_on_range(ann_info)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            ann_info['gt_bboxes_3d'],
            origin=(0.5, 0.5, 0.5)
        ).convert_to(self.box_mode_3d)


        ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        print(f"Training on data {ann_info}")
        return ann_info

    def parse_data_info(self, info):
        """
        Parse raw data from dataset
        """
        # for instance in info['instances']:
        #     instance['bbox'][2] = instance['bbox'][2] - instance['bbox'][5] / 2 
        #     instance['bbox_3d'][2] = instance['bbox_3d'][2] - instance['bbox_3d'][5] / 2 
        info = super().parse_data_info(info)
        return info

    def prepare_data(self, idx):
        data = super().prepare_data(idx)
        # maxx = torch.max(data['inputs']['points'][:,0])
        # maxy = torch.max(data['inputs']['points'][:,1])
        # maxz = torch.max(data['inputs']['points'][:,2])
        # minx = torch.min(data['inputs']['points'][:,0])
        # miny = torch.min(data['inputs']['points'][:,1])
        # minz = torch.min(data['inputs']['points'][:,2])
        # if maxx > currentmaxpoint[0]:
        #     currentmaxpoint[0] = maxx
        #     print(f"Max x value increased. Current max and min are {currentmaxpoint} and {currentminpoint}")
        # if maxy > currentmaxpoint[1]:
        #     currentmaxpoint[1] = maxy
        #     print(f"Max y value increased. Current max and min are {currentmaxpoint} and {currentminpoint}")
        # if maxz > currentmaxpoint[2]:
        #     currentmaxpoint[2] = maxz
        #     print(f"Max z value increased. Current max and min are {currentmaxpoint} and {currentminpoint}")
        # if minx < currentminpoint[0]:
        #     currentminpoint[0] = minx
        #     print(f"Min x value decreased. Current max and min are {currentmaxpoint} and {currentminpoint}")
        # if miny < currentminpoint[1]:
        #     currentminpoint[1] = miny
        #     print(f"Min y value decreased. Current max and min are {currentmaxpoint} and {currentminpoint}")
        # if minz < currentminpoint[2]:
        #     currentminpoint[2] = minz
        #     print(f"Min z value decreased. Current max and min are {currentmaxpoint} and {currentminpoint}")
        # if not self.printed:
        #     print(f"Loading data: from idx {idx} points have length {len(data['inputs']['points'])}")
        #     self.printed = True
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
        filtered_annotation_instances = []
        for instance, keep in zip(ann_info['instances'], filter_mask):
            if keep:
                filtered_annotation_instances.append(instance)
        filtered_annotations['instances'] = filtered_annotation_instances
        return filtered_annotations
        
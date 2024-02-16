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
        'classes': ['Vehicle', 'VulnerableVehicle', 'Pedestrian', 'Animal', 'PoleObject', 'TrafficBeacon', 'TrafficSign', 'TrafficSignal', 'TrafficGuide', 'DynamicBarrier', 'Unclear']
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
            ann_info = dict()
            # empty instance
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)

        # filter the gt classes not used in training
        ann_info = self._remove_dontcare(ann_info)
        gt_bboxes_3d = LiDARInstance3DBoxes(ann_info['gt_bboxes_3d'])
        ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        return ann_info
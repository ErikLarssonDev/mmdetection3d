import warnings
from copy import deepcopy

import mmcv

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Compose


@PIPELINES.register_module()
class MultiScaleFlipAug3D(object):
    """Test-time augmentation with multiple scales and flipping

    Args:
        transforms (list[dict]): Transforms to apply in each augmentation.
        img_scale (tuple | list[tuple]: Images scales for resizing.
        pts_scale_ratio (float | list[float]): Points scale ratios for
            resizing.
        flip (bool): Whether apply flip augmentation. Default: False.
        flip_direction (str | list[str]): Flip augmentation directions,
            options are "horizontal" and "vertical". If flip_direction is list,
            multiple flip augmentations will be applied.
            It has no effect when flip == False. Default: "horizontal".
    """

    def __init__(self,
                 transforms,
                 img_scale,
                 pts_scale_ratio,
                 flip=False,
                 flip_direction='horizontal'):
        self.transforms = Compose(transforms)
        self.img_scale = img_scale if isinstance(img_scale,
                                                 list) else [img_scale]
        self.pts_scale_ratio = pts_scale_ratio \
            if isinstance(pts_scale_ratio, list) else[float(pts_scale_ratio)]

        assert mmcv.is_list_of(self.img_scale, tuple)
        assert mmcv.is_list_of(self.pts_scale_ratio, float)

        self.flip = flip
        self.flip_direction = flip_direction if isinstance(
            flip_direction, list) else [flip_direction]
        assert mmcv.is_list_of(self.flip_direction, str)
        if not self.flip and self.flip_direction != ['horizontal']:
            warnings.warn(
                'flip_direction has no effect when flip is set to False')
        if (self.flip
                and not any([t['type'] == 'RandomFlip' for t in transforms])):
            warnings.warn(
                'flip has no effect when RandomFlip is not in transforms')

    def __call__(self, results):
        aug_data = []
        flip_aug = [False, True] if self.flip else [False]
        for scale in self.img_scale:
            for pts_scale_ratio in self.pts_scale_ratio:
                for flip in flip_aug:
                    for direction in self.flip_direction:
                        # results.copy will cause bug since it is shallow copy
                        _results = deepcopy(results)
                        _results['scale'] = scale
                        _results['flip'] = flip
                        _results['pcd_scale_factor'] = pts_scale_ratio
                        _results['flip_direction'] = direction
                        data = self.transforms(_results)
                        aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'img_scale={self.img_scale}, flip={self.flip}, '
        repr_str += f'pts_scale_ratio={self.pts_scale_raio}, '
        repr_str += f'flip_direction={self.flip_direction})'
        return repr_str
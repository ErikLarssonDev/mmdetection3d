# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from os import path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import torch
import numba
from mmengine import load
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log

from mmdet3d.evaluation import kitti_eval, do_eval
from mmdet3d.registry import METRICS
from mmdet3d.structures import (Box3DMode, CameraInstance3DBoxes,
                                LiDARInstance3DBoxes, points_cam2img)


@METRICS.register_module()
class ZodMetric(BaseMetric):
    """Kitti evaluation metric.

    Args:
        ann_file (str): Annotation file path.
        metric (str or List[str]): Metrics to be evaluated. Defaults to 'bbox'.
        pcd_limit_range (List[float]): The range of point cloud used to filter
            invalid predicted boxes. Defaults to [0, -40, -3, 70.4, 40, 0.0].
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix will
            be used instead. Defaults to None.
        pklfile_prefix (str, optional): The prefix of pkl files, including the
            file path and the prefix of filename, e.g., "a/b/prefix". If not
            specified, a temp file will be created. Defaults to None.
        default_cam_key (str): The default camera for lidar to camera
            conversion. By default, KITTI: 'CAM2', Waymo: 'CAM_FRONT'.
            Defaults to 'CAM2'.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result to a
            specific format and submit it to the test server.
            Defaults to False.
        submission_prefix (str, optional): The prefix of submission data. If
            not specified, the submission data will not be generated.
            Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 ann_file: str,
                 metric: Union[str, List[str]] = 'bbox',
                 pcd_limit_range: List[float] = [-0, -50, -5, 250, 50, 3],
                 prefix: Optional[str] = None,
                 pklfile_prefix: Optional[str] = None,
                 default_cam_key: str = 'CAM2',
                 format_only: bool = False,
                 submission_prefix: Optional[str] = None,
                 collect_device: str = 'cpu',
                 backend_args: Optional[dict] = None) -> None:
        self.default_prefix = 'Kitti metric'
        super(ZodMetric, self).__init__(
            collect_device=collect_device, prefix=prefix)
        self.pcd_limit_range = pcd_limit_range
        self.ann_file = ann_file
        self.pklfile_prefix = pklfile_prefix
        self.format_only = format_only
        if self.format_only:
            assert submission_prefix is not None, 'submission_prefix must be '
            'not None when format_only is True, otherwise the result files '
            'will be saved to a temp directory which will be cleaned up at '
            'the end.'

        self.submission_prefix = submission_prefix
        self.default_cam_key = default_cam_key
        self.backend_args = backend_args

        allowed_metrics = ['bbox', 'img_bbox', 'mAP', 'LET_mAP']
        self.metrics = metric if isinstance(metric, list) else [metric]
        for metric in self.metrics:
            if metric not in allowed_metrics:
                raise KeyError("metric should be one of 'bbox', 'img_bbox', "
                               f'but got {metric}.')

   
    def compute_metrics(self, results: List) -> Dict:
        """
        results: [
            dict:
                pred_instances_3d:
                    labels_3d
                    scores_3d
                    bboxes_3d
                gt_instances_3d:
                    labels_3d
                    bboxes_3d
        ]
        """
        logger: MMLogger = MMLogger.get_current_instance()
        self.classes = self.dataset_meta['classes']
        print(len(results))
        print(f"CLASSES IN COMPUTE METRICS {self.classes}")
        print(f"PREDS ARE {results[0]['pred_instances_3d']}")
        eval_class(
            dt_annos=results[0]["pred_instances_3d"],
            gt_annos=results[0]["gt_instances_3d"],
            classes=self.classes)
        raise NotImplementedError
        return dict({})
    

    def process(self, data_batch, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Any): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        gt_batch = data_batch["data_samples"]

        for pred_sample, gt_frame in zip(data_samples, gt_batch):
            gt_annotation = gt_frame.eval_ann_info
            result = dict()
            pred_3d = pred_sample['pred_instances_3d']
            gt_3d = dict(
                labels_3d = gt_annotation["gt_bboxes_labels"],
                bboxes_3d = gt_annotation["gt_bboxes_3d"]
            )
            for attr_name in pred_3d:
                pred_3d[attr_name] = pred_3d[attr_name].to('cpu')
            result['pred_instances_3d'] = pred_3d
            result['gt_instances_3d'] = gt_3d
            self.results.append(result)
            
        return
    

def eval_class(gt_annos,
            dt_annos,
            classes,
            min_overlaps = None,
            num_parts=200):
    """
    dt_annos:
        labels_3d
        scores_3d
        bboxes_3d
    gt_annos:
        labels_3d
        bboxes_3d
    classes: A list of str in order
    """
    # assert len(gt_annos) == len(dt_annos)
    if min_overlaps is None:
        min_overlaps = [0.25, 0.5, 0.7]
    dt_boxes = np.concatenate([annot[np.newaxis, ...] for annot in dt_annos['bboxes_3d']])
    gt_boxes = np.concatenate([annot[np.newaxis, ...] for annot in gt_annos['bboxes_3d']])
    dt_labels = np.hstack([annot for annot in dt_annos['labels_3d']])
    dt_scores = np.hstack([annot for annot in dt_annos['scores_3d']])
    gt_labels = np.hstack([annot for annot in gt_annos['labels_3d']])
    print(f"GT_LABELS ARE {gt_labels}")
    print(f"GT BOXES ARE {gt_boxes}")
    overlaps, num_preds, num_gts = calculate_iou_partly(dt_boxes, gt_boxes)
    print(np.max(overlaps))
    best_fit = np.where(overlaps==np.max(overlaps))
    print(best_fit)
    print(f"gt: {gt_boxes[best_fit[1]]} {gt_labels[best_fit[1]]}")
    print(f"pd: {dt_boxes[best_fit[0]]} {dt_labels[best_fit[0]]}")
    print(f"Overlaps {overlaps}")
    num_minoverlap = len(min_overlaps)
    N_SAMPLE_PTS = 41
    num_class = len(classes)

    precision = np.zeros(
        [num_class, num_minoverlap, N_SAMPLE_PTS])
    recall = np.zeros(
        [num_class, N_SAMPLE_PTS])
    aos = np.zeros([num_class, num_minoverlap, N_SAMPLE_PTS])
    for class_index, current_class in enumerate(classes):
        for k, min_overlap in enumerate(min_overlaps):
            thresholdss = []
            rets = compute_statistics_jit(
                overlaps,
                gt_boxes,
                dt_boxes,
                current_class_label=class_index,
                dt_pred_labels=dt_labels,
                gt_pred_labels=gt_labels,
                dt_scores=dt_scores,
                min_overlap=min_overlap,
                thresh=0.0,
                compute_fp=True)
            
            tp, fp, fn, similarity, thresholds = rets
            print(f"{current_class}, IoU >= {min_overlap}")
            print(f"tp {tp}", f"fp {fp}", f"fn {fn}", f"similarity {similarity}", f"thresholds {thresholds}", sep="\n")

def calculate_iou_partly(dt_boxes, gt_boxes):

    # assert len(dt_annos) == len(gt_annos)
    total_gt_num = len(gt_boxes)
    total_dt_num = len(dt_boxes)
    # split_parts = get_split_parts(num_examples, num_parts)
    # parted_overlaps = []
    example_idx = 0


    print(dt_boxes.shape)
    print(gt_boxes.shape)
    
    overlaps = d3_box_overlap(dt_boxes,
                                gt_boxes).astype(np.float64)

    return overlaps, total_dt_num, total_gt_num

def get_split_parts(num, num_part):
    if num % num_part == 0:
        same_part = num // num_part
        return [same_part] * num_part
    else:
        same_part = num // (num_part - 1)
        remain_num = num % (num_part - 1)
        return [same_part] * (num_part - 1) + [remain_num]

def d3_box_overlap(boxes, qboxes, criterion=-1):
    from mmdet3d.evaluation.functional.kitti_utils.rotate_iou import rotate_iou_gpu_eval
    # Get the BEV IoU
    rinc = rotate_iou_gpu_eval(boxes[:, [0, 1, 3, 4, 6]], #Changed from 0, 2, 3, 5, 6
                               qboxes[:, [0, 1, 3, 4, 6]], -1)
    # Commenting out this part since it only seems to be for CAMERA,
    # not lidar
    # d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
    return rinc

# @numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
    # ONLY support overlap in CAMERA, not lidar.
    # TODO: change to use prange for parallel mode, should check the difference
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in numba.prange(N):
        for j in numba.prange(K):
            if rinc[i, j] > 0:
                # iw = (min(boxes[i, 1] + boxes[i, 4], qboxes[j, 1] +
                #         qboxes[j, 4]) - max(boxes[i, 1], qboxes[j, 1]))
                iw = (
                    min(boxes[i, 1], qboxes[j, 1]) -
                    max(boxes[i, 1] - boxes[i, 4],
                        qboxes[j, 1] - qboxes[j, 4]))

                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = inc
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0
                    
# @numba.jit(nopython=True)
def compute_statistics_jit(overlaps,
                           gt_datas,
                           dt_datas,
                           min_overlap,
                           current_class_label,
                           dt_pred_labels,
                           gt_pred_labels,
                           dt_scores,
                           thresh=0,
                           compute_fp=False,
                           compute_aos=False):

    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    ignored_det = [0 for _ in range(det_size)]
    for pindex, pred_label in enumerate(dt_pred_labels):
        if not pred_label == current_class_label:
            ignored_det[pindex] = -1 # -1 seems to be wrong class prediction (based on clean_data function) which should not be counted when evaluating a certain class 
    # gt_bboxes = gt_datas[:, :4]
    ignored_gt = [0] * gt_size
    for pindex, pred_label in enumerate(gt_pred_labels):
        if not pred_label == current_class_label:
            ignored_gt[pindex] = -1

    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size
    if compute_fp:
        for i in range(det_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    # thresholds = [0.0]
    # delta = [0.0]
    thresholds = np.zeros((gt_size, ))
    thresh_idx = 0
    delta = np.zeros((gt_size, ))
    delta_idx = 0
    for i in range(gt_size):
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0
        assigned_ignored_det = False
        for j in range(det_size):
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            if (ignored_threshold[j]):
                continue
            overlap = overlaps[j, i]
            dt_score = dt_scores[j]
            if (not compute_fp and (overlap > min_overlap)
                    and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score
            elif (compute_fp and (overlap > min_overlap)
                  and (overlap > max_overlap or assigned_ignored_det)
                  and ignored_det[j] == 0):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (compute_fp and (overlap > min_overlap)
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[j] == 1):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
        elif ((valid_detection != NO_DETECTION)
              and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            tp += 1
            # thresholds.append(dt_scores[det_idx])
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            assigned_detection[det_idx] = True
    if compute_fp:
        for i in range(det_size):
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1
        # # I think this was to not count ignored assigns as false positives
        # nstuff = 0
        # if metric == 0:
        #     for i in range(dc_bboxes.shape[0]):
        #         for j in range(det_size):
        #             if (assigned_detection[j]):
        #                 continue
        #             if (ignored_det[j] == -1 or ignored_det[j] == 1):
        #                 continue
        #             if (ignored_threshold[j]):
        #                 continue
        #             if overlaps[j, i] > min_overlap:
        #                 assigned_detection[j] = True
        #                 nstuff += 1
        # fp -= nstuff
        if compute_aos:
            tmp = np.zeros((fp + delta_idx, ))
            # tmp = [0] * fp
            for i in range(delta_idx):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
                # tmp.append((1.0 + np.cos(delta[i])) / 2.0)
            # assert len(tmp) == fp + tp
            # assert len(delta) == tp
            if tp > 0 or fp > 0:
                similarity = np.sum(tmp)
            else:
                similarity = -1
    return tp, fp, fn, similarity, thresholds[:thresh_idx]
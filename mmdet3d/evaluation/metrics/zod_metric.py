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
        min_overlaps = [0.1, 0.3, 0.5, 0.7, 0.9]
        range_intervals = [[0,400], [0,100], [100, 150], [150, 250]]
        class_tps = np.zeros([len(range_intervals), len(self.classes), len(min_overlaps)])
        class_fps = np.zeros([len(range_intervals), len(self.classes), len(min_overlaps)])
        class_fns = np.zeros([len(range_intervals), len(self.classes), len(min_overlaps)])

        class_preds = np.zeros([len(range_intervals), len(self.classes)])
        class_gts = np.zeros([len(range_intervals), len(self.classes)])

        for range_interval_index, range_interval in enumerate(range_intervals):
            for image in results:

                dt_instances_in_range = filter_on_range(image["pred_instances_3d"], range_interval)
                gt_instances_in_range = filter_on_range(image["gt_instances_3d"], range_interval)
                
                tps, fps, fns, num_gts, num_preds, image_gt_n_class, image_dt_n_class = eval_image(
                    dt_annos=dt_instances_in_range,
                    gt_annos=gt_instances_in_range,
                    classes=self.classes,
                    min_overlaps=min_overlaps
                )
                
                class_tps[range_interval_index, :, :] += tps
                class_fps[range_interval_index, :, :] += fps
                class_fns[range_interval_index, :, :] += fns
                class_gts[range_interval_index, :] += image_gt_n_class
                class_preds[range_interval_index, :] += image_dt_n_class
            
        
        precision = class_tps / (class_tps + class_fps + 1e-6) # 1e-6 to avoid division by zero
        recall = class_tps / (class_tps + class_fns + 1e-6) # 1e-6 to avoid division by zero
        # accuracy = class_tps / (total_preds + 1e-6) # TODO check if this is correct
        ap = np.mean(precision, axis=0)
        m_ap = np.mean(ap)
        return dict(
            precision=precision,
            recall=recall,
            ap=ap,
            map=m_ap,
            class_gts=class_gts,
            class_preds=class_preds,
        )
    

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
            gt_annotation = gt_frame.gt_instances_3d # .eval_ann_info
            result = dict()
            # pred_3d = pred_sample['pred_instances_3d']
            # print(gt_annotation)
            # gt_3d = dict(
            #     labels_3d = gt_annotation.labels_3d, #["gt_bboxes_labels"]
            #     bboxes_3d = gt_annotation.bboxes_3d #["gt_bboxes_3d"]
            # )
            # for attr_name in pred_3d:
            #     pred_3d[attr_name] = pred_3d[attr_name].to('cpu')
            result['pred_instances_3d'] = pred_sample['pred_instances_3d']
            result['gt_instances_3d'] = gt_annotation
            self.results.append(result)
            
        return
    

def filter_on_range(instances, range_interval):
    filtered_instances = dict({})
    boxes_center = instances['bboxes_3d'].center.cpu().numpy()
    center_distances = np.sqrt(np.sum(boxes_center** 2, axis=1))
    filter_mask = np.all([center_distances > range_interval[0], center_distances < range_interval[1]], axis=0)
    
    for key in instances.keys():
        filtered_instances[key] = (instances[key][filter_mask])
    
    return filtered_instances

def eval_image(gt_annos,
            dt_annos,
            classes,
            min_overlaps):
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
    dt_per_class = np.zeros([len(classes)], dtype=np.int64)
    gt_per_class = np.zeros([len(classes)], dtype=np.int64)

    if not len(gt_annos['bboxes_3d']) == 0: 
        gt_boxes = np.concatenate([annot[np.newaxis, ...] for annot in gt_annos['bboxes_3d'].cpu()])
        gt_labels = np.hstack([annot for annot in gt_annos['labels_3d']])
        for i, name in enumerate(classes):
            gt_per_class[i] = np.sum(gt_labels == i)

    
    if not len(dt_annos['bboxes_3d']) == 0:
        dt_boxes = np.concatenate([annot[np.newaxis, ...] for annot in dt_annos['bboxes_3d'].cpu()])
        dt_labels = np.hstack([annot for annot in dt_annos['labels_3d']])
        dt_scores = np.hstack([annot for annot in dt_annos['scores_3d']]) # TODO if this is the class score or the object score, make certain that it is the score for the predicted class 
        for i, name in enumerate(classes):
            dt_per_class[i] = np.sum(dt_labels == i)


     # Return in case detections or ground truths don't exist
    if len(gt_annos['bboxes_3d']) == 0 or len(dt_annos['bboxes_3d']) == 0:
        tps = np.zeros([len(classes), len(min_overlaps)])
        fns = gt_per_class # if there are no detections, all ground truths are false negatives, if there also are no ground truths, this will be zero
        fps = dt_per_class # if there are no ground truths, all detections are false positives, if there also are no detections, this will be zero
        return tps, fps, fns, len(gt_annos['bboxes_3d']), len(gt_annos['bboxes_3d']), gt_per_class, dt_per_class


    overlaps = calculate_iou_partly(dt_boxes, gt_boxes)
    num_minoverlap = len(min_overlaps)
    num_class = len(classes)

    tps = np.zeros(
        [num_class, num_minoverlap])
    fps = np.zeros(
        [num_class, num_minoverlap])
    fns = np.zeros(
        [num_class, num_minoverlap])
    
    thresholdss = []
    rets = compute_statistics_jit(
        overlaps,
        min_overlaps=min_overlaps,
        num_classes=num_class,
        pred_labels=dt_labels,
        gt_labels=gt_labels)
        
    tps, fps, fns = rets

    return tps, fps, fns, len(gt_labels), len(dt_labels), gt_per_class, dt_per_class

def calculate_iou_partly(dt_boxes, gt_boxes):

    # assert len(dt_annos) == len(gt_annos)
    # split_parts = get_split_parts(num_examples, num_parts)
    # parted_overlaps = []


    
    overlaps = d3_box_overlap(dt_boxes,
                                gt_boxes).astype(np.float64)

    return overlaps
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
                           min_overlaps,
                           num_classes,
                           pred_labels,
                           gt_labels):

    tps = np.zeros([num_classes, len(min_overlaps)])
    fps = np.zeros([num_classes, len(min_overlaps)])
    fns = np.zeros([num_classes, len(min_overlaps)])
    for iou_threshold_index, iou_threshold in enumerate(min_overlaps):
        prediction_indexes_to_match = list(range(len(pred_labels))) #TODO change to deque
        matched_gts = np.full(len(gt_labels), -1) # -1 means that the ground truth has not been matched to a detection, each value in the list corresponds to a detection index
        while len(prediction_indexes_to_match) > 0:
            prediction_index = prediction_indexes_to_match.pop(0)
            iou_sorted_gt_indexes = np.argsort(overlaps[prediction_index, :])[::-1]

            for candidate_gt_index in iou_sorted_gt_indexes:
                if overlaps[prediction_index, candidate_gt_index] < iou_threshold:
                    break
                previous_match = matched_gts[candidate_gt_index]
                if previous_match == -1: # The candidate gt is not matched to a prediction
                    matched_gts[candidate_gt_index] = prediction_index
                    break
                elif overlaps[previous_match, candidate_gt_index] < overlaps[prediction_index, candidate_gt_index]: #Current prediction is a better match than previous prediction
                    matched_gts[candidate_gt_index] = prediction_index
                    prediction_indexes_to_match.append(previous_match)
                    break
            
        for gt_index, dt_index in enumerate(matched_gts):
            if dt_index == -1:
                fns[gt_labels[gt_index], iou_threshold_index] += 1

            elif pred_labels[dt_index] == gt_labels[gt_index]:
                tps[gt_labels[gt_index], iou_threshold_index] += 1

            else:
                fps[pred_labels[dt_index], iou_threshold_index] += 1
                fns[gt_labels[gt_index], iou_threshold_index] += 1

        for prediction_index in range(len(pred_labels)):
            if prediction_index not in matched_gts:
                fps[pred_labels[prediction_index], iou_threshold_index] += 1

    return tps, fps, fns

# @numba.jit(nopython=True)
def compute_statistics_jit_old(overlaps,
                           gt_datas,
                           dt_datas,
                           min_overlaps,
                           num_classes,
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
    for i in range(gt_size): # Every ground truth is matched to a detection (det_idx)
        det_idx = -1
        valid_detection = NO_DETECTION 
        max_overlap = 0 # The highest overlap found so far
        assigned_ignored_det = False
        for j in range(det_size):
            if (ignored_det[j] == -1): # The detection is not of the class we are evaluating
                continue
            if (assigned_detection[j]): # The detection has already been assigned to a ground truth
                continue
            if (ignored_threshold[j]): # The detection has a score below the threshold
                continue
            overlap = overlaps[j, i]
            dt_score = dt_scores[j]
            if (not compute_fp and (overlap > min_overlap) # 
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
        elif valid_detection != NO_DETECTION: # We have a detection which should be valid since the ground truth is not ignored
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

        if compute_aos:
            tmp = np.zeros((fp + delta_idx, ))
            # tmp = [0] * fp
            for i in range(delta_idx):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0

            if tp > 0 or fp > 0:
                similarity = np.sum(tmp)
            else:
                similarity = -1
    return tp, fp, fn, similarity, thresholds[:thresh_idx]
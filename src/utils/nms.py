from collections import defaultdict

import numpy as np
""" Parts copied from https://github.com/ZFTurbo/Weighted-Boxes-Fusion
MIT License

Copyright (c) 2020 ZFTurbo (Roman Solovyev)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def iou_stats(boxes, scores, labels, iou_thr=0.5, weights=None):
    boxes = np.concatenate(boxes)
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)

    # Fix coordinates and removed zero area boxes (prepare boxes from ensemble boxes)
    # boxes, scores, labels = prepare_boxes(boxes, scores, labels)
    unique_labels = np.unique(labels)
    final_boxes = []
    final_scores = []
    final_labels = []
    final_metrics = defaultdict(list)

    for l in unique_labels:
        condition = (labels == l)
        boxes_by_label = boxes[condition]
        scores_by_label = scores[condition]
        labels_by_label = np.array([l] * len(boxes_by_label))

        keep, metrics = nms_float_fast_new(boxes_by_label, scores_by_label, thresh=iou_thr)

        final_boxes.append(boxes_by_label[keep])
        final_scores.append(scores_by_label[keep])
        final_labels.append(labels_by_label[keep])
        for met, val in metrics.items():
            final_metrics[met].append(val)

    final_boxes = np.concatenate(final_boxes)
    final_scores = np.concatenate(final_scores)
    final_labels = np.concatenate(final_labels)

    for met, val in final_metrics.items():
        final_metrics[met] = np.concatenate(val)

    return final_boxes, final_scores, final_labels, final_metrics


def nms_float_fast_new(dets, scores, thresh):
    """
    # It's different from original nms because we have float coordinates on range [0; 1]
    :param dets: numpy array of boxes with shape: (N, 5). Order: x1, y1, x2, y2, score. All variables in range [0; 1]
    :param thresh: IoU value for boxes
    :return: index of boxes to keep
    :return: number of suppressed dets by each det
    """

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    non_intersection = iou_areas(dets, scores)
    prod_suppressed_rem_area = np.prod(non_intersection, axis=0)  # inv ?
    prod_remaining_area = np.prod(non_intersection, axis=1)
    min_suppressed_rem_area = np.min(non_intersection, axis=0) # inv ?
    min_remaining_area = np.min(non_intersection, axis=1)
    keep = []
    count_supp = []
    if thresh == 1.:
        keep = order
        count_supp = [0] * len(order)
    else:
        while order.size > 0:
            i = order[0]
            len_order = order.size - 1
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
            count_supp.append(len_order - order.size)

    metrics = {"nms_count": count_supp,
               "rel_area": areas[keep],
               "prod_remaining_area": prod_remaining_area[keep],
               "prod_suppressed_rem_area": prod_suppressed_rem_area[keep],
               "prod_suppressed_area": 1. - prod_suppressed_rem_area[keep],
               "min_remaining_area": min_remaining_area[keep],
               "min_suppressed_rem_area": min_suppressed_rem_area[keep],
               "min_suppressed_area": 1. - min_suppressed_rem_area[keep]}
    return keep, metrics


def iou_areas(boxes, scores):

    # Compute the areas of the bounding boxes
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    areas = (y2 - y1) * (x2 - x1)

    # Compute the coordinates of the intersection rectangle for all pairs of bounding boxes
    y1 = np.maximum(boxes[:, 0][:, None], boxes[:, 0][None, :])
    x1 = np.maximum(boxes[:, 1][:, None], boxes[:, 1][None, :])
    y2 = np.minimum(boxes[:, 2][:, None], boxes[:, 2][None, :])
    x2 = np.minimum(boxes[:, 3][:, None], boxes[:, 3][None, :])

    # Compute the area of the intersection rectangle for all pairs of bounding boxes
    w = np.maximum(0, x2 - x1)
    h = np.maximum(0, y2 - y1)
    intersection_areas = w * h

    # Compute the ratio of the intersection to union for all pairs of bounding boxes
    ratios = intersection_areas / (areas[:, None] + areas[None, :] - intersection_areas)
    non_intersection = 1. - ratios
    non_intersection[np.logical_not(np.tri(len(areas), k=-1))] = 1.

    return non_intersection

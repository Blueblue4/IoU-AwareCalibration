""" Source: https://github.com/ZFTurbo/Weighted-Boxes-Fusion minor fixes and modifications
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

import functools
import io
import itertools
from multiprocessing import Pool

from src.utils.nms import iou_stats
from contextlib import redirect_stdout
from typing import List
import ensemble_boxes
import inspect
import numpy as np

from src.utils.coco_helpers import coco_dets_to_np, np_to_coco_dets


def nms_method(boxes, scores, labels, method=3, iou_thr=0.5, sigma=0.5, thresh=0.01, weights=None):
    """
    :param boxes: list of boxes predictions from each model, each box is 4 numbers.
    It has 3 dimensions (models_number, model_preds, 4)
    Order of boxes: x1, y1, x2, y2. We expect float normalized coordinates [0; 1]
    :param scores: list of scores for each model
    :param labels: list of labels for each model
    :param method: 1 - linear soft-NMS, 2 - gaussian soft-NMS, 3 - standard NMS
    :param iou_thr: IoU value for boxes to be a match
    :param sigma: Sigma value for SoftNMS
    :param thresh: threshold for boxes to keep (important for SoftNMS)
    :param weights: list of weights for each model. Default: None, which means weight == 1 for each model

    :return: boxes: boxes coordinates (Order of boxes: x1, y1, x2, y2).
    :return: scores: confidence scores
    :return: labels: boxes labels
    """

    # If weights are specified
    if weights is not None:
        if len(boxes) != len(weights):
            print('Incorrect number of weights: {}. Must be: {}. Skip it'.format(len(weights), len(boxes)))
        else:
            weights = np.array(weights)
            for i in range(len(weights)):
                scores[i] = (np.array(scores[i]) * weights[i]) / weights.sum()

    # We concatenate everything
    boxes = np.concatenate(boxes)
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)

    # Fix coordinates and removed zero area boxes
    #boxes, scores, labels = prepare_boxes(boxes, scores, labels)

    # Run NMS independently for each label
    unique_labels = np.unique(labels)
    final_boxes = []
    final_scores = []
    final_labels = []
    for l in unique_labels:
        condition = (labels == l)
        boxes_by_label = boxes[condition]
        scores_by_label = scores[condition]
        labels_by_label = np.array([l] * len(boxes_by_label))

        keep, scores_keep, bboxes_keep = cpu_soft_nms_float(boxes_by_label.copy(),
                                                            scores_by_label.copy(),
                                                            Nt=iou_thr,
                                                            sigma=sigma,
                                                            thresh=thresh,
                                                            method=method)

        final_boxes.append(bboxes_keep)
        final_scores.append(scores_keep)
        final_labels.append(labels_by_label[keep])
    final_boxes = np.concatenate(final_boxes)
    final_scores = np.concatenate(final_scores)
    final_labels = np.concatenate(final_labels)

    return final_boxes, final_scores, final_labels


def py_soft_nms(dets, method='linear', iou_thr=0.3, sigma=0.5, score_thr=0.001):
    """Pure python implementation of soft NMS as described in the paper
    `Improving Object Detection With One Line of Code`_.
    Args:
        dets (numpy.array): Detection results with shape `(num, 5)`,
            data in second dimension are [x1, y1, x2, y2, score] respectively.
        method (str): Rescore method. Only can be `linear`, `gaussian`
            or 'greedy'.
        iou_thr (float): IOU threshold. Only work when method is `linear`
            or 'greedy'.
        sigma (float): Gaussian function parameter. Only work when method
            is `gaussian`.
        score_thr (float): Boxes that score less than the.
    Returns:
        numpy.array: Retained boxes.
    .. _`Improving Object Detection With One Line of Code`:
        https://arxiv.org/abs/1704.04503
    """
    if method not in ('linear', 'gaussian', 'greedy'):
        raise ValueError('method must be linear, gaussian or greedy')

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # expand dets with areas, and the second dimension is
    # x1, y1, x2, y2, score, area
    dets = np.concatenate((dets, areas[:, None]), axis=1)

    retained_box = []
    while dets.size > 0:
        max_idx = np.argmax(dets[:, 4], axis=0)
        dets[[0, max_idx], :] = dets[[max_idx, 0], :]
        retained_box.append(dets[0, :-1])

        xx1 = np.maximum(dets[0, 0], dets[1:, 0])
        yy1 = np.maximum(dets[0, 1], dets[1:, 1])
        xx2 = np.minimum(dets[0, 2], dets[1:, 2])
        yy2 = np.minimum(dets[0, 3], dets[1:, 3])

        w = np.maximum(xx2 - xx1 + 1, 0.0)
        h = np.maximum(yy2 - yy1 + 1, 0.0)
        inter = w * h
        iou = inter / (dets[0, 5] + dets[1:, 5] - inter)

        if method == 'linear':
            weight = np.ones_like(iou)
            weight[iou > iou_thr] -= iou[iou > iou_thr]
        elif method == 'gaussian':
            weight = np.exp(-(iou * iou) / sigma)
        else:  # traditional nms
            weight = np.ones_like(iou)
            weight[iou > iou_thr] = 0

        dets[1:, 4] *= weight
        retained_idx = np.where(dets[1:, 4] >= score_thr)[0]
        dets = dets[retained_idx + 1, :]

    return np.vstack(retained_box)


def cpu_soft_nms_float(dets, sc, Nt, sigma, thresh, method):
    """
    Based on: https://github.com/DocF/Soft-NMS/blob/master/soft_nms.py
    It's different from original soft-NMS because we have float coordinates on range [0; 1]

    :param dets:   boxes format [x1, y1, x2, y2]
    :param sc:     scores for boxes
    :param Nt:     required iou
    :param sigma:
    :param thresh:
    :param method: 1 - linear soft-NMS, 2 - gaussian soft-NMS, 3 - standard NMS
    :return: index of boxes to keep
    """

    # indexes concatenate boxes with the last column
    N = dets.shape[0]
    indexes = np.array([np.arange(N)])
    dets = np.concatenate((dets, indexes.T), axis=1)

    # the order of boxes coordinate is [y1, x1, y2, x2]
    y1 = dets[:, 1]
    x1 = dets[:, 0]
    y2 = dets[:, 3]
    x2 = dets[:, 2]
    scores = sc
    areas = (x2 - x1) * (y2 - y1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tBD = dets[i, :].copy()
        tscore = scores[i].copy()
        tarea = areas[i].copy()
        pos = i + 1

        #
        if i != N - 1:
            maxscore = np.max(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
        else:
            maxscore = scores[-1]
            maxpos = 0
        if tscore < maxscore:
            dets[i, :] = dets[maxpos + i + 1, :]
            dets[maxpos + i + 1, :] = tBD
            tBD = dets[i, :]

            scores[i] = scores[maxpos + i + 1]
            scores[maxpos + i + 1] = tscore
            tscore = scores[i]

            areas[i] = areas[maxpos + i + 1]
            areas[maxpos + i + 1] = tarea
            tarea = areas[i]

        # IoU calculate
        xx1 = np.maximum(dets[i, 1], dets[pos:, 1])
        yy1 = np.maximum(dets[i, 0], dets[pos:, 0])
        xx2 = np.minimum(dets[i, 3], dets[pos:, 3])
        yy2 = np.minimum(dets[i, 2], dets[pos:, 2])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[pos:] - inter)

        # Three methods: 1.linear 2.gaussian 3.original NMS
        if method == 1:  # linear
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = weight[ovr > Nt] - ovr[ovr > Nt]
        elif method == 2:  # gaussian
            weight = np.exp(-(ovr * ovr) / sigma)
        else:  # original NMS
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = 0

        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    inds = dets[:, 4][scores > thresh]
    keep = inds.astype(int)
    return keep, scores[scores > thresh], dets[scores > thresh][:, :-1]


combine_functions = {
    "nms": functools.partial(nms_method, method=3),
    "soft_nms": functools.partial(nms_method, method=2),
    "linear_soft_nms": functools.partial(nms_method, method=1),
    "nmw": ensemble_boxes.non_maximum_weighted,
    "wbf": ensemble_boxes.weighted_boxes_fusion,
    "iou_stats": iou_stats,
}


def combine_boxes_nms(boxes: List[list], image: dict, nms_type: str, nms_settings: dict) -> List[dict]:
    if nms_type == "no_nms":
        return list(itertools.chain.from_iterable(boxes))
    assert nms_type in combine_functions, f"{nms_type} not a valid nms type!"
    if not any(boxes):
        return list()
    boxes_np = coco_dets_to_np(boxes, image)
    with redirect_stdout(io.StringIO()):
        merged_boxes = assign_kwargs_to_function(combine_functions[nms_type],
                                                 *boxes_np, **nms_settings
                                                 )
    return np_to_coco_dets(merged_boxes, image)


def assign_kwargs_to_function(function, *args, **kwargs):
    """ Drops kwargs that are not used by the function.
    """
    target_args = inspect.getfullargspec(function)
    all_args = target_args.kwonlyargs + target_args.args
    return function(*args, **{k: v for k, v in kwargs.items() if k in all_args})


def apply_nms(cfg, coco_dets, coco_gt, multi_processing=True):
    new_coco_dets = []

    nms_func = functools.partial(combine_boxes_nms, nms_type=cfg["nms"]["run_type"],
                                 nms_settings=cfg["nms"]["settings"])

    dets_imgs = [([coco_det.imgToAnns[img_id] for coco_det in coco_dets], image) for img_id, image in
                 coco_gt.imgs.items()]
    if multi_processing:
        with Pool(4) as pool:
            combined_dets = pool.starmap(nms_func, dets_imgs)
        new_coco_dets = list(itertools.chain.from_iterable(combined_dets))
    else:
        new_coco_dets = [nms_func(*dets_img) for dets_img in dets_imgs]
    with redirect_stdout(io.StringIO()):
        new_coco_dets = coco_gt.loadRes(new_coco_dets)
    return new_coco_dets

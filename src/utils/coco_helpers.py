import io
import json
import logging
import pathlib
from collections import defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
from typing import Union, Optional, Dict, List, Tuple
import numpy as np
from hydra.utils import to_absolute_path
from pycocotools.coco import COCO
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from src.calib import ece
from src.utils.general import save_dict, DisableLoggerContext

log = logging.getLogger(__name__)
from pycocotools.cocoeval import COCOeval

try:
    import torch
    from fast_coco_eval import COCOeval_fast
except ImportError:
    COCOeval_fast = COCOeval
    log.info(
        "Failed to import COCOeval_fast with dependency torch! Using slower COCOeval."
    )


def dataset_from_image_ids(img_ids: list, coco_ds: COCO) -> COCO:
    with redirect_stdout(io.StringIO()):
        new_coco_ds = COCO(None)
        anns = coco_ds.loadAnns(ids=coco_ds.getAnnIds(imgIds=img_ids))
        new_coco_ds.dataset = {
            "categories": coco_ds.dataset["categories"],
            "annotations": deepcopy(anns),
            "images": coco_ds.loadImgs(img_ids),
        }
        new_coco_ds.createIndex()
    return new_coco_ds


def evaluate_coco(coco_gt: COCO, coco_pred: COCO, calibration_metrics=False, calib_truth_threshold=0.5) -> Dict[str, Union[np.ndarray, float]]:
    with redirect_stdout(io.StringIO()), DisableLoggerContext():
        coco_eval = COCOeval_fast(coco_gt, coco_pred, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        maps_per_class = map_per_class(coco_eval)
        map50s_per_class = map_per_class(coco_eval, iou_thr=0.5)
        coco_eval.summarize()
    eval_vals = [metric * 100. for metric in coco_eval.stats]
    eval_cats = [
        "mAP",
        "mAP50",
        "mAP75",
        "mAP_small",
        "mAP_medium",
        "mAP_large",
        "mAR_1",
        "mAR_10",
        "mAR_100",
        "mAR_small",
        "mAR_medium",
        "mAR_large",
    ]
    cal_metrics = dict()
    if calibration_metrics:
        if isinstance(calib_truth_threshold, list):
            for thres in calib_truth_threshold:
                metrics_thres = eval_calibration_metrics(coco_pred, coco_gt, thres)
                cal_metrics.update({k + f"@{thres:0.2f}": v for k, v in metrics_thres.items()})
        else:
            cal_metrics = eval_calibration_metrics(coco_pred, coco_gt, calib_truth_threshold)
    return {
        **dict(zip(eval_cats, eval_vals)),
        **{"mAP_per_class": maps_per_class, "mAP50_per_class": map50s_per_class},
        **cal_metrics,
    }


def eval_calibration_metrics(coco_pred, coco_gt, calib_truth_threshold):
    labeled_dets = extract_labeled_detections(coco_pred, coco_gt,
                                              iou_threshold=calib_truth_threshold, all_class=True)
    cal_metrics = dict()
    cal_metrics["ECE"] = 100. * ece(labeled_dets["all"]["true_positives"], labeled_dets["all"]["confidences"])
    cal_metrics["ACE"] = 100. * ece(labeled_dets["all"]["true_positives"], labeled_dets["all"]["confidences"],
                             strategy="quantile")
    cal_metrics["SCE"] = 100. * np.mean([ece(labeled_dets[cls]["true_positives"], labeled_dets[cls]["confidences"])
                                  for cls in labeled_dets.keys() if cls != "all"])
    cal_metrics["NLL"] = log_loss(y_true=labeled_dets["all"]["true_positives"], y_pred=labeled_dets["all"]["confidences"],
                                  labels=[True, False])
    return cal_metrics


def evaluate_coco_per_area(area_ranges, coco_gt, pred):
    with redirect_stdout(io.StringIO()), DisableLoggerContext():
        coco_eval = COCOeval_fast(coco_gt, pred, iouType="bbox")
        coco_eval.params.areaRng = area_ranges
        coco_eval.evaluate()
        coco_eval.accumulate()
        map_per_area = np.zeros((len(area_ranges)))
        prec = coco_eval.eval['precision']
        for idx, _ in enumerate(area_ranges):
            prec_area = prec[:, :, :, idx, -1]
            map_per_area[idx] = np.mean(prec_area[prec_area > -1])
    return map_per_area


def map_per_class(
    coco_eval: COCOeval,
    iou_thr: Optional[float] = None,
    area_rng: str = "all",
    max_dets: int = 100,
) -> np.ndarray:
    p = coco_eval.eval["params"]
    aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == area_rng]
    mind = [i for i, mDet in enumerate(p.maxDets) if mDet == max_dets]
    s = coco_eval.eval["precision"]
    maps = list()
    if iou_thr is not None:
        t = np.where(iou_thr == p.iouThrs)[0]
        s = s[t]
    for cat in range(coco_eval.eval["counts"][2]):
        prec = s[:, :, cat, aind, mind]
        maps.append(np.mean(prec[prec > -1]))
    return np.array(maps) * 100.0


def split_train_test(
    coco_pred: COCO, coco_gt: COCO, seed: int = 123, split: float = 0.5
) -> Tuple[COCO, COCO, COCO, COCO]:
    train_img_ids, test_img_ids = train_test_split(
        list(coco_gt.imgs.keys()), train_size=split, random_state=seed
    )
    train_pred = dataset_from_image_ids(train_img_ids, coco_pred)
    train_gt = dataset_from_image_ids(train_img_ids, coco_gt)
    test_pred = dataset_from_image_ids(test_img_ids, coco_pred)
    test_gt = dataset_from_image_ids(test_img_ids, coco_gt)
    return train_pred, train_gt, test_pred, test_gt


def extract_labeled_detections(
    pred: COCO, gt: COCO, iou_threshold=0.5, all_class=False
):
    coco_eval = coco_iou_eval(pred, gt, iou_threshold)
    detections = {categ: defaultdict(lambda: list()) for categ in gt.cats}

    # Order detections and their labels by src categories
    for image_eval in [image for image in coco_eval.evalImgs if image is not None]:

        category_id = image_eval["category_id"]

        valid_det = np.any(np.logical_not(image_eval["dtIgnore"]), axis=0)
        # need to load detections for actual area
        dets = coco_eval.cocoDt.loadAnns(
            np.array(image_eval["dtIds"])[valid_det]
        )
        if len(dets) > 0:
            confidence = np.array([det["score"] for det in dets])
            assert len(confidence.shape) == 1
            detections[category_id]["confidences"].append(confidence)
            tp_ness = np.mean(image_eval["dtMatches"][:, valid_det] > 0, axis=0, dtype=float)
            detections[category_id]["true_positives"].append(
                tp_ness
            )
            for attrib in [attrib for attrib in dets[0].keys()
                           if attrib not in ["id", "score", "bbox", "image_id",
                                             "segmentation", "iscrowd", "category_id"]]:
                detections[category_id][attrib].append(np.array([det[attrib] for det in dets]))

    detections = {
        cat_id: {
            attr: np.concatenate(attr_list) for attr, attr_list in attr_dict.items()
        }
        for cat_id, attr_dict in detections.items()
    }
    if all_class:
        detections["all"] = {attr_name: np.concatenate([detections[cat_id][attr_name] for cat_id in detections.keys()])
                            for attr_name in detections[category_id].keys()}
    return detections


def label_coco(
    pred: COCO, gt: COCO, iou_threshold=0.5) -> COCO:
    coco_eval = coco_iou_eval(pred, gt, iou_threshold)

    new_dets = []
    # Order detections and their labels by src categories
    for image_eval in [image for image in coco_eval.evalImgs if image is not None]:

        valid_det = np.any(np.logical_not(image_eval["dtIgnore"]), axis=0)
        # need to load detections for actual area
        dets = coco_eval.cocoDt.loadAnns(
            np.array(image_eval["dtIds"])[valid_det]
        )
        if len(dets) > 0:
            tp_ness = np.mean(image_eval["dtMatches"][:, valid_det] > 0, axis=0, dtype=float)
            for ind, det in enumerate(dets):
                det["true_positive"] = tp_ness[ind]
        new_dets.extend(dets)
    pred.dataset["annotations"] = new_dets
    with redirect_stdout(io.StringIO()):
        pred.createIndex()
    return pred


def coco_iou_eval(pred, gt, iou_threshold):
    with redirect_stdout(io.StringIO()):
        coco_eval = COCOeval(gt, pred, iouType="bbox")
        if iou_threshold is not None:
            coco_eval.params.iouThrs = [iou_threshold]
        coco_eval.params.maxDets = [500]
        coco_eval.params.areaRng = [[0 ** 2, 1e5 ** 2]]  # all
        coco_eval.evaluate()
    return coco_eval


def calibrate_coco_predictions(calibrations: dict, coco_predictions: COCO, condition_key: str = "area") -> COCO:
    for index, prediction in enumerate(coco_predictions.dataset["annotations"]):
        try:
            calibrated_score = calibrations[prediction["category_id"]][
                prediction[condition_key]
            ]["calibrate"](prediction["score"])
        except TypeError:
            ind = calibrations[prediction["category_id"]].get_bin_index(
                prediction[condition_key]
            )
            logging.error(
                f"No Calibration found for {condition_key}: {prediction[condition_key]}, "
                f"mapped to index: {ind}, "
                f"bins: {calibrations[prediction['category_id']].bins}"
            )
            calibrated_score = prediction["score"]
        coco_predictions.dataset["annotations"][index]["score"] = calibrated_score

    with redirect_stdout(io.StringIO()):
        coco_predictions.createIndex()
    return coco_predictions


def evaluate_calibration(
    calibration_obj: dict,
    split: str,
    gt_det: Dict[str, COCO],
    preds: Dict[str, COCO],
    condition_key: str = "area",
    save_name: Optional[str] = None,
) -> Dict[str, Union[np.ndarray, float]]:
    logging.info(f"Applying calibration to {split}")
    calibrated_preds = calibrate_coco_predictions(
        calibration_obj, deepcopy(preds[split]), condition_key=condition_key
    )
    logging.info(f"Evaluating calibrated {split} set")
    eval_cal = evaluate_coco(gt_det[split], calibrated_preds, calibration_metrics=True)

    if save_name is not None:
        logging.info(f"Saving results to {save_name}")
        save_path = pathlib.Path(save_name)
        save_dict(calibrated_preds.dataset["annotations"], save_path)
        save_path = save_path.with_name("evaluations_" + save_path.name)
        save_dict(eval_cal, save_path)
    del calibrated_preds
    return eval_cal


def coco_dets_to_np(
    detections: List[List[dict]], image: dict
) -> Tuple[List[np.ndarray], list, list]:
    boxes, scores, ids = list(), list(), list()
    for detection in detections:
        if not detection:
            continue
        detection = {k: np.array([dic[k] for dic in detection]) for k in detection[0]}
        bbox = detection["bbox"]
        bbox = np.stack(
            [bbox[:, 0], bbox[:, 1], bbox[:, 0] + bbox[:, 2], bbox[:, 1] + bbox[:, 3]],
            axis=1,
        )
        boxes.append(
            bbox
        )
        scores.append(detection["score"]), ids.append(detection["category_id"])
    return boxes, scores, ids


def np_to_coco_dets(detections: tuple, image: dict) -> List[dict]:
    dets = dict()
    if len(detections) == 4:
        bbox, dets["score"], dets["category_id"], metrics = detections
        dets.update(metrics)
    else:
        bbox, dets["score"], dets["category_id"] = detections
    bbox = np.stack(
        [bbox[:, 0], bbox[:, 1], bbox[:, 2] - bbox[:, 0], bbox[:, 3] - bbox[:, 1]],
        axis=1,
    )
    dets["bbox"] = bbox
    dets["image_id"] = np.full_like(dets["score"], image["id"], dtype=int)
    return [dict(zip(dets, t)) for t in zip(*dets.values())]


def load_anns_safe(filename: Union[pathlib.Path, str]) -> list:
    filename = (
        pathlib.Path(filename) if not isinstance(filename, pathlib.Path) else filename
    )
    assert (
        filename.is_file()
    ), f"Failed to load anns from {filename}, because the file does not exist!"
    with filename.open("r") as f:
        anns = json.load(f)
    if isinstance(anns, dict):
        anns = anns["annotations"]
    assert isinstance(
        anns, list
    ), f"Failed to load anns from {filename}, the format unknown!"
    return anns


def preprocess_coco(coco: COCO):
    anns_filtered: List = []
    for img_id in coco.getImgIds():
        anns = coco.loadAnns(coco.getAnnIds(img_id))
        anns_filtered.extend(sorted(anns, key=lambda ann: ann["score"], reverse=True)[:100])
    return preprocess_annotations(anns_filtered)


def preprocess_annotations(preds):
    return list(map(clip_float_attrs, preds))


def clip_float_attrs(detection):
    detection["score"] = clip_float(detection["score"])
    detection["bbox"] = [clip_float(position) for position in detection["bbox"]]
    detection.pop("segmentation")
    return detection


def clip_float(x):
    return float("{:.2f}".format(x))


def load_dets_gt_coco(cfg, split=None):
    if split is None:
        split = cfg["split"]
    path_to_annos = pathlib.Path(
        to_absolute_path(cfg["paths"]["annotations"]["path"]),
        cfg["paths"]["annotations"]["split"][split],
    )
    path_to_dets = pathlib.Path(to_absolute_path(cfg["paths"]["detections"]["path"]))
    # load target coco
    logging.info(f"Attempting to load {split} Annotations from {str(path_to_annos)}.")
    with redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(path_to_annos))
    # load coco detection

    detectors = get_detectors(cfg)
    det_paths = [
        path_to_dets.joinpath(cfg["detections"][detector][split])
        for detector in detectors
    ]
    logging.info(f"Attempting to load {split} Detections from {str(det_paths[0])}.")
    det_paths = map(load_anns_safe, det_paths)
    with redirect_stdout(io.StringIO()):
        coco_dets = list(map(coco_gt.loadRes, det_paths))

    return coco_dets, coco_gt


def get_detectors(cfg):
    if "scales" in cfg:
        detectors = [cfg['detector'] + "_" + scale for scale in cfg["scales"]]
    elif "detectors" in cfg:
        detectors = cfg["detectors"]
    else:
        detectors = [cfg["detector"]]
    return detectors

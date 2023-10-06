import copy
import io
import itertools
import logging
from collections import defaultdict
from contextlib import redirect_stdout
import pyminizip as pyzip

from src.plots import plot_cal_func
import hydra
import numpy as np
import tqdm
from netcal.scaling import BetaCalibrationDependent, BetaCalibration, LogisticCalibration, LogisticCalibrationDependent
from sklearn.model_selection import train_test_split

from src.utils.nms_helpers import apply_nms
from src.utils.coco_helpers import evaluate_coco, \
    dataset_from_image_ids, label_coco, preprocess_coco, load_dets_gt_coco, get_detectors
from src.utils.general import ensure_list, save_dict_as_table, save_dict


@hydra.main(config_path="config", config_name="iou_aware_calibration")
def iou_aware_confidence_calibration(cfg):
    preds, gt = load_dets_gt_coco(cfg)

    logging.info("Performing IOU computation")
    preds = apply_nms(cfg, preds, gt)

    logging.info("Evaluating detections")
    labeled_preds = label_coco(preds, gt, iou_threshold=cfg.get("gt_iou_thresh", 0.5))

    if cfg["splits"] > 0:
        logging.info("Validating calibration on cross-validation splits")
        results = []
        calib_funcs = defaultdict(dict)
        metric_dict = get_cond_metrics(cfg)
        for seed in tqdm.trange(cfg["splits"]):
            split1_img_ids, split2_img_ids = train_test_split(
                list(gt.imgs.keys()),
                train_size=cfg["split_size"],
                random_state=seed,
            )
            gt_test = dataset_from_image_ids(split2_img_ids, gt)
            dt_train = dataset_from_image_ids(split1_img_ids, labeled_preds)
            dt_test = dataset_from_image_ids(split2_img_ids, labeled_preds)

            # Baseline metrics
            res_pre = evaluate_coco(gt_test, dt_test, calibration_metrics=True,
                                    calib_truth_threshold=0.5)
            res_pre["split"] = seed
            res_pre["cond_keys"] = "-"
            results.append(res_pre)

            # Calibration metrics
            for met_name, condition_met in metric_dict.items():
                dt_test_cal, calib_funcs[met_name][seed] = conditional_iou_calibration(cfg,
                                                                                       copy.deepcopy(dt_test),
                                                                                       dt_train,
                                                                                       condition_met)
                res = evaluate_coco(gt_test, dt_test_cal, calibration_metrics=True,
                                    calib_truth_threshold=0.5)
                res["split"] = seed
                res["cond_keys"] = met_name
                logging.info(res)
                results.append(res)

        del gt_test, dt_test, dt_train, dt_test_cal

        if cfg.get("plot", False):
            plot_cal_func(calib_funcs, metric_dict, cfg)

        logging.info("Computing summary statistics of metrics")
        metric_dict["-"] = "-"
        cum_results = []
        for cond_key in metric_dict.keys():
            filter_results = [res for res in results if cond_key in res["cond_keys"]]
            max_res = {key: np.max([run[key] for run in filter_results]) for key in res.keys()
                            if key not in {"split", "cond_keys"}}
            min_res = {key: np.min([run[key] for run in filter_results]) for key in res.keys()
                            if key not in {"split", "cond_keys"}}
            mean_res = {key: np.mean([run[key] for run in filter_results]) for key in res.keys()
                            if key not in {"split", "cond_keys"}}
            cum_res = {key: f"valmaxmin{{{mean_res[key]:0.2f}}}" \
                            f"{{{max_res[key]-mean_res[key]:0.2f}}}" \
                            f"{{{mean_res[key]-min_res[key]:0.2f}}}" for key in res.keys()
                       if key not in {"split", "cond_keys"}}
            cum_res["cond_key"] = cond_key
            cum_results.append(cum_res)

        logging.info("Saving cross-validation results")
        res_table = save_dict_as_table(results,
                                       ignore_cols=["mAP75", "d_mAP75"],
                                       name=f"nms_calibration_{cfg['name']}")
        cum_res_table = save_dict_as_table(cum_results,
                                           ignore_cols=["mAP75", "d_mAP75"],
                                           name=f"cum_nms_calibration_{cfg['name']}")
        short_res_table = cum_res_table.drop(columns=[name for name in cum_res_table.columns if
                                                      any([kw in name for kw in ["AR", "small", "large",
                                                                                 "medium", "75", "per_class"]])])
        logging.info(short_res_table.to_latex(float_format='{:0.2f}'.format))

    if cfg.get("apply_on", False):
        logging.info(f"Applying nms-aware calibration to {cfg['apply_on']}")
        dt_test, gt = load_dets_gt_coco(cfg, split=cfg["apply_on"])
        dt_test_l = apply_nms(cfg, dt_test, gt)
        if cfg.nms.run_type != "nms_count":
            save_dict(preprocess_coco(copy.deepcopy(dt_test_l)), f"{cfg.nms.run_type}_{get_detectors(cfg)[0]}_{cfg['apply_on']}.json")
        for met_name, condition_met in get_cond_metrics(cfg).items():
            dt_test_cal, _ = conditional_iou_calibration(cfg, copy.deepcopy(dt_test_l), labeled_preds, condition_met)
            logging.info(f"Saving results")
            filename = f"ccal_{met_name}_{get_detectors(cfg)[0]}_{cfg['apply_on']}"
            save_dict(preprocess_coco(dt_test_cal), f"{filename}.json")
            pyzip.compress(f"{filename}.json", None, f"{filename}.zip", None, 4)
    return res_table


all_metrics = [
    "min_remaining_area",
    "min_suppressed_rem_area",
    "prod_remaining_area",
    "prod_suppressed_rem_area",
]


def get_cond_metrics(cfg):
    if cfg["variates"] == "none":
        metric_dict = {"none": None}

    elif cfg["variates"] == "first":
        metric_dict = {"min_remaining_area": ["min_remaining_area"]}

    elif cfg["variates"] == "single":
        metric_dict = {name: [name] for name in all_metrics}

    elif cfg["variates"] == "duo":
        metric_dict = {"min_prod_remaining_area": ["min_remaining_area", "prod_remaining_area"]}

    elif cfg["variates"] == "additions":
        metric_dict_additions = all_metric_combinations(all_metrics[1:])
        metric_dict = {f"{all_metrics[0]}__{key}": [all_metrics[0]] + val for key, val in metric_dict_additions.items()}
        metric_dict[all_metrics[0]] = [all_metrics[0]]

    elif cfg["variates"] == "all":
        metric_dict = all_metric_combinations(all_metrics)

    else:
        raise ValueError(f"Unknown variate setting: {cfg['variates']}")
    return metric_dict


def all_metric_combinations(metrics):
    metric_dict = {}
    for len_combinations in range(1, len(metrics) + 1):
        for condition_met in itertools.combinations(metrics, len_combinations):
            metric_dict['__'.join(condition_met)] = list(condition_met)
    return metric_dict


def conditional_iou_calibration(cfg, dt_test, dt_train, condition_keys=None):
    get_params = params_getter_fn(ensure_list(condition_keys) if condition_keys is not None else None)
    train_x = np.array([get_params(det) for det in dt_train.anns.values()])
    train_y = np.array([det["true_positive"] for det in dt_train.anns.values()])
    calibration_func = get_cal_object(cfg)
    with redirect_stdout(io.StringIO()):
        calibration_func.fit(train_x, train_y)
        for det in dt_test.anns.values():
            det["score"] = calibration_func.transform(np.array(get_params(det))[None, :])[0]
        dt_test.createIndex()
    return dt_test, calibration_func


def get_cal_object(cfg):
    calib_type = cfg.get("calib_type", "Beta_cond")
    use_cuda = 'cuda:0' if cfg.get("use_cuda", True) else False
    if calib_type == "Beta_cond":
        return BetaCalibrationDependent(use_cuda=use_cuda, method='mle')
    elif calib_type == "Beta":
        return BetaCalibration(use_cuda=use_cuda, detection=True, method='mle')
    elif calib_type == "Logistic_cond":
        return LogisticCalibrationDependent(use_cuda=use_cuda, method='mle')
    elif calib_type == "Logistic":
        return LogisticCalibration(use_cuda=use_cuda, detection=True, method='mle')
    else:
        raise ValueError(f"Unknown calibration type: {calib_type}")


def params_getter_fn(condition_keys):
    if condition_keys is None:
        return lambda x: [x["score"]]
    else:
        logging.info(f"condition_keys:{[key for key in condition_keys]}")

        def get_params_keys(det):
            return [det["score"]] + [det[key] for key in condition_keys]
        return get_params_keys


if __name__ == "__main__":
    iou_aware_confidence_calibration()

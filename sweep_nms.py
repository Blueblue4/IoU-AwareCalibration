import copy
import logging

import hydra
import numpy as np
from omegaconf import DictConfig
from omegaconf import open_dict

from src.utils.nms_helpers import apply_nms
from src.utils.coco_helpers import (
    evaluate_coco, load_dets_gt_coco, )
from src.utils.general import save_dict_as_table

log = logging.getLogger(__name__)
sweep_spaces = {
    "nms": {
        "iou_thr": {
            "keys": ["nms", "settings", "iou_thr"],
            "values": np.linspace(0.4, 0.9, num=11, endpoint=True)
        },
    },
    "soft_nms": {
        "sigma": {
            "keys": ["nms", "settings", "sigma"],
            "values": np.linspace(0.1, 0.7, num=10, endpoint=True)
        },
    },
    "linear_soft_nms": {
        "lin_iou_thr": {
            "keys": ["nms", "settings", "iou_thr"],
            "values": np.linspace(0.1, 0.9, num=11, endpoint=True)
        },
    },
    "wbf": {
        "iou_thr": {
            "keys": ["nms", "settings", "iou_thr"],
            "values": np.linspace(0.5, 0.9, num=11, endpoint=True)
        },
    },
}


@hydra.main(config_path="config", config_name="sweep_nms")
def seep_nms(cfg: DictConfig) -> dict:
    with open_dict(cfg):
        coco_dets, coco_gt = load_dets_gt_coco(cfg)
        result = evaluate_coco(coco_gt, coco_dets[0])
        results = []
        best_params = {}

        log.info(f"Pre-NMS mAP:{result['mAP']}")
        for nms_type, nms_space in sweep_spaces.items():

            log.info(f"Running {nms_type}")
            local_cfg = copy.deepcopy(cfg)
            recursive_set(local_cfg, ["nms", "run_type"], nms_type)
            result = recursive_sweep(local_cfg, copy.deepcopy(nms_space), basic_nms, coco_dets, coco_gt)

            maps = [res['mAP'] for res in result]
            best_index = np.argmax(maps)
            log.info(f"Worst mAP:{min(maps)}")
            log.info(f"Best {nms_type} mAP:{maps[best_index]} with:")

            for parm in nms_space.keys():
                best_params[nms_type] = {parm: [res[parm] for res in result][best_index]}
                log.info(f"{parm}:{[res[parm] for res in result][best_index]}")

            for res in result:
                res["nms_type"] = nms_type

            results.extend(result)

        ds_table = save_dict_as_table(results, ignore_cols=None, name="nms_sweep")

    log.info("done!")
    return best_params


def basic_nms(cfg, coco_dets, coco_gt):
    preds_local = copy.deepcopy(coco_dets)
    preds_local_nms = apply_nms(cfg, preds_local, coco_gt)
    result = evaluate_coco(coco_gt, preds_local_nms)
    return result


def recursive_sweep(cfg, sweep_space, sweep_func, *args):
    results = list()

    sweep_values_name = list(sweep_space.keys())[0]
    value_settings = sweep_space.pop(sweep_values_name)
    for sweep_value in value_settings["values"]:
        recursive_set(cfg, value_settings["keys"], float(sweep_value))
        if len(sweep_space) == 0:
            result = sweep_func(cfg, *args)
            result[sweep_values_name] = sweep_value
            results.append(copy.deepcopy(result))
        else:
            result = recursive_sweep(cfg, copy.deepcopy(sweep_space), sweep_func, *args)
            for res in result:
                res[sweep_values_name] = sweep_value
            results.extend(copy.deepcopy(result))
    return results


def recursive_get(dictionary: dict, keys: list):
    if len(keys) == 1:
        return dictionary[keys[0]]
    return recursive_get(dictionary[keys[0]], keys[1:])


def recursive_set(dictionary: dict, keys: list, value):
    if len(keys) == 1:
        dictionary[keys[0]] = value
    else:
        recursive_set(dictionary[keys[0]], keys[1:], value)


if __name__ == "__main__":
    seep_nms()

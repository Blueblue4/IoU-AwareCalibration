import copy
import logging

import dufte
import hydra
import numpy as np
import tikzplotlib
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from src.utils.nms_helpers import apply_nms
from iou_aware_calibration import get_cond_metrics, conditional_iou_calibration
from src.reliability import reliability_diagram
from src.utils.coco_helpers import dataset_from_image_ids, label_coco, load_dets_gt_coco, get_detectors

plt.style.use(dufte.style)


@hydra.main(config_path="config", config_name="iou_aware_calibration")
def plot_iou_aware_confidence_calibration(cfg):
    logging.info(f"Loading Detections and Annotations for {get_detectors(cfg)[0]}")
    preds, gt = load_dets_gt_coco(cfg)

    logging.info("Performing NMS")
    preds = apply_nms(cfg, preds, gt)

    logging.info("Evaluating detections")
    labeled_preds = label_coco(preds, gt, iou_threshold=cfg.get("gt_iou_thresh", 0.5))

    split1_img_ids, split2_img_ids = train_test_split(
        list(gt.imgs.keys()),
        train_size=cfg["split_size"],
        random_state=1,
    )
    gt_test = dataset_from_image_ids(split2_img_ids, gt)
    dt_train = dataset_from_image_ids(split1_img_ids, labeled_preds)
    dt_test = dataset_from_image_ids(split2_img_ids, labeled_preds)

    metric_dict = get_cond_metrics(cfg)
    metric_dict["-"] = "-"
    # Calibration metrics
    for met_name, condition_met in metric_dict.items():
        if met_name !=  "-":
            dt_test_cal, _ = conditional_iou_calibration(cfg,
                                                         copy.deepcopy(dt_test),
                                                         dt_train,
                                                         condition_met)
        else:
            dt_test_cal = copy.deepcopy(dt_test)
        anns_filtered = []
        for img_id in dt_test_cal.getImgIds():
            anns = dt_test_cal.loadAnns(dt_test_cal.getAnnIds(img_id))
            anns_filtered.extend(sorted(anns, key=lambda ann: ann["score"], reverse=True)[:100])
        dt_test_filtered = [ann for ann in anns_filtered if ann['score'] > 0.01]
        target_labels = [int(ann['true_positive']) for ann in dt_test_filtered]
        confs = [ann['score'] for ann in dt_test_filtered]
        fig = reliability_diagram(np.array(target_labels), np.ones_like(target_labels), np.array(confs), draw_ece=False,
                                  return_fig=True, draw_averages=False, title="Reliability Diagram")

        plt.savefig(cfg.name + met_name + ".svg")
        tikzplotlib.save(cfg.name + met_name + ".tex")
        plt.show()


if __name__ == "__main__":
    plot_iou_aware_confidence_calibration()


def soft_nms_dummy(conf, iou_value, sigma=0.2):
    return np.exp(-(iou_value ** 2) / sigma) * conf


def plot_soft_nms():
    iou_values = np.linspace(0.01, 0.99, 200)
    conf_in = [0.3, 0.6, 0.9]
    for conf_value in conf_in:

        all_confs = [soft_nms_dummy(conf_value, iou_value, sigma=0.5) for iou_value in iou_values]
        plt.plot(
            iou_values,
            all_confs,
            label=f"Conf_in: {conf_value}",
            alpha=0.9,
        )
    plt.legend()
    plt.title(
        f"soft-nms"
    )
    plt.ylabel("Adjusted Confidence")
    plt.xlabel("IoU")
    plt.axis([0, 1, 0, 1.0])
    filename = f"soft-nms_over_select_confs"
    plt.tight_layout()
    plt.savefig(filename + ".svg")
    tikzplotlib.save(filename + ".tex")
    plt.show()

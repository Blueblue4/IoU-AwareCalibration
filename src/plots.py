import logging

import numpy as np

import tikzplotlib
from matplotlib import pyplot as plt


def plot_cal_func(cal_func, metric_dict, cfg):
    for metric_name, func_dict in cal_func.items():
        if len(metric_dict[metric_name]) != 1:
            pass
        # conf sweep
        confs_in = np.linspace(0.01, 0.99, 200) # to avoid border effects at 0 from log
        iou_values = [0.3, 0.6, 0.9]
        for iou_value in iou_values:
            all_confs = []
            for func in func_dict.values():
                all_confs.append([func.transform(np.array([conf, 1.-iou_value])[None, :])[0] for conf in confs_in])
            all_confs = np.array(all_confs)
            plt.plot(
                confs_in,
                np.mean(all_confs, axis=0),
                label=f"IoU: {iou_value}",
                alpha=0.9,
            )
            plt.fill_between(
                confs_in,
                np.min(all_confs, axis=0),
                np.max(all_confs, axis=0),
                alpha=0.1,
            )
        plt.legend()
        plt.title(
            f"{cfg['name']}: Calib curve over confidence"
        )
        plt.ylabel("Adjusted Confidence")
        plt.xlabel("Confidence")
        plt.axis([0, 1, 0, 1.0])
        filename = f"Conf-conf_over_select_ious"
        plt.tight_layout()
        plt.savefig(filename + ".svg")
        tikzplotlib.save(filename + ".tex")
        plt.show()

        # IoU sweep
        iou_values = np.linspace(0.01, 0.99, 200)
        conf_in = [0.3, 0.6, 0.9]
        for conf_value in conf_in:
            all_confs = []
            for func in func_dict.values():
                all_confs.append([func.transform(np.array([conf_value, 1.-iou_value])[None, :])[0] for iou_value in iou_values])
            all_confs = np.array(all_confs)
            plt.plot(
                iou_values,
                np.mean(all_confs, axis=0),
                label=f"Conf_in: {conf_value}",
                alpha=0.9,
            )
            plt.fill_between(
                iou_values,
                np.min(all_confs, axis=0),
                np.max(all_confs, axis=0),
                alpha=0.1,
            )
        plt.legend()
        plt.title(
            f"{cfg['name']}: Calib curve over iou"
        )
        plt.ylabel("Adjusted Confidence")
        plt.xlabel("IoU")
        plt.axis([0, 1, 0, 1.0])
        filename = f"Conf-iou_over_select_confs"
        plt.tight_layout()
        plt.savefig(filename + ".svg")
        tikzplotlib.save(filename + ".tex")
        plt.show()
        logging.info("done plotting")

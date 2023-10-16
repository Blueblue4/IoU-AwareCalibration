# IoU-Aware Confidence Calibration - Code

> [!NOTE]
>This repository is for research purposes only. It is designed to reproduce the experiments described in **"Do We Still Need Non-Maximum Suppression? Accurate Confidence Estimates and Implicit Duplication Modeling with IoU-Aware Calibration."**

[![arxiv](https://img.shields.io/badge/arXiv-cs.CV:2309.03110-B31B1B.svg)](https://arxiv.org/abs/2309.03110)

We show that we can eliminate classic NMS-style post-processing by using IoU-aware calibration. IoU-aware calibration is a conditional Beta calibration; this makes it parallelizable with no hyper-parameters. Instead of arbitrary cutoffs or discounts, it implicitly accounts for the likelihood of each detection being a duplicate and adjusts the confidence score accordingly, resulting in empirically based precision estimates for each detection. Our extensive experiments on diverse detection architectures show that the proposed IoU-aware calibration can successfully model duplicate detections and improve calibration. Compared to the standard sequential NMS and calibration approach, our joint modeling can deliver performance gains over the best NMS-based alternative while producing consistently better-calibrated confidence predictions with less complexity.
## Setup
### Download Annotations
Download COCO2017 annotations for train, val, and tes-dev from [here](https://cocodataset.org/#download)
and move them into the folder structure like this (alternatively change the config in `config/all/paths/annotations/coco_2017.yaml` to your local folder structure): 
```
 .
 └── data
   └── coco
      └── annotations
        ├── instances_train2017.json
        ├── instances_val2017.json
        └── image_info_test-dev2017.json
```

### Generate Detections

Generate detections on the (train), val, and test-dev COCO2017 and save them in the COCO file format as JSON files.
Be aware that the detections need to be generated without nms postprocessing and without the usual limit of 100 detections per image, to be as close as possible to the original model outputs.
Move detections to `detections/` (change setting in `config/all/paths/detections.yaml`). The detections should be saved as `${SPLIT}_results.json` with SPLIT in `[val, test]` (change setting in `config/all/detections/default_all.yaml`).
```
 .
 └── detections
   ├── train_results.json
   ├── val_results.json
   └── test_results.json
```
### Set up the environment

Set up a virtual environment:
```bash
virtualenv -p python3 venv
source venv/bin/activate
```
Install dependencies:
```bash
pip3 install -r requirements.txt
```

## Usage 
### Cross validation of iou aware calibration
Perform cross validation on the validation detections:
```bash
Python iou_aware_calibration.py 
```
Fit iou aware calibration on validation split and apply them on test split:
```bash
Python iou_aware_calibration.py splits=0 +apply_on="test"
```

## Settings
The settings for the calibration are defined in via a [Hydra](https://hydra.cc/) config in `config/iou_aware_calibration.yaml` and can also be set via the CLI.
```yaml
name: "iou_aware_calibration"
# Name of the experiment and name of output directory `output/$name/date-of-experiment.
save_outputs: true
#  If true, the outputs of the calibration are saved. 
variates: "first" 
# Defines the conditional variables used for the calibration.
#    none: No variates are used i.e. normal calibration
#    first: Conditioning on minimum Jaccard index 
#    single: Conditioning on each of minimum Jaccard indexes, product of Jaccard indexes,
#      minimum suppressing jaccard indexes and product of suppressing Jaccard indexes individually.
#    duo: Conditioning on both of minimum jaccard indexes, product of Jaccard indexes jointly.
#    additions: Conditioning on all combinations of two of the mentioned variates. 
#    all: Conditioning on all combinations of the mentioned variates. 
calib_type: "Beta_cond"
# Defines the calibration type used.
#    Beta_cond: Beta calibration with conditional variables
#    Beta: Beta calibration without conditional variables
#    Logistic_cond: Logistic calibration with conditional variables
#    Logistic: Logistic calibration without conditional variables
detector: "detector"
# Defines the setting in /config/all/detections/default_all.yaml is used for detectors detection files. 
seed: 123
# Random seed used for the experiments.
iou_threshold: 0.5
# Defines the iou threshold for evaluating if a detection is a TP for the purpose of the calibration.
split_size: 0.6
# Defines the size of the train split used for the cross validation calibration.
splits: 10
# Number of splits used for the cross validation calibration.
```
Optionally add +apply_on="test" to apply conditional calibration on the test split and save the result for evaluation ot the test server:
```bash
Python iou_aware_confidence_calibration.py +apply_on="test"
```

## Citation
If you find this repository useful for your research, please cite our paper:
```
@article{gilg2023we,
  title={Do We Still Need Non-Maximum Suppression? Accurate Confidence Estimates and Implicit Duplication Modeling with IoU-Aware Calibration},
  author={Gilg, Johannes and Teepe, Torben and Herzog, Fabian and Wolters, Philipp and Rigoll, Gerhard},
  journal={arXiv preprint arXiv:2309.03110},
  year={2023}
}
```

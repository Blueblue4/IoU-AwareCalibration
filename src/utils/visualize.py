from itertools import islice
from pathlib import Path
import fiftyone as fo


def visualize_coco_dets(coco_dets, nr_imgs=100, seed=1, path_to_imgs=None):
    """
    Visualize the detections in the COCO dataset
    """

    for ind, image in dict(islice(coco_dets.cocoGt.imgs, nr_imgs)).items():

        dets_gt = coco_dets.cocoGt.getAnnIds(img_ids=[ind])[0]
        dets_dt = coco_dets.cocoDt.getAnnIds(img_ids=[ind])[0]
        fo.Sample(
            filepath=str(Path.joinpath(path_to_imgs, image["file_name"])),
            gt_objects=fo.Detections(
                detections=[coco_gt_to_fo(det_gt, coco_dets.cocoGt.cats, image) for det_gt in dets_gt]
            ),
            pred_objects=fo.Detections(
                detections=[coco_det_to_fo(det_dt, coco_dets.cocoGt.cats, image) for det_dt in dets_dt],
                ),
        )

    # Get the annotations


def coco_det_to_fo(coco_det, label_dict, image):
    """
    Convert the COCO detection to the FiftyOne format
    """

    return fo.Detection(label=label_dict[coco_det.pop("category_id")]["name"],
                        bounding_box=bbox_to_rel(coco_det.pop("bbox"), image),
                        confidence=coco_det.pop("score"),
                        index=coco_det.pop("id"),
                        attibutes=coco_det)


def coco_gt_to_fo(coco_det, label_dict, image):
    """
    Convert the COCO annotations to the FiftyOne format
    """

    return fo.Detection(label=label_dict[coco_det.pop("category_id")]["name"],
                        bounding_box=bbox_to_rel(coco_det.pop("bbox"), image),
                        index=coco_det.pop("id"),
                        attibutes=coco_det)


def bbox_to_rel(bbox, image):
    """
    Convert the COCO bounding box (absolute) to the FiftyOne format (relative)
    """

    return [bbox[0]/image["width"],
            bbox[1]/image["height"],
            bbox[2]/image["width"],
            bbox[3]/image["height"]]

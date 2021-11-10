from mrcnn.utils import compute_ap, Dataset
from config import PredictionConfig
from mrcnn.model import MaskRCNN, load_image_gt, mold_image
from mrcnn.config import Config
from numpy import ndarray
import numpy as np
from typing import List, Dict
from dataset import KangarooDataset, AnalogMeterDataset
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def evaluate_model(dataset: Dataset, model: MaskRCNN, cfg: Config) -> ndarray:
    # cfg = PredictionConfig()
    # # define model
    # model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
    # # load weights
    # model.load_weights('mask_rcnn_kangaroo_cfg_0005.h5', by_name=True)
    APs = list()
    for image_id in dataset.image_ids:
        # load image, bboxes, and mask for image id
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset=dataset, config=cfg, image_id=image_id,
                                                                         use_mini_mask=False)
        # convert pixel values (e.g. center)
        scaled_image: ndarray = mold_image(image, cfg)
        sample: ndarray = np.expand_dims(scaled_image, 0)
        # make prediction
        yhat: List = model.detect(sample, verbose=0)
        # extract results for first sample
        r: Dict = yhat[0]
        # calculate statistics, including AP
        AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r['rois'], r["class_ids"], r["scores"], r["masks"])
        # store
        APs.append(AP)
    # calculate the mean AP across all images
    mAP: ndarray = np.mean(APs)

    return mAP


def evaluate():
    cfg = PredictionConfig()
    # define model
    model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
    # load weights
    model.load_weights('./amr_cfg/mask_rcnn_amr_cfg_0005.h5', by_name=True)
    train_set = AnalogMeterDataset()
    train_set.load_dataset('amr', is_train=True)
    train_set.prepare()
    # prepare test/val set
    test_set = AnalogMeterDataset()
    test_set.load_dataset('amr', is_train=False)
    test_set.prepare()
    # evaluate model on training dataset
    train_mAP: ndarray = evaluate_model(train_set, model, cfg)
    print("Train mAP: %.3f" % train_mAP)
    # evaluate model on test dataset
    test_mAP: ndarray = evaluate_model(test_set, model, cfg)
    print("Test mAP: %.3f" % test_mAP)


if __name__ == "__main__":
    evaluate()

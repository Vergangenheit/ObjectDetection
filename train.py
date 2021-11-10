from config import KangarooConfig, AnalogConfig
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
from dataset import KangarooDataset, AnalogMeterDataset
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def train_model(train_set: AnalogMeterDataset, test_set: AnalogMeterDataset):
    config = AnalogConfig()
    model = MaskRCNN(mode='training', model_dir='./', config=config)
    # load weights (mscoco) and exclude the output layers
    model.load_weights('mask_rcnn_coco.h5', by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    # train weights (output layers or 'heads')
    model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')


if __name__ == "__main__":
    # prepare train set
    train_set = AnalogMeterDataset()
    train_set.load_dataset('amr', is_train=True)
    train_set.prepare()
    # prepare test/val set
    test_set = AnalogMeterDataset()
    test_set.load_dataset('amr', is_train=False)
    test_set.prepare()
    train_model(train_set, test_set)

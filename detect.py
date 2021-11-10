from mrcnn.utils import Dataset
from dataset import KangarooDataset, AnalogMeterDataset
from mrcnn.config import Config
from mrcnn.model import mold_image, MaskRCNN
from numpy import ndarray
import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from config import Config, PredictionConfig
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def plot_actual_vs_predicted(dataset: Dataset, cfg: Config, model: MaskRCNN, n_images: int = 5):
    for i in range(n_images):
        # load image and the mask
        image: ndarray = dataset.load_image(i)
        mask: ndarray
        mask, _ = dataset.load_mask(i)
        # convert pixel values (e.g. center)
        scaled_image: ndarray = mold_image(image, cfg)
        # convert image into one sample
        sample: ndarray = np.expand_dims(scaled_image, 0)
        # make prediction
        yhat: Dict = model.detect(sample, verbose=0)[0]
        print(yhat['rois'].shape)
        # define subplot
        plt.subplot(n_images, 2, i * 2 + 1)
        # plot raw pixel data
        plt.imshow(image)
        plt.title('Actual')
        # plot masks
        for j in range(mask.shape[2]):
            plt.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
        # get the context for drawing boxes
        plt.subplot(n_images, 2, i * 2 + 2)
        # plot raw pixel data
        plt.imshow(image)
        plt.title('Predicted')
        ax = plt.gca()
        # plot each box
        for box in yhat['rois']:
            # get coordinates
            y1, x1, y2, x2 = box
            # calculate width and height of the box
            width, height = x2 - x1, y2 - y1
            # create the shape
            rect = Rectangle((x1, y1), width, height, fill=False, color='red')
            # draw the box
            ax.add_patch(rect)
    # show the figure
    plt.show()


if __name__ == "__main__":
    # load the train dataset
    train_set = KangarooDataset()
    train_set.load_dataset('kangaroo', is_train=True)
    train_set.prepare()
    print('Train: %d' % len(train_set.image_ids))
    # load the test dataset
    test_set = KangarooDataset()
    test_set.load_dataset('kangaroo', is_train=False)
    test_set.prepare()
    print('Test: %d' % len(test_set.image_ids))
    # create config
    cfg = PredictionConfig()
    # define the model
    model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
    # load model weights
    model_path = 'kangaroo_cfg/mask_rcnn_kangaroo_cfg_0005.h5'
    model.load_weights(model_path, by_name=True)
    # plot predictions for train dataset
    plot_actual_vs_predicted(train_set, cfg, model)
    # plot predictions for test dataset
    plot_actual_vs_predicted(test_set, cfg, model)

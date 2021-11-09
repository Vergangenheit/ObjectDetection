from mrcnn.utils import Dataset, extract_bboxes
from mrcnn.visualize import display_instances
import os
from typing import List, Dict
from numpy import asarray, zeros, ndarray
from xml.etree import ElementTree as et
from xml.etree.ElementTree import Element, ElementTree
import matplotlib.pyplot as plt


class KangarooDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir: str, is_train: bool = True) -> None:
        # define one class
        self.add_class("dataset", 1, "kangaroo")
        # define data locations
        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annots/'
        # find all images
        for filename in os.listdir(images_dir):
            # extract image id
            image_id = filename[:-4]
            # skip bad images
            if image_id in ['00090']:
                continue
            # skip all images after 150 if we are building the train set
            if is_train and int(image_id) >= 150:
                continue
            # skip all images before 150 if we are building the test/val set
            if not is_train and int(image_id) < 150:
                continue
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'
            # add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

        # extract bounding boxes from an annotation file

    def extract_boxes(self, filename: str) -> (List, int, int):
        # load and parse the file
        tree: ElementTree = et.parse(filename)
        # get the root of the document
        root: Element = tree.getroot()
        # extract each bounding box
        boxes = []
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

    # load the masks for an image
    def load_mask(self, image_id: str) -> (ndarray, ndarray):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info['annotation']
        # load XML
        boxes: List
        w: int
        h: int
        boxes, w, h = self.extract_boxes(path)
        # create one array for all masks, each on a different channel
        masks: ndarray = zeros([h, w, len(boxes)], dtype='uint8')
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('kangaroo'))
        return masks, asarray(class_ids, dtype='int32')

    # load an image reference
    def image_reference(self, image_id: str) -> str:
        info = self.image_info[image_id]
        return info['path']


class AnalogMeter(Dataset):
    def load_dataset(self, dataset_dir: str, is_train: bool = True):
        # define one class
        self.add_class("dataset", 1, "amr")
        # define data locations
        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annots/'
        # find all images
        counter = 0
        for filename in os.listdir(images_dir):
            # extract image id
            image_id: str = filename.split('.')[0].split('(')[-1].replace(')', '')
            # skip all images after 150 if we are building the train set
            if is_train and counter >= 225:
                continue
            # skip all images before 150 if we are building the test/val set
            if not is_train and counter < 225:
                continue
            img_path = images_dir + filename
            ann_path = annotations_dir + filename.replace('.jpg', '.xml')
            # add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
            counter += 1

    def extract_boxes(self, filename: str) -> (List, int, int):
        # load and parse the file
        tree: ElementTree = et.parse(filename)
        # get the root of the document
        root: Element = tree.getroot()
        # extract each bounding box
        boxes = []
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)

        return boxes, width, height

    # load the masks for an image
    def load_mask(self, image_id: str) -> (ndarray, ndarray):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info['annotation']
        # load XML
        boxes: List
        w: int
        h: int
        boxes, w, h = self.extract_boxes(path)
        # create one array for all masks, each on a different channel
        masks: ndarray = zeros([h, w, len(boxes)], dtype='uint8')
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('amr'))

        return masks, asarray(class_ids, dtype='int32')

    #load an image reference
    def image_reference(self, image_id: str) -> str:
        info = self.image_info[image_id]
        return info['path']




def test_dataset():
    # train set
    train_set = AnalogMeter()
    train_set.load_dataset('amr', is_train=True)
    train_set.prepare()
    # load an image
    image_id = 7
    image: ndarray = train_set.load_image(image_id)
    print(image.shape)
    # load image mask
    mask: ndarray
    class_ids: ndarray
    mask, class_ids = train_set.load_mask(image_id)
    print(mask.shape)
    # plot image
    plt.imshow(image)
    # plot mask
    plt.imshow(mask[:, :, 0], cmap='gray', alpha=0.5)
    plt.show()


def test_image_info():
    # train set
    train_set = AnalogMeter()
    train_set.load_dataset('amr', is_train=True)
    train_set.prepare()
    # enumerate all images in the dataset
    for image_id in train_set.image_ids:
        # load image info
        info: Dict = train_set.image_info[image_id]
        # display on the console
        print(info)


def test_instances():
    # train set
    train_set = AnalogMeter()
    train_set.load_dataset('amr', is_train=True)
    train_set.prepare()
    image_id = 101
    # load the image
    image: ndarray = train_set.load_image(image_id)
    # load the masks and the class ids
    mask, class_ids = train_set.load_mask(image_id)
    # extract bounding boxes from the masks
    bbox: ndarray = extract_bboxes(mask)
    # display image with masks and bounding boxes
    display_instances(image, bbox, mask, class_ids, train_set.class_names)


if __name__ == "__main__":
    test_instances()

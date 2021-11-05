from xml.etree import ElementTree as et
from xml.etree.ElementTree import Element, ElementTree
from typing import List


def extract_boxes(filename: str) -> (List, int, int):
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


if __name__ == "__main__":
    # extract details form annotation file
    boxes, w, h = extract_boxes('kangaroo/annots/00001.xml')
    # summarize extracted details
    print(boxes, w, h)

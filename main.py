import cv2
import numpy as np
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
from numpy.lib.polynomial import poly
import os.path


def detect_objects(img_path):
    # load the image
    if not os.path.exists(img_path):
        print("cannot find the file specified.")
        exit(1)
    BGR_image = cv2.imread(img_path)

    # cv reads image as BRG by default, convert to RGB
    image = cv2.cvtColor(BGR_image, cv2.COLOR_BGR2RGB)

    # draw boxes around common objects
    # list of objects: https://github.com/arunponnusamy/object-detection-opencv/blob/master/yolov3.txt
    box, label, conf = cv.detect_common_objects(image)
    output = draw_bbox(image, box, label, conf)

    print(box)  # coords that define each bounding box
    print(label)  # labels for objects found
    print(conf)  # confidence that the label accurately defines its object

    # show the result
    plt.imshow(output)
    plt.show()

    # print count of things found
    # set removes duplicates
    labelset = set(label)
    for l in labelset:
        plural = "s" if label.count(l) > 1 else ""
        print(f"The image contains {label.count(l)} {l}{plural}")


if __name__ == "__main__":
    img_path = "data/samples/4.png"
    detect_objects(img_path)

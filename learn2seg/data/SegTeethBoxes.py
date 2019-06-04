# Class to handle teeth data calls
import os
import cv2
import json
import numpy as np

# TODO: Don't save it as a hard-coded location
dataset_root = 'dataset/data/teeth/pan-teeth'

module_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(os.path.dirname(module_path))
data_path = os.path.join(root_path, 'dataset/seg_teeth')
print(data_path)


class SegTeethBoxesDataset(object):

    def __init__(self):
        if os.path.exists(data_path):
            print("found dir!")
        else:
            print("hi")
        return

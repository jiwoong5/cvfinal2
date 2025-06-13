import os
import time
import math
import random
import warnings
from glob import glob
from collections import Counter

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms
import torchvision.ops as ops
from torchvision.ops import box_iou

from sklearn.metrics import average_precision_score

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device

    #################################################
    ############### HYPERPARAMETERS #################
    #################################################

    root_dir = '../data_scene_flow'

    batch_size = 32
    num_workers = 2

    num_epochs = 200

    #################################################
    #################################################
    #################################################

    # KITTI class names
    KITTI_CLASSES = [
        'Car', 'Van', 'Truck', 'Pedestrian',
        'Person_sitting', 'Cyclist', 'Tram', 'Misc'
    ]
    CLASS_MAP = {c: i for i, c in enumerate(KITTI_CLASSES)}

    # Visualization color map (RGB)
    orig_color_map = {
        'Car': (255, 0, 0),
        'Van': (255, 128, 0),
        'Truck': (200, 200, 0),
        'Pedestrian': (0, 255, 0),
        'Person_sitting': (0, 200, 200),
        'Cyclist': (0, 0, 255),
        'Tram': (128, 0, 255),
        'Misc': (255, 0, 255)
    }
    class_color_map = { CLASS_MAP[c]: orig_color_map[c] for c in KITTI_CLASSES }
    
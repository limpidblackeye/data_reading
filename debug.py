# debug.py

import os
import sys
import random
import math
import re
import time
import numpy as np
# import cv2
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from skimage.draw import circle, polygon
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from config import Config
import utils
import model as modellib
import visualize
from model import log

pred_result=np.load("pred_resuilt1.npz")['arr_0']

# ## ================ run-length encoding ================ ##
# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    print(lab_img.max())
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

rle_0=prob_to_rles(pred_result[0])
rle=list(rle_0)

# iterate over the test IDs and generate run-length encodings for each seperate mask identified by skimage
new_test_ids = []
rles = []
for n in range(len(dataset_predict.image_ids)):
    print(n)
    id_ = next(os.walk(TEST_PATH))[1][n]
    rle = list(prob_to_rles(pred_result[n]))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))

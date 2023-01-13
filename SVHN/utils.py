import random
from collections import defaultdict

import numpy as np
from keras import backend as K
from keras.models import Model

import os
import glob
import math

# util function to convert a tensor into a valid image
def deprocess_image(x):
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x.reshape((x.shape[1], x.shape[2], x.shape[3]))  # original shape (1,img_rows, img_cols,3)


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

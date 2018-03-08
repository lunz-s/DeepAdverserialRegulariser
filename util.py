import os
import tensorflow as tf
import fnmatch
import matplotlib
from xml.etree import ElementTree
import numpy as np
import random
import odl
import scipy.ndimage
import fnmatch
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import dicom as dc
from scipy.misc import imresize
import platform



def cut_image(pic):
    pic = np.maximum(pic, 0.0)
    pic = np.minimum(pic, 1.0)
    return pic

def normalize_image(pic):
    av = np.average(pic)
    pic = pic - av
    sigma = np.sqrt(np.average(np.square(pic)))
    pic = pic/(sigma + 0.001)
    return pic

def scale_to_unit_intervall(pic):
    min = np.amin(pic)
    pic = pic - min
    max = np.amax(pic)
    pic = pic/(max+0.0001)
    return pic

def create_single_folder(folder):
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except OSError:
            pass

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name).replace("\\", "/"))
    return result

def lrelu(x):
    return (tf.nn.relu(x) - 0.1*tf.nn.relu(-x))

# l2 norm for a tensor in typical (batch, x, y, channel) format
def l2_norm(tensor):
    return np.mean(np.sqrt(np.sum(np.square(tensor), axis=(1,2,3))))
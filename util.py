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
from skimage.measure import compare_ssim as ssim

def quality(truth, recon):
    recon = cut_image(recon)
    l2 = np.average(np.sqrt(np.sum(np.square(truth - recon), axis = (1,2,3))))
    psnr = - 10 * np.log10(np.average(np.square(truth - recon)))
    amount_images = truth.shape[0]
    ssi = 0
    for k in range(amount_images):
        ssi = ssi + ssim(truth[k,...,0], cut_image(recon[k,...,0]))
    ssi = ssi/amount_images
    return [l2, psnr, ssi]

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

# a dilated convolutional layer
def dilated_conv_layer(inputs, name, filters=16, kernel_size=(5, 5), padding="same", rate = 1,
                                 activation=lrelu, reuse=False):
    inputs_dim = inputs.get_shape().as_list()
    input_channels = inputs_dim[3]
    with tf.variable_scope(name, reuse=reuse):
        weights = tf.get_variable(name='weights', shape=[kernel_size[0], kernel_size[1], input_channels, filters],
                            initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable(name='bias', shape=[1, 1, 1, filters],
                            initializer=tf.zeros_initializer)
    conv = tf.nn.atrous_conv2d(inputs, weights, rate = rate, padding=padding)
    output = activation(tf.add(conv,bias))
    return output

# contracts an image tensor of shape [batch, size, size, channels] to its l1 values along the size dimensions
def image_l1(inputs):
    return tf.reduce_mean(tf.abs(inputs), axis = (1,2))



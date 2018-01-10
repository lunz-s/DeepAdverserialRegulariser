import tensorflow as tf
import os
import fnmatch
import matplotlib
from xml.etree import ElementTree
import numpy as np
import random
import odl
import odl.contrib.tensorflow
matplotlib.use('agg')
import matplotlib.pyplot as plt
import dicom as dc
from scipy.misc import imresize
import platform

class postprocesser(object):
    model_name = 'default'
    # The batch size
    batch_size = 64
    # learning rate for Adams
    learning_rate = 0.001
    # hard code image size
    image_size = (128,128)

    def create_folders(self):
        paths = {}
        paths['Image Folder'] = 'Saves/Pictures/' + self.model_name
        paths['Saves Folder'] = 'Saves/Data/' + self.model_name
        paths['Evaluations Folder'] = 'Saves/Evaluations/' + self.model_name
        paths['Logging Folder'] = 'Saves/Logs/' + self.model_name
        for key, value in paths.items():
            if not os.path.exists(value):
                try:
                    os.makedirs(value)
                except OSError:
                    pass
                print(key + ' created')

    # to be overwritten in subclass
    def get_training_data(self):
        return [],[]

    # to be overwritten in subclass
    def network(self, input):
        return input

    def __init__(self, colour = 3):
        self.colour = colour

        # create needed folders
        self.create_folders()
        # start a tensorflow session
        self.sess = tf.InteractiveSession()
        # set placeholder for input and correct output
        self.x = tf.placeholder(shape=[None, self.image_size[0], self.image_size[1], colour], dtype=tf.float32)
        self.y = tf.placeholder(shape=[None, self.image_size[0], self.image_size[1], colour], dtype=tf.float32)
        # network output
        self.out = self.network(self.x)
        # compute loss
        data_mismatch = tf.square(self.out - self.y)
        self.loss = tf.reduce_mean(tf.reduce_sum(data_mismatch, axis=(1, 2, 3)))
        # optimizer
        # optimizer for Wasserstein network
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                                global_step=self.global_step)
        # logging tools
        tf.summary.scalar('Loss', self.loss)

        # set up the logger
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('Saves/Logs/' + self.model_name + '/Network_Training/',
                                            self.sess.graph)

        # set up variables saver
        self.saver = tf.train.Saver()

        # initialize Variables
        tf.global_variables_initializer().run()

        # load existing saves
        self.load()

    def save(self):
        self.saver.save(self.sess, 'Saves/Data/' + self.model_name + '/model', global_step=self.global_step)
        print('Progress saved')

    def load(self):
        if os.listdir('Saves/Data/' + self.model_name + '/'):
            self.saver.restore(self.sess, tf.train.latest_checkpoint(os.path.join('Saves', 'Data', self.model_name, '')))
            print('Save restored')
        else:
            print('No save found')

    def log(self, x,y):
        summary, step = self.sess.run([self.merged, self.global_step],
                                      feed_dict={self.x : x,
                                                 self.y : y})
        self.writer.add_summary(summary, step)

    def train(self, steps):
        for k in range(steps):
            x, y = self.get_training_data()
            self.sess.run(self.optimizer, feed_dict={self.x : x,
                                                    self.y : y})
            if k%50 == 0:
                self.log(x,y)

class UNet(postprocesser):
    model_name = 'UNet'
    def network(self, input):
        # 128
        conv1 = tf.layers.conv2d(inputs=input, filters=32, kernel_size=[5, 5],
                                      padding="same", activation=tf.nn.relu)
        # 64
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5],
                                      padding="same", activation=tf.nn.relu)
        # 32
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        conv3 = tf.layers.conv2d(inputs=pool2, filters=64, kernel_size=[3, 3],
                                      padding="same", activation=tf.nn.relu)
        # 64
        conv4 = tf.layers.conv2d_transpose(inputs=conv3, filters=32, kernel_size=[5, 5],
                                           strides= (2,2), padding="same", activation=tf.nn.relu)
        concat1 = tf.concat([conv4, pool1], axis= 3)
        # 128
        conv5 =  tf.layers.conv2d_transpose(inputs=concat1, filters=32, kernel_size=[5, 5],
                                           strides= (2,2), padding="same", activation=tf.nn.relu)
        concat2 = tf.concat([conv5, input], axis= 3)
        output = tf.layers.conv2d(inputs=concat2, filters=self.colour, kernel_size=[5, 5],
                                  padding="same", activation=tf.nn.relu)
        return output



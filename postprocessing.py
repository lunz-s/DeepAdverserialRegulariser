import tensorflow as tf
import os
import fnmatch
import matplotlib
from xml.etree import ElementTree
import numpy as np
import random
import odl
import odl.contrib.tensorflow
import scipy.ndimage
import fnmatch
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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

    # methode to cut image to [0,1] value range
    @staticmethod
    def cut_image(pic):
        pic = np.maximum(pic, 0.0)
        pic = np.minimum(pic, 1.0)
        return pic

    # methode to create a single folder
    def create_single_folder(self, folder):
        if not os.path.exists(folder):
            try:
                os.makedirs(folder)
            except OSError:
                pass

    # to be overwritten in subclass
    def generate_images(self, batch_size, training_data=True):
        return np.zeros(1),np.zeros(1)

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
        self.true = tf.placeholder(shape=[None, self.image_size[0], self.image_size[1], colour], dtype=tf.float32)
        self.y = tf.placeholder(shape=[None, self.image_size[0], self.image_size[1], colour], dtype=tf.float32)
        # network output
        self.out = self.network(self.y)
        # compute loss
        data_mismatch = tf.square(self.out - self.true)
        self.loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(data_mismatch, axis=(1, 2, 3))))
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

    def end(self):
        tf.reset_default_graph()
        self.sess.close()

    def log(self, x,y):
        summary, step = self.sess.run([self.merged, self.global_step],
                                      feed_dict={self.x : x,
                                                 self.y : y})
        self.writer.add_summary(summary, step)

    def train(self, steps):
        for k in range(steps):
            x, y = self.generate_images(self.batch_size)
            self.sess.run(self.optimizer, feed_dict={self.true : x,
                                                    self.y : y})
            if k%50 == 0:
                iteration, loss = self.sess.run([self.global_step, self.loss], feed_dict={self.true : x,
                                                    self.y : y})
                print('Iteration: ' + str(iteration) + ', MSE: ' +str(loss))
                self.log(x,y)
                output = self.sess.run(self.out, feed_dict={self.true : x,
                                                    self.y : y})
                self.visualize(x, y, output, iteration)
        self.save()

    # visualization methode
    def visualize(self, true, noisy, recon, global_step):
        quality = np.average(np.sqrt(np.sum(np.square(true - recon), axis=(1, 2, 3))))
        print('Quality of reconstructed image: ' + str(quality))
        self.create_single_folder('Saves/Pictures/' + self.model_name + '/' +str(global_step))
        plt.figure()
        plt.subplot(131)
        plt.imshow(self.cut_image(true[-1, ...]))
        plt.axis('off')
        plt.title('Original')
        plt.subplot(132)
        plt.imshow(self.cut_image(noisy[-1,...]))
        plt.axis('off')
        plt.title('Corrupted')
        plt.suptitle('L2 :' + str(quality))
        plt.subplot(133)
        plt.imshow(self.cut_image(recon[-1,...]))
        plt.title('Reconstruction')
        plt.axis('off')
        plt.savefig('Saves/Pictures/' + self.model_name + '/' +
                    'iteration-' + str(global_step) + '.png')
        plt.close()

    def evaluate_pp(self, true, y):
        return self.sess.run(self.out, feed_dict={self.y : y, self.true: true})



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


class postDenoising(UNet):
    model_name = 'Denoising_UNet'
    noise_level = 0.03
    def __init__(self):
        self.train_list = self.find('*.jpg', './BSDS/images/train')
        self.train_amount = len(self.train_list)
        print('Training Pictures found: ' + str(self.train_amount))
        self.eval_list = self.find('*.jpg', './BSDS/images/val')
        self.eval_amount = len(self.eval_list)
        print('Evaluation Pictures found: ' + str(self.eval_amount))

        # call mother class init
        super(postDenoising, self).__init__()

    # creates list of training data
    @staticmethod
    def find(pattern, path):
        result = []
        for root, dirs, files in os.walk(path):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    result.append(os.path.join(root, name).replace("\\", "/"))
        return result


    # methode to draw raw picture samples
    def load_data(self, training_data=True):
        if training_data:
            rand = random.randint(0, self.train_amount - 1)
            pic = mpimg.imread(self.train_list[rand])
        else:
            rand = random.randint(0, self.eval_amount - 1)
            pic = scipy.ndimage.imread(self.eval_list[rand])
        return pic/255.0


    # visualize a single image
    @staticmethod
    def visualize_single_pic(pic, number):
        plt.figure()
        plt.imshow(postDenoising.cut_image(pic))
        plt.axis('off')
        if not os.path.exists('Saves/Test/'):
            try:
                os.makedirs('Saves/Test/')
            except OSError:
                pass
        plt.savefig('Saves/Test/' + str(number) + '.jpg')

    # Draw random edgepoint
    def edgepoint(self, x_size, y_size):
        x_vary = x_size - self.image_size[0]
        x_coor = random.randint(0, x_vary)
        y_vary = y_size - self.image_size[1]
        y_coor = random.randint(0, y_vary)
        upper_left = [x_coor, y_coor]
        lower_right = [x_coor + self.image_size[0], y_coor + self.image_size[1]]
        return upper_left, lower_right

    # methode to cut a image_size area out of the training images
    def load_exemple(self, training_data= True):
        pic = self.load_data(training_data=training_data)
        size = pic.shape
        ul, lr = self.edgepoint(size[0], size[1])
        image = pic[ul[0]:lr[0], ul[1]:lr[1],:]
        return image

    # methode to create training or evaluation batches
    def generate_images(self, batch_size, training_data=True):
        true = np.empty(shape=[batch_size, self.image_size[0], self.image_size[1], 3])
        cor = np.empty(shape=[batch_size, self.image_size[0], self.image_size[1], 3])

        for i in range(batch_size):
            image = self.load_exemple(training_data=training_data)
            # generate noise to put on picture
            noise = np.random.normal(0, self.noise_level, (self.image_size[0], self.image_size[1],3))
            image_cor = image + noise
            true[i,...] = image
            cor[i,...] = image_cor
        return true, cor
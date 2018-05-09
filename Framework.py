import random
import numpy as np
import scipy.ndimage
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import platform
import odl
import odl.contrib.tensorflow

import dicom as dc
from scipy.misc import imresize
import tensorflow as tf
import util as ut

from forward_models import ct
from data_pips import LUNA
from networks import binary_classifier
from networks import UNet
from networks import fully_convolutional
from data_pips import BSDS


# This class provides methods necessary
class generic_framework(object):
    model_name = 'no_model'
    experiment_name = 'default_experiment'

    # set the noise level used for experiments
    noise_level = 0.02

    # methods to define the models used in framework
    def get_network(self, size, colors):
        return binary_classifier(size=size, colors=colors)

    def get_Data_pip(self):
        return LUNA()

    def get_model(self, size):
        return ct(size=size)


    def __init__(self):
        self.data_pip = self.get_Data_pip()
        self.colors = self.data_pip.colors
        self.image_size = self.data_pip.image_size
        self.network = self.get_network(self.image_size, self.colors)
        self.model = self.get_model(self.image_size)
        self.image_space = self.model.get_image_size()
        self.measurement_space = self.model.get_measurement_size()


        # finding the correct path for saving models
        name = platform.node()
        path_prefix = ''
        if name == 'LAPTOP-E6AJ1CPF':
            path_prefix=''
        elif name == 'motel':
            path_prefix='/local/scratch/public/sl767/DeepAdversarialRegulariser/'
        self.path = path_prefix+'Saves/{}/{}/{}/{}/'.format(self.model.name, self.data_pip.name, self.model_name, self.experiment_name)
        # start tensorflow sesssion
        self.sess = tf.InteractiveSession()

        # generate needed folder structure
        self.generate_folders()

    # method to generate training data given the current model type
    def generate_training_data(self, batch_size, training_data = True):
        y = np.empty((batch_size, self.measurement_space[0], self.measurement_space[1], self.colors), dtype='float32')
        x_true = np.empty((batch_size, self.image_space[0], self.image_space[1], self.colors), dtype='float32')
        fbp = np.empty((batch_size, self.image_space[0], self.image_space[1], self.colors), dtype='float32')

        for i in range(batch_size):
            if training_data:
                image = self.data_pip.load_data(training_data=True)
            else:
                image = self.data_pip.load_data(training_data=False)
            for k in range(self.data_pip.colors):
                data = self.model.forward_operator(image[...,k])

                # add white Gaussian noise
                noisy_data = data + np.random.normal(size = self.measurement_space) * self.noise_level

                fbp [i, ..., k] = self.model.inverse(noisy_data)
                x_true[i, ..., k] = image[...,k]
                y[i, ..., k] = noisy_data
        return y, x_true, fbp

    # puts in place the folders needed to save the results obtained with the current model
    def generate_folders(self):
        paths = {}
        paths['Image Folder'] = self.path + 'Images'
        paths['Saves Folder'] = self.path + 'Data'
        paths['Logging Folder'] = self.path + 'Logs'
        for key, value in paths.items():
            if not os.path.exists(value):
                try:
                    os.makedirs(value)
                except OSError:
                    pass
                print(key + ' created')

    # visualizes the quality of the current method
    def visualize(self, true, fbp, guess, name):
        quality = np.average(np.sqrt(np.sum(np.square(true - guess), axis=(1, 2, 3))))
        print('Quality of reconstructed image: ' + str(quality))
        if self.colors == 1:
            t = true[-1,...,0]
            g = guess[-1, ...,0]
            p = fbp[-1, ...,0]
        else:
            t = true[-1,...]
            g = guess[-1, ...]
            p = fbp[-1, ...]
        plt.figure()
        plt.subplot(131)
        plt.imshow(ut.cut_image(t))
        plt.axis('off')
        plt.title('Original')
        plt.subplot(132)
        plt.imshow(ut.cut_image(p))
        plt.axis('off')
        plt.title('PseudoInverse')
        plt.suptitle('L2 :' + str(quality))
        plt.subplot(133)
        plt.imshow(ut.cut_image(g))
        plt.title('Reconstruction')
        plt.axis('off')
        plt.savefig(self.path + name + '.png')
        plt.close()

    def save(self, global_step):
        saver = tf.train.Saver()
        saver.save(self.sess, self.path+'Data/model', global_step=global_step)
        print('Progress saved')

    def load(self):
        saver = tf.train.Saver()
        if os.listdir(self.path+'Data/'):
            saver.restore(self.sess, tf.train.latest_checkpoint(self.path+'Data/'))
            print('Save restored')
        else:
            print('No save found')

    def end(self):
        tf.reset_default_graph()
        self.sess.close()

    ### generic method for subclasses
    def deploy(self, true, guess, measurement):
        pass

# Framework for the adversarial regulariser network
class adversarial_regulariser(generic_framework):
    model_name = 'Adversarial_Regulariser'
    # override noise level
    noise_level = 0.01
    # The batch size
    batch_size = 16
    # relation between L2 error and regulariser
    mu_default = 1.5
    # weight on gradient norm regulariser for wasserstein network
    lmb = 20
    # learning rate for Adams
    learning_rate = 0.0001
    # step size for picture optimization
    step_size = 1
    # the amount of steps of gradient descent taken on loss functional
    total_steps_default = 30
    # default sampling pattern
    default_sampling_pattern = 'uniform'

    def get_network(self, size, colors):
        return binary_classifier(size=size, colors=colors)

    def get_Data_pip(self):
        return LUNA()

    def get_model(self, size):
        return ct(size=size)

    def set_total_steps(self, steps):
        self.total_steps = steps

    def set_sampling_pattern(self, pattern=None):
        if pattern == None:
            self.sampling_pattern = self.default_sampling_pattern
        else:
            self.sampling_pattern = pattern

    # sets up the network architecture
    def __init__(self):
        # call superclass init
        super(adversarial_regulariser, self).__init__()
        self.total_steps = self.total_steps_default
        self.sampling_pattern = self.set_sampling_pattern()

        ### Training the regulariser

        # placeholders for NN
        self.gen_im = tf.placeholder(shape=[None, self.image_space[0], self.image_space[1], self.colors],
                                     dtype=tf.float32)
        self.true_im = tf.placeholder(shape=[None, self.image_space[0], self.image_space[1], self.colors],
                                      dtype=tf.float32)
        self.random_uint = tf.placeholder(shape=[None],
                                          dtype=tf.float32)

        # the network outputs
        self.gen_was = self.network.net(self.gen_im)
        self.data_was = self.network.net(self.true_im)

        # Wasserstein loss
        self.wasserstein_loss = tf.reduce_mean(self.data_was - self.gen_was)

        # intermediate point
        random_uint_exp = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.random_uint, axis=1), axis=1), axis=1)
        self.inter = tf.multiply(self.gen_im, random_uint_exp) + \
                     tf.multiply(self.true_im, 1 - random_uint_exp)
        self.inter_was = self.network.net(self.inter)

        # calculate derivative at intermediate point
        self.gradient_was = tf.gradients(self.inter_was, self.inter)[0]

        # take the L2 norm of that derivative
        self.norm_gradient = tf.sqrt(tf.reduce_sum(tf.square(self.gradient_was), axis=(1, 2, 3)))
        self.regulariser_was = tf.reduce_mean(tf.square(tf.nn.relu(self.norm_gradient - 1)))

        # Overall Net Training loss
        self.loss_was = self.wasserstein_loss + self.lmb * self.regulariser_was

        # optimizer for Wasserstein network
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss_was,
                                                                                global_step=self.global_step)

        ### The reconstruction network
        # placeholders
        self.reconstruction = tf.placeholder(shape=[None, self.image_space[0], self.image_space[0], self.colors],
                                             dtype=tf.float32)
        self.data_term = tf.placeholder(shape=[None, self.measurement_space[0], self.measurement_space[1], self.colors],
                                        dtype=tf.float32)
        self.mu = tf.placeholder(dtype=tf.float32)

        # data loss
        self.ray = self.model.tensorflow_operator(self.reconstruction)
        data_mismatch = tf.square(self.ray - self.data_term)
        self.data_error = tf.reduce_mean(tf.reduce_sum(data_mismatch, axis=(1, 2, 3)))

        # the loss functional
        self.was_output = tf.reduce_mean(self.network.net(self.reconstruction))
        self.full_error = self.mu * self.was_output + self.data_error

        # get the batch size - all gradients have to be scaled by the batch size as they are taken over previously
        # averaged quantities already
        batch_s = tf.cast(tf.shape(self.reconstruction)[0], tf.float32)

        # Optimization for the picture
        self.pic_grad = tf.gradients(self.full_error * batch_s, self.reconstruction)

        # Measure quality of reconstruction
        self.ground_truth = tf.placeholder(shape=[None, self.image_space[0], self.image_space[0], self.colors], dtype=tf.float32)
        self.quality = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(self.ground_truth - self.reconstruction),
                                                            axis=(1, 2, 3))))

        # logging tools
        with tf.name_scope('Network_Optimization'):
            tf.summary.scalar('Wasserstein_Loss', self.wasserstein_loss)
            tf.summary.scalar('Regulariser_Wasser', self.regulariser_was)
            tf.summary.scalar('Overall_Net_Loss', self.loss_was)
        with tf.name_scope('Picture_Optimization'):
            data_loss = tf.summary.scalar('Data_Loss', self.data_error)
            wasser_loss = tf.summary.scalar('Wasserstein_Loss', self.was_output)
        with tf.name_scope('Model_L2_strength'):
            quality_assesment = tf.summary.scalar('L2', self.quality)

        # set up the logger
        self.merged = tf.summary.merge_all()
        self.merged_pic = tf.summary.merge([data_loss, wasser_loss, quality_assesment])
        self.writer = tf.summary.FileWriter(self.path + 'Logs/Network_Optimization/',
                                            self.sess.graph)


        # initialize Variables
        tf.global_variables_initializer().run()

        # load existing saves
        self.load()

    # uses network to update picture with
    def update_pic(self, steps, stepsize, measurement, guess, mu):
        for k in range(steps):
            gradient = self.sess.run(self.pic_grad, feed_dict={self.reconstruction: guess,
                                                               self.data_term: measurement,
                                                               self.mu: mu})
            guess = guess - stepsize * gradient[0]
        return guess

    # unregularised minimization - finds minimizer of data term
    def unreg_mini(self, y, fbp):
        return self.update_pic(20, 0.1, y, fbp, 0)


    # visualization of Picture optimization
    def evaluate_image_optimization(self, batch_size = None, steps=None, step_s=None,
                                mu=None, starting_point=None):
        if batch_size == None:
            batch_size = self.batch_size
        if steps == None:
            steps= self.total_steps
        if step_s == None:
            step_s = self.step_size
        if mu == None:
            mu = self.mu_default
        if starting_point ==None:
            starting_point = 'Mini'
        print('Mu: {}, AmountSteps: {}, Step_size: {}, '
              'batch_size: {}, starting_point: {}'.format(mu,steps, step_s, batch_size, starting_point))
        y, x_true, fbp = self.generate_training_data(batch_size)
        guess = np.copy(fbp)
        if starting_point == 'Mini':
            guess = self.unreg_mini(y, fbp)
        g_step = self.sess.run(self.global_step)
        writer = tf.summary.FileWriter(self.path + '/Logs/Picture_Opt/Iteration_' +
                                       str(g_step) + '/' + str(mu) + '/')
        for k in range(steps):
            summary = self.sess.run(self.merged_pic,
                                    feed_dict={self.reconstruction: guess,
                                               self.data_term: y,
                                               self.ground_truth: x_true,
                                               self.mu: mu})
            writer.add_summary(summary, k)
            if (k % 5 == 0):
                wass = self.sess.run(self.wasserstein_loss, feed_dict={self.gen_im: guess,
                                                                       self.true_im: x_true})
                print('Remaining Wasserstein Distance: ' + str(wass))
                ut.create_single_folder(self.path + 'Images/Global_Step:_{}/'.format(g_step))
                self.visualize(x_true, fbp, guess, 'Images/Global_Step:_{}/Opt._Step:{}'.format(g_step, k))
            guess = self.update_pic(1, step_s, y, guess, mu)
        writer.close()

    def check_recursive_patter(self):
        true_im, output_fbp, output_im = self.generate_optimized_images(16)
        ut.create_single_folder(self.path + 'Images/Test_Sampling')
        self.visualize(true_im, output_fbp, output_im, 'Images/Test_Sampling/1')
        for k in range(16):
            print(np.sqrt(np.sum(np.square(output_im[k,...]-true_im[k,...]))))
            print(np.sqrt(np.sum(np.square(output_im[k,...]))))


    # evaluates and prints the network performance
    def evaluate_Network(self, mu = None, amount_steps = None, starting_point=None):
        if amount_steps == None:
            amount_steps = self.total_steps
        if mu == None:
            mu = self.mu_default
        if starting_point ==None:
            starting_point = 'Mini'
        y, true, fbp = self.generate_optimized_images(batch_size=self.batch_size)
        # generate random distribution for rays
        epsilon = np.random.uniform(size=(self.batch_size))
        step, Was, reg, norm = self.sess.run([self.global_step, self.wasserstein_loss, self.regulariser_was,
                                        self.norm_gradient],
                                                     feed_dict={self.gen_im: fbp, self.true_im: true,
                                                                self.random_uint: epsilon})
        print('Iteration prior: ' + str(step) + ', Was: ' + str(Was) + ', Reg: ' + str(reg))

        # tensorflow logging
        guess = np.copy(fbp)
        guess = self.update_pic(amount_steps, self.step_size, y, guess, mu)
        summary, step = self.sess.run([self.merged, self.global_step],
                                      feed_dict={self.gen_im: fbp,
                                                 self.true_im: true,
                                                 self.random_uint: epsilon,
                                                 self.reconstruction: guess,
                                                 self.data_term: y,
                                                 self.ground_truth: true,
                                                 self.mu: mu})
        self.writer.add_summary(summary, step)

        # print posterior regression data term parameters
        step, Was, reg = self.sess.run([self.global_step, self.wasserstein_loss, self.regulariser_was],
                                                     feed_dict={self.gen_im: guess, self.true_im: true,
                                                                self.random_uint: epsilon})
        print('Iteration posterior: ' + str(step) + ', Was: ' + str(Was) + ', Reg: ' + str(reg))

    # method to generate new training images using posterior distribution of the algorithm itself
    def generate_optimized_images(self, batch_size = None,
                                  amount_steps = None, mu=None, starting_point=None):
        if amount_steps == None:
            amount_steps = self.total_steps
        if batch_size == None:
            batch_size = self.batch_size
        if mu == None:
            mu = self.mu_default
        if starting_point ==None:
            starting_point = 'Mini'

        true_im = np.zeros(shape=(batch_size, 128, 128, self.colors))
        output_im = np.zeros(shape=(batch_size, 128, 128, self.colors))
        output_fbp = np.zeros(shape=(batch_size, 128, 128, self.colors))
        ### speed up by only drawing randomly for batches of 8 or even 16

        if self.sampling_pattern == 'uniform':
            for j in range(batch_size):
                y, x_true, fbp = self.generate_training_data(1)
                guess = np.copy(fbp)
                if starting_point == 'Mini':
                    guess = self.unreg_mini(y, fbp)
                s = random.randint(0, amount_steps)
                guess = self.update_pic(s, self.step_size, y, guess, mu)
                true_im[j, ...] = x_true[0, ...]
                output_fbp[j, ...] = fbp[0, ...]
                output_im[j, ...] = guess[0, ...]
        if self.sampling_pattern == 'startend':
            halftime = int(batch_size/2)
            for k in range(2):
                y, x_true, fbp = self.generate_training_data(halftime)
                guess = np.copy(fbp)
                if starting_point == 'Mini':
                    guess = self.unreg_mini(y, fbp)
                if k == 1:
                    result = self.update_pic(amount_steps, self.step_size, y, guess, mu)
                else:
                    result = guess
                true_im[k*halftime:(k+1)*halftime,...] = x_true
                output_fbp[k * halftime:(k + 1) * halftime, ...] = fbp
                output_im[k * halftime:(k + 1) * halftime, ...] = result

        return true_im, output_fbp, output_im

    # optimize network on initial guess input only, with initial guess being fbp
    def pretrain_Wasser_FBP(self, steps, mu=None):
        if mu == None:
            mu = self.mu_default
        for k in range(steps):
            if k % 20 == 0:
                self.evaluate_Network(mu = mu, starting_point='FBP')
            if k % 100 == 0:
                self.evaluate_image_optimization(starting_point='FBP')
            y, x_true, fbp = self.generate_training_data(self.batch_size)
            # generate random distribution for rays
            epsilon = np.random.uniform(size=(self.batch_size))
            # optimize network
            self.sess.run(self.optimizer,
                          feed_dict={self.gen_im: fbp, self.true_im: x_true, self.random_uint: epsilon})
        self.save(self.global_step)

    # optimize network on initial guess input only, with initial guess being minimizer of ||Kx - y||
    def pretrain_Wasser_DataMinimizer(self, steps, mu=None):
        if mu == None:
            mu = self.mu_default
        for k in range(steps):
            if k % 20 == 0:
                self.evaluate_Network(mu, starting_point='Mini')
            if k % 100 == 0:
                self.evaluate_image_optimization(64, starting_point='Mini')
            y, x_true, fbp = self.generate_training_data(self.batch_size)
            # optimize the fbp to fit the data term
            mini = self.unreg_mini(y, fbp)
            # generate random distribution for rays
            epsilon = np.random.uniform(size=(self.batch_size))
            # optimize network
            self.sess.run(self.optimizer,
                          feed_dict={self.gen_im: mini, self.true_im: x_true, self.random_uint: epsilon})
        self.save(self.global_step)

    # recursive training methode, using actual output distribtion instead of initial guess distribution
    def train(self, steps, amount_steps = None, starting_point = None, mu=None):
        if amount_steps == None:
            amount_steps = self.total_steps
        if mu == None:
            mu = self.mu_default
        if starting_point == None:
            starting_point = 'Mini'
        for k in range(steps):
            if k % 20 == 0:
                self.evaluate_Network(mu, starting_point=starting_point)
            if k % 200 == 0:
                self.evaluate_image_optimization(batch_size=self.batch_size, steps=amount_steps,
                                                 starting_point=starting_point, step_s=self.step_size)
            true, fbp, gen = self.generate_optimized_images(self.batch_size, amount_steps=amount_steps,
                                                           mu=mu, starting_point=starting_point)
            # generate random distribution for rays
            epsilon = np.random.uniform(size=(self.batch_size))
            # optimize network
            self.sess.run(self.optimizer,
                          feed_dict={self.gen_im: gen, self.true_im: true, self.random_uint: epsilon})
        self.save(self.global_step)

    # Method to estimate a good value of the regularisation paramete.
    # This is done via estimation of 2 ||K^t (Kx-y)||_2 where x is the ground truth
    def find_good_lambda(self, sample = 64):
        ### compute optimal lambda with as well
        y, x_true, fbp = self.generate_training_data(sample)
        gradient_truth = self.sess.run(self.pic_grad, {self.reconstruction: x_true,
                                                 self.data_term: y,
                                                 self.ground_truth: x_true,
                                                 self.mu: 0})
        print(np.sqrt(np.sum(np.square(gradient_truth[0]), axis=(1,2,3))))
        print(np.mean(np.sqrt(np.sum(np.square(gradient_truth[0]), axis=(1,2,3)))))

# Framework for the adversarial regulariser network
class positiv_adversarial_regulariser(generic_framework):
    model_name = 'Adversarial_Regulariser'
    # override noise level
    noise_level = 0.01
    # The batch size
    batch_size = 32
    # relation between L2 error and regulariser
    mu_default = 1.5
    # weight on gradient norm regulariser for wasserstein network
    lmb = 20
    # learning rate for Adams
    learning_rate = 0.0002
    # step size for picture optimization
    step_size = 1
    # the amount of steps of gradient descent taken on loss functional
    total_steps_default = 30
    # the weighting of the term enforcing average value 0
    eps = 0.1

    def get_network(self, size, colors):
        return binary_classifier(size=size, colors=colors)

    def get_Data_pip(self):
        return LUNA()

    def get_model(self, size):
        return ct(size=size)

    def set_total_steps(self, steps):
        self.total_steps = steps

    # sets up the network architecture
    def __init__(self):
        # call superclass init
        super(positiv_adversarial_regulariser, self).__init__()
        self.total_steps = self.total_steps_default

        ### Training the regulariser

        # placeholders for NN
        self.gen_im = tf.placeholder(shape=[None, self.image_space[0], self.image_space[1], 1],
                                     dtype=tf.float32)
        self.true_im = tf.placeholder(shape=[None, self.image_space[0], self.image_space[1], 1],
                                      dtype=tf.float32)
        self.random_uint = tf.placeholder(shape=[None],
                                          dtype=tf.float32)

        # the network outputs
        self.gen_was = tf.nn.abs(self.network.net(self.gen_im))
        self.data_was = tf.nn.abs(self.network.net(self.true_im))

        # Wasserstein loss
        self.wasserstein_loss = tf.reduce_mean(self.data_was - self.gen_was)

        # intermediate point
        random_uint_exp = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.random_uint, axis=1), axis=1), axis=1)
        self.inter = tf.multiply(self.gen_im, random_uint_exp) + \
                     tf.multiply(self.true_im, 1 - random_uint_exp)
        self.inter_was = tf.nn.abs(self.network.net(self.inter))

        # calculate derivative at intermediate point
        self.gradient_was = tf.gradients(self.inter_was, self.inter)[0]

        # take the L2 norm of that derivative
        self.regulariser_was = tf.reduce_mean(tf.square(tf.nn.relu(tf.sqrt(
            tf.reduce_sum(tf.square(self.gradient_was), axis=(1, 2, 3))) - 1)))

        # term that ensures that network takes value 0 on data manifold
        self.zero_enf = tf.reduce_mean(tf.reduce_sum(tf.square(self.data_was), axis=(1,2,3)))

        # Overall Net Training loss
        self.loss_was = self.wasserstein_loss + self.lmb * self.regulariser_was + self.eps * self.zero_enf

        # optimizer for Wasserstein network
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss_was,
                                                                                global_step=self.global_step)

        ### The reconstruction network
        # placeholders
        self.reconstruction = tf.placeholder(shape=[None, self.image_space[0], self.image_space[0], 1],
                                             dtype=tf.float32)
        self.data_term = tf.placeholder(shape=[None, self.measurement_space[0], self.measurement_space[1], 1],
                                        dtype=tf.float32)
        self.mu = tf.placeholder(dtype=tf.float32)

        # data loss
        self.ray = self.model.tensorflow_operator(self.reconstruction)
        data_mismatch = tf.square(self.ray - self.data_term)
        self.data_error = tf.reduce_mean(tf.reduce_sum(data_mismatch, axis=(1, 2, 3)))

        # the loss functional
        self.was_output = tf.reduce_mean(tf.nn.abs(self.network.net(self.reconstruction)))
        self.full_error = self.mu * self.was_output + self.data_error

        # get the batch size - all gradients have to be scaled by the batch size as they are taken over previously
        # averaged quantities already
        batch_s = tf.cast(tf.shape(self.reconstruction)[0], tf.float32)

        # Optimization for the picture
        self.pic_grad = tf.gradients(self.full_error * batch_s, self.reconstruction)

        # Measure quality of reconstruction
        self.ground_truth = tf.placeholder(shape=[None, self.image_space[0], self.image_space[0], 1],
                                           dtype=tf.float32)
        self.quality = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(self.ground_truth - self.reconstruction),
                                                            axis=(1, 2, 3))))

        # logging tools
        with tf.name_scope('Network_Optimization'):
            tf.summary.scalar('Wasserstein_Loss', self.wasserstein_loss)
            tf.summary.scalar('Regulariser_Wasser', self.lmb*self.regulariser_was)
            tf.summary.scalar('Zero_enforcer', self.eps* self.zero_enf)
            tf.summary.scalar('Overall_Net_Loss', self.loss_was)
        with tf.name_scope('Picture_Optimization'):
            data_loss = tf.summary.scalar('Data_Loss', self.data_error)
            wasser_loss = tf.summary.scalar('Wasserstein_Loss', self.was_output)
        with tf.name_scope('Model_L2_strength'):
            quality_assesment = tf.summary.scalar('L2', self.quality)

        # set up the logger
        self.merged = tf.summary.merge_all()
        self.merged_pic = tf.summary.merge([data_loss, wasser_loss, quality_assesment])
        self.writer = tf.summary.FileWriter(self.path + 'Logs/Network_Optimization/',
                                            self.sess.graph)

        # initialize Variables
        tf.global_variables_initializer().run()

        # load existing saves
        self.load()

    # uses network to update picture with
    def update_pic(self, steps, stepsize, measurement, guess, mu):
        for k in range(steps):
            gradient = self.sess.run(self.pic_grad, feed_dict={self.reconstruction: guess,
                                                               self.data_term: measurement,
                                                               self.mu: mu})
            guess = guess - stepsize * gradient[0]
        return guess

    # unregularised minimization - finds minimizer of data term
    def unreg_mini(self, y, fbp):
        return self.update_pic(20, 0.1, y, fbp, 0)

    # visualization of Picture optimization
    def evaluate_image_optimization(self, batch_size=None, steps=None, step_s=None,
                                    mu=None, starting_point=None):
        if batch_size == None:
            batch_size = self.batch_size
        if steps == None:
            steps = self.total_steps
        if step_s == None:
            step_s = self.step_size
        if mu == None:
            mu = self.mu_default
        if starting_point == None:
            starting_point = 'Mini'
        print('Mu: {}, AmountSteps: {}, Step_size: {}, '
              'batch_size: {}, starting_point: {}'.format(mu, steps, step_s, batch_size, starting_point))
        y, x_true, fbp = self.generate_training_data(batch_size)
        guess = np.copy(fbp)
        if starting_point == 'Mini':
            guess = self.unreg_mini(y, fbp)
        g_step = self.sess.run(self.global_step)
        writer = tf.summary.FileWriter(self.path + '/Logs/Picture_Opt/Iteration_' +
                                       str(g_step) + '/' + str(mu) + '/')
        for k in range(steps):
            summary = self.sess.run(self.merged_pic,
                                    feed_dict={self.reconstruction: guess,
                                               self.data_term: y,
                                               self.ground_truth: x_true,
                                               self.mu: mu})
            writer.add_summary(summary, k)
            if (k % 5 == 0):
                ut.create_single_folder(self.path + 'Images/Global_Step:_{}/'.format(g_step))
                self.visualize(x_true, fbp, guess, 'Images/Global_Step:_{}/Opt._Step:{}'.format(g_step, k))
            guess = self.update_pic(1, step_s, y, guess, mu)
        writer.close()

    # evaluates and prints the network performance
    def evaluate_Network(self, mu=None, amount_steps=None, starting_point=None):
        if amount_steps == None:
            amount_steps = self.total_steps
        if mu == None:
            mu = self.mu_default
        if starting_point == None:
            starting_point = 'Mini'
        y, true, fbp = self.generate_training_data(batch_size=self.batch_size)
        if starting_point == 'Mini':
            fbp = self.unreg_mini(y, fbp)
        # generate random distribution for rays
        epsilon = np.random.uniform(size=(self.batch_size))
        step, Was, reg = self.sess.run([self.global_step, self.wasserstein_loss, self.regulariser_was],
                                       feed_dict={self.gen_im: fbp, self.true_im: true,
                                                  self.random_uint: epsilon})
        print('Iteration prior: ' + str(step) + ', Was: ' + str(Was) + ', Reg: ' + str(reg))

        # tensorflow logging
        guess = np.copy(fbp)
        guess = self.update_pic(amount_steps, self.step_size, y, guess, mu)
        summary, step = self.sess.run([self.merged, self.global_step],
                                      feed_dict={self.gen_im: fbp,
                                                 self.true_im: true,
                                                 self.random_uint: epsilon,
                                                 self.reconstruction: guess,
                                                 self.data_term: y,
                                                 self.ground_truth: true,
                                                 self.mu: mu})
        self.writer.add_summary(summary, step)

        # print posterior regression data term parameters
        step, Was, reg = self.sess.run([self.global_step, self.wasserstein_loss, self.regulariser_was],
                                       feed_dict={self.gen_im: guess, self.true_im: true,
                                                  self.random_uint: epsilon})
        print('Iteration posterior: ' + str(step) + ', Was: ' + str(Was) + ', Reg: ' + str(reg))

    # method to generate new training images using posterior distribution of the algorithm itself
    def generate_optimized_images(self, batch_size=None,
                                  amount_steps=None, mu=None, starting_point=None):
        if amount_steps == None:
            amount_steps = self.total_steps
        if batch_size == None:
            batch_size = self.batch_size
        if mu == None:
            mu = self.mu_default
        if starting_point == None:
            starting_point = 'Mini'

        true_im = np.zeros(shape=(batch_size, 128, 128, 1))
        output_im = np.zeros(shape=(batch_size, 128, 128, 1))
        output_fbp = np.zeros(shape=(batch_size, 128, 128, 1))
        ### speed up by only drawing randomly for batches of 8 or even 16

        # create remaining samples
        for j in range(batch_size):
            y, x_true, fbp = self.generate_training_data(1)
            guess = np.copy(fbp)
            if starting_point == 'Mini':
                guess = self.unreg_mini(y, fbp)
            s = random.randint(0, amount_steps)
            guess = self.update_pic(s, self.step_size, y, guess, mu)
            true_im[j, ...] = x_true[0, ...]
            output_fbp[j, ...] = fbp[0, ...]
            output_im[j, ...] = guess[0, ...]
        return true_im, output_fbp, output_im

    # optimize network on initial guess input only, with initial guess being fbp
    def pretrain_Wasser_FBP(self, steps, mu=None):
        if mu == None:
            mu = self.mu_default
        for k in range(steps):
            if k % 20 == 0:
                self.evaluate_Network(mu=mu, starting_point='FBP')
            if k % 100 == 0:
                self.evaluate_image_optimization(starting_point='FBP')
            y, x_true, fbp = self.generate_training_data(self.batch_size)
            # generate random distribution for rays
            epsilon = np.random.uniform(size=(self.batch_size))
            # optimize network
            self.sess.run(self.optimizer,
                          feed_dict={self.gen_im: fbp, self.true_im: x_true, self.random_uint: epsilon})
        self.save(self.global_step)

    # optimize network on initial guess input only, with initial guess being minimizer of ||Kx - y||
    def pretrain_Wasser_DataMinimizer(self, steps, mu=None):
        if mu == None:
            mu = self.mu_default
        for k in range(steps):
            if k % 20 == 0:
                self.evaluate_Network(mu, starting_point='Mini')
            if k % 100 == 0:
                self.evaluate_image_optimization(64, starting_point='Mini')
            y, x_true, fbp = self.generate_training_data(self.batch_size)
            # optimize the fbp to fit the data term
            mini = self.unreg_mini(y, fbp)
            # generate random distribution for rays
            epsilon = np.random.uniform(size=(self.batch_size))
            # optimize network
            self.sess.run(self.optimizer,
                          feed_dict={self.gen_im: mini, self.true_im: x_true, self.random_uint: epsilon})
        self.save(self.global_step)

    # recursive training methode, using actual output distribtion instead of initial guess distribution
    def train(self, steps, amount_steps=None, starting_point=None, mu=None):
        if amount_steps == None:
            amount_steps = self.total_steps
        if mu == None:
            mu = self.mu_default
        if starting_point == None:
            starting_point = 'Mini'
        for k in range(steps):
            if k % 20 == 0:
                self.evaluate_Network(mu, starting_point=starting_point)
            if k % 200 == 0:
                self.evaluate_image_optimization(batch_size=self.batch_size, steps=amount_steps,
                                                 starting_point=starting_point, step_s=self.step_size)
            true, fbp, gen = self.generate_optimized_images(self.batch_size, amount_steps=amount_steps,
                                                            mu=mu, starting_point=starting_point)
            # generate random distribution for rays
            epsilon = np.random.uniform(size=(self.batch_size))
            # optimize network
            self.sess.run(self.optimizer,
                          feed_dict={self.gen_im: gen, self.true_im: true, self.random_uint: epsilon})
        self.save(self.global_step)

    # Method to estimate a good value of the regularisation paramete.
    # This is done via estimation of 2 ||K^t (Kx-y)||_2 where x is the ground truth
    def find_good_lambda(self, sample=64):
        ### compute optimal lambda with as well
        y, x_true, fbp = self.generate_training_data(sample)
        gradient_truth = self.sess.run(self.pic_grad, {self.reconstruction: x_true,
                                                       self.data_term: y,
                                                       self.ground_truth: x_true,
                                                       self.mu: 0})
        print(np.sqrt(np.sum(np.square(gradient_truth[0]), axis=(1, 2, 3))))
        print(np.mean(np.sqrt(np.sum(np.square(gradient_truth[0]), axis=(1, 2, 3)))))

# Framework for postprocessing
class postprocessing(generic_framework):
    model_name = 'PostProcessing'

    # learning rate for Adams
    learning_rate = 0.001
    # The batch size
    batch_size = 64

    # methods to define the models used in framework
    def get_network(self, size, colors):
        return UNet(size=size, colors=colors)

    def get_Data_pip(self):
        return LUNA()

    def get_model(self, size):
        return ct(size=size)

    def __init__(self):
        # call superclass init
        super(postprocessing, self).__init__()

        # set placeholder for input and correct output
        self.true = tf.placeholder(shape=[None, self.image_space[0], self.image_space[1], self.data_pip.colors], dtype=tf.float32)
        self.y = tf.placeholder(shape=[None, self.image_space[0], self.image_space[1], self.data_pip.colors], dtype=tf.float32)
        # network output
        self.out = self.network.net(self.y)
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
        self.writer = tf.summary.FileWriter(self.path + 'Logs/',
                                            self.sess.graph)

        # initialize Variables
        tf.global_variables_initializer().run()

        # load existing saves
        self.load()

    def log(self, x, y):
        summary, step = self.sess.run([self.merged, self.global_step],
                                      feed_dict={self.true : x,
                                                 self.y : y})
        self.writer.add_summary(summary, step)

    def train(self, steps):
        for k in range(steps):
            y, x_true, fbp = self.generate_training_data(self.batch_size)
            self.sess.run(self.optimizer, feed_dict={self.true : x_true,
                                                    self.y : fbp})
            if k%50 == 0:
                iteration, loss = self.sess.run([self.global_step, self.loss], feed_dict={self.true : x_true,
                                                    self.y : fbp})
                print('Iteration: ' + str(iteration) + ', MSE: ' +str(loss))

                # logging has to be adopted
                self.log(x_true,fbp)
                output = self.sess.run(self.out, feed_dict={self.true : x_true,
                                                    self.y : fbp})
                self.visualize(x_true, fbp, output, 'Iteration_{}'.format(iteration))
        self.save(self.global_step)

    def evaluate(self):
        y, x_true, fbp = self.generate_training_data(self.batch_size)

# implementation of iterative scheme from Jonas and Ozans paper
class iterative_scheme(generic_framework):
    model_name = 'Learned_gradient_descent'

    # hyperparameters
    iterations = 3
    learning_rate = 0.001
    # The batch size
    batch_size = 32

    def get_network(self, size, colors):
        return fully_convolutional(size=size, colors=colors)

    def __init__(self):
        # call superclass init
        super(iterative_scheme, self).__init__()

        # set placeholder for input and correct output
        self.true = tf.placeholder(shape=[None, self.image_space[0], self.image_space[1], self.data_pip.colors],
                                   dtype=tf.float32)
        self.guess = tf.placeholder(shape=[None, self.image_space[0], self.image_space[1], self.data_pip.colors],
                                   dtype=tf.float32)
        self.y = tf.placeholder(shape=[None, self.measurement_space[0], self.measurement_space[1], self.data_pip.colors],
                                dtype=tf.float32)


        # network output - iterative scheme
        x = self.guess
        for i in range(self.iterations):
            measurement = self.model.tensorflow_operator(x)
            g_x = self.model.tensorflow_adjoint_operator(self.y - measurement)
            tf.summary.image('Data_gradient', g_x, max_outputs=1)
            tf.summary.scalar('Data_gradient_Norm', tf.norm(g_x))
            # network input
            net_input = tf.concat([x, g_x], axis=3)

            # use the network model defined in
            x_update = self.network.net(net_input)
            tf.summary.scalar('x_update', tf.norm(x_update))
            x = x - x_update
            tf.summary.image('Current_Guess', x, max_outputs=1)
        self.out = x

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
        self.writer = tf.summary.FileWriter(self.path + 'Logs/',
                                            self.sess.graph)

        # initialize Variables
        tf.global_variables_initializer().run()

        # load existing saves
        self.load()

    def train(self, steps):
        for k in range(steps):
            y, x_true, fbp = self.generate_training_data(self.batch_size)
            self.sess.run(self.optimizer, feed_dict={self.true : x_true,
                                                    self.y : y,
                                                    self.guess : fbp})
            if k%10 == 0:
                summary, iteration, loss, output = self.sess.run([self.merged, self.global_step, self.loss, self.out],
                                                         feed_dict={self.true : x_true,
                                                                    self.y : y,
                                                                    self.guess : fbp})
                print('Iteration: ' + str(iteration) + ', MSE: ' +str(loss) + ', Original Error: '
                      + str(ut.l2_norm(x_true - fbp)))

                self.writer.add_summary(summary, iteration)
                self.visualize(x_true, fbp, output, 'Iteration_{}'.format(iteration))

        self.save(self.global_step)

# TV reconstruction
class total_variation(generic_framework):
    model_name = 'TV'

    # TV hyperparameters
    noise_level = 0.01
    def_lambda = 0.0013

    def __init__(self):
        # call superclass init
        super(total_variation, self).__init__()
        self.space = self.model.get_odl_space()
        self.operator = self.model.get_odl_operator()
        self.range = self.operator.range

    def tv_reconstruction(self, y, param=def_lambda):
        # the operators
        gradients = odl.Gradient(self.space, method='forward')
        broad_op = odl.BroadcastOperator(self.operator, gradients)
        # define empty functional to fit the chambolle_pock framework
        g = odl.solvers.ZeroFunctional(broad_op.domain)

        # the norms
        l1_norm = param * odl.solvers.L1Norm(gradients.range)
        l2_norm_squared = odl.solvers.L2NormSquared(self.range).translated(y)
        functional = odl.solvers.SeparableSum(l2_norm_squared, l1_norm)

        # Find parameters
        op_norm = 1.1 * odl.power_method_opnorm(broad_op)
        tau = 10.0 / op_norm
        sigma = 0.1 / op_norm
        niter = 500

        # find starting point
        x = self.range.element(self.model.inverse(y))

        # Run the optimization algoritm
        # odl.solvers.chambolle_pock_solver(x, functional, g, broad_op, tau = tau, sigma = sigma, niter=niter)
        odl.solvers.pdhg(x, functional, g, broad_op, tau=tau, sigma=sigma, niter=niter)
        return x

    def find_TV_lambda(self, lmd):
        amount_test_images = 32
        y, true, cor = self.generate_training_data(amount_test_images)
        for l in lmd:
            error = np.zeros(amount_test_images)
            or_error = np.zeros(amount_test_images)
            for k in range(amount_test_images):
                recon = self.tv_reconstruction(y[k, ..., 0], l)
                error[k] = np.sum(np.square(recon - true[k, ..., 0]))
                or_error[k] = np.sum(np.square(cor[k, ..., 0] - true[k, ..., 0]))
            total_e = np.mean(np.sqrt(error))
            total_o = np.mean(np.sqrt(or_error))
            print('Lambda: ' + str(l) + ', MSE: ' + str(total_e) + ', OriginalError: ' + str(total_o))



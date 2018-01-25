import AR_for_denoising as ar
import util as ut
import math
import os
import random
import numpy as np
import tensorflow as tf
import scipy.ndimage
import fnmatch
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class network(object):
    def get_weights(self):
        pass

    def wasserstein_network(self, weights, input_pic):
        pass

class convnet1(network):
    def get_weights(self):
        con1 = tf.get_variable(name="conv1_ad", shape=[5, 5, 16, 32],
                               initializer=(
                                   tf.contrib.layers.xavier_initializer_conv2d(uniform=False, dtype=tf.float32)))
        bias1 = tf.Variable(tf.constant(0.1, shape=[1, 1, 1, 32]), name="bias1_ad")
        con2 = tf.get_variable(name="conv2_ad", shape=[5, 5, 32, 64],
                               initializer=(
                                   tf.contrib.layers.xavier_initializer_conv2d(uniform=False, dtype=tf.float32)))
        bias2 = tf.Variable(tf.constant(0.1, shape=[1, 1, 1, 64]), name="bias2_ad")
        con3 = tf.get_variable(name="conv3_ad", shape=[5, 5, 64, 128],
                               initializer=(
                                   tf.contrib.layers.xavier_initializer_conv2d(uniform=False, dtype=tf.float32)))
        bias3 = tf.Variable(tf.constant(0.1, shape=[1, 1, 1, 128]), name="bias3_ad")
        con4 = tf.get_variable(name="conv4_ad", shape=[5, 5, 128, 128],
                               initializer=(
                                   tf.contrib.layers.xavier_initializer_conv2d(uniform=False, dtype=tf.float32)))
        bias4 = tf.Variable(tf.constant(0.1, shape=[1, 1, 1, 128]), name="bias4_ad")
        logits_W = tf.get_variable(name="logits_W_ad", shape=[128 * 8 * 8, 1],
                                   initializer=(
                                       tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32)))
        logits_bias = tf.Variable(tf.constant(0.0, shape=[1, 1]), name='logits_bias_ad')
        con_pre = tf.get_variable(name="conv_pre", shape=[5, 5, 3, 16],
                                  initializer=(
                                      tf.contrib.layers.xavier_initializer_conv2d(uniform=False, dtype=tf.float32)))
        bias_pre = tf.Variable(tf.constant(0.1, shape=[1, 1, 1, 16]), name="bias_pre")

        return [con1, bias1, con2, bias2, con3, bias3, con4, bias4, logits_W, logits_bias, con_pre, bias_pre]

    def wasserstein_network(self, weights, input_pic):
        # convolutional layer without downsampling
        conv_pre = ut.lrelu(tf.nn.conv2d(input_pic, weights[10], strides=[1, 1, 1, 1], padding='SAME') + weights[11])

        # 1st convolutional layer (pic size 128)
        conv1 = ut.lrelu(tf.nn.conv2d(conv_pre, weights[0], strides=[1, 2, 2, 1], padding='SAME') + weights[1])

        # 2nd conv layer (pic size 64)
        conv2 = ut.lrelu(tf.nn.conv2d(conv1, weights[2], strides=[1, 2, 2, 1], padding='SAME') + weights[3])

        # 3rd conv layer (pic size 32)
        conv3 = ut.lrelu(tf.nn.conv2d(conv2, weights[4], strides=[1, 2, 2, 1], padding='SAME') + weights[5])

        # 4th conv layer (pic size 16)
        conv4 = ut.lrelu(tf.nn.conv2d(conv3, weights[6], strides=[1, 2, 2, 1], padding='SAME') + weights[7])

        # reshape (pic size 8)
        p2resh = tf.reshape(conv4, [-1, 128 * 8 * 8])

        # dropout layer
        # drop = tf.layers.dropout(p2resh, 0.3, training=True)

        # logits
        output = tf.matmul(p2resh, weights[8]) + weights[9]
        return output

class single_stack(object):

    # method to save the network parameters
    def save(self):
        self.saver.save(self.sess, self.path + 'model', global_step=self.global_step)
        print('Progress saved')

    # method to load latest save
    def load(self):
        if os.listdir(self.path):
            self.saver.restore(self.sess, self.path)
            print('Restored: ' + str(self.stack))
        else:
            print('No save found: ' + str(self.stack))

    # generates the computational graph for the current layer in tensorflow
    def __init__(self, network, sess, image_size, batch_size, mu, lmb, learning_rate, step_size, model_name, stack):
        # Background objects
        self.sess = sess
        self.image_size = image_size

        # hyperparameters for Networks. Provided by generating class
        self.batch_size = batch_size
        self.mu_default = mu
        self.lmb = lmb
        self.learning_rate = learning_rate
        self.step_size = step_size

        # Saving Path and Stack name
        self.model_name = model_name
        self.stack = stack
        self.path = 'Saves/Data/{}/{}/'.format(self.model_name, self.stack)
        self.logging_path = 'Saves/Logs/{}/{}/'.format(self.model_name, self.stack)
        # The Network class object that provides the network methods
        self.network = network

        # create the folders needed for saving and logging
        ut.create_single_folder(self.logging_path)
        ut.create_single_folder(self.path)


        ### the Network
        # ensure that generated variables are saved in different namespace, given by current stack
        with tf.variable_scope(stack):
            # placeholders for NN
            self.gen_im = tf.placeholder(shape=[None, self.image_size[0], self.image_size[1], 3],
                                         dtype=tf.float32, name='gen_im')
            self.true_im = tf.placeholder(shape=[None, self.image_size[0], self.image_size[1], 3],
                                          dtype=tf.float32)
            self.random_uint = tf.placeholder(shape=[None], dtype=tf.float32)

            # the network outputs
            self.weights = self.network.get_weights()
            self.gen_was = self.network.wasserstein_network(self.weights, self.gen_im)
            self.data_was = self.network.wasserstein_network(self.weights, self.true_im)

            # Wasserstein loss
            self.wasserstein_loss = tf.reduce_mean(self.data_was - self.gen_was)
            # gradients for tracking
            self.g1 = tf.reduce_mean(tf.square(tf.gradients(self.wasserstein_loss, self.weights[0])[0]))

            # intermediate point
            random_uint_exp = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.random_uint, axis=1), axis=1), axis=1)
            self.inter = tf.multiply(self.gen_im, random_uint_exp) + \
                         tf.multiply(self.true_im, 1 - random_uint_exp)
            self.inter_was = self.network.wasserstein_network(self.weights, self.inter)

            # calculate derivative at intermediate point
            self.gradient_was = tf.gradients(self.inter_was, self.inter)[0]

            # take the L2 norm of that derivative
            self.regulariser_was = tf.reduce_mean(tf.square(tf.nn.relu(tf.sqrt(
                tf.reduce_sum(tf.square(self.gradient_was), axis=(1, 2, 3))) - 1)))
            # gradients for tracking
            self.g2 = tf.reduce_mean(tf.square(tf.gradients(self.regulariser_was, self.weights[0])[0]))

            # Overall Net Training loss
            self.loss_was = self.wasserstein_loss + self.lmb * self.regulariser_was

            # optimizer for Wasserstein network
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss_was,
                                                                                    global_step=self.global_step)

            ### The reconstruction network
            self.reconstruction = tf.placeholder(shape=[None, self.image_size[0], self.image_size[1], 3], dtype=tf.float32)
            self.noisy_image = tf.placeholder(shape=[None, self.image_size[0], self.image_size[1], 3], dtype=tf.float32)
            self.mu = tf.placeholder(dtype=tf.float32)

            # data loss
            data_mismatch = tf.square(self.reconstruction - self.noisy_image)
            self.data_error = tf.reduce_mean(tf.reduce_sum(data_mismatch, axis=(1, 2, 3)))

            # the loss functional
            self.was_output = tf.reduce_mean(self.network.wasserstein_network(self.weights, self.reconstruction))
            self.full_error = self.mu * self.was_output + self.data_error

            # get the batch size - all gradients have to be scaled by the batch size as they are taken over previously
            # averaged quantities already
            batch_s = tf.cast(tf.shape(self.reconstruction)[0], tf.float32)

            # Optimization for the picture
            self.pic_grad = tf.gradients(self.full_error * batch_s, self.reconstruction)

            # Gradients for logging
            self.g3 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.gradients(self.was_output * batch_s,
                                                                                  self.reconstruction)[0]),
                                                           axis=(1, 2, 3))))
            self.g4 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.gradients(self.data_error * batch_s,
                                                                                  self.reconstruction)[0]),
                                                           axis=(1, 2, 3))))

            # Measure quality of reconstruction
            self.ground_truth = tf.placeholder(shape=[None, self.image_size[0], self.image_size[1], 3], dtype=tf.float32)
            self.quality = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(self.ground_truth - self.reconstruction),
                                                                axis=(1, 2, 3))))

            # logging tools
            with tf.name_scope('Network_Optimization'):
                tf.summary.scalar('Wasserstein_Loss', self.wasserstein_loss)
                tf.summary.scalar('Wasserstein_Loss_grad', self.g1)
                tf.summary.scalar('Regulariser_Wasser', self.regulariser_was)
                tf.summary.scalar('Regulariser_Wasser_grad', self.g2)
                tf.summary.scalar('Overall_Net_Loss', self.loss_was)
            with tf.name_scope('Picture_Optimization'):
                data_loss = tf.summary.scalar('Data_Loss', self.data_error)
                data_loss_grad = tf.summary.scalar('Data_Loss_grad', self.g4)
                wasser_loss = tf.summary.scalar('Wasserstein_Loss', self.was_output)
                wasser_loss_grad = tf.summary.scalar('Wasserstein_Loss_grad', self.g3)
            with tf.name_scope('Model_L2_strength'):
                quality_assesment = tf.summary.scalar('L2', self.quality)

            # set up the logger
            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(self.logging_path, self.sess.graph)
            # set up the logger for image optimization
            self.merged_pic = tf.summary.merge(
                [data_loss, data_loss_grad, wasser_loss, wasser_loss_grad, quality_assesment])

        # set up variables saver
        self.saver = tf.train.Saver(self.weights)

    # visualization methode
    def visualize(self, true, noisy, recon, global_step, step, mu):
        quality = np.average(np.sqrt(np.sum(np.square(true - recon), axis=(1, 2, 3))))
        print('Quality of reconstructed image: ' + str(quality))
        ut.create_single_folder('Saves/Pictures/' + self.model_name + '/' +str(global_step))
        plt.figure()
        plt.subplot(131)
        plt.imshow(ut.cut_image(true[-1, ...]))
        plt.axis('off')
        plt.title('Original')
        plt.subplot(132)
        plt.imshow(ut.cut_image(noisy[-1,...]))
        plt.axis('off')
        plt.title('Corrupted')
        plt.suptitle('L2 :' + str(quality))
        plt.subplot(133)
        plt.imshow(ut.cut_image(recon[-1,...]))
        plt.title('Reconstruction')
        plt.axis('off')
        plt.savefig('Saves/Pictures/' + self.model_name + '/' +str(global_step) +'/mu-' + str(mu)+
                    'iteration-' + str(step) + '.png')
        plt.close()

    # picture update
    def update_pic(self, steps, stepsize, cor, guess, mu):
        for k in range(steps):
            gradient = self.sess.run(self.pic_grad, feed_dict={self.reconstruction: guess,
                                                                self.noisy_image: cor,
                                                               self.mu: mu})
            guess = guess - stepsize * gradient[0]
        return guess

    # evaluates and prints the network performance
    def evaluate_Network(self, true, cor):
        # generate random distribution for rays
        epsilon = np.random.uniform(size=(self.batch_size))
        step, Was_g, reg_g, Was, reg = self.sess.run([self.global_step, self.g1, self.g2,
                                                      self.wasserstein_loss, self.regulariser_was],
                                                     feed_dict={self.gen_im: cor, self.true_im: true,
                                                                self.random_uint: epsilon})
        print('Iteration: ' + str(step) + ', Was: ' + str(Was) + ', Reg: ' + str(reg) +
              ', Was grad: ' + str(Was_g) + ', Reg grad: ' + str(reg_g))

        # tensorflow logging
        guess = self.update_pic(15, self.step_size, cor, cor, self.mu_default)
        summary, step = self.sess.run([self.merged, self.global_step],
                                      feed_dict={self.gen_im: cor,
                                                 self.true_im: true,
                                                 self.random_uint: epsilon,
                                                 self.reconstruction: guess,
                                                 self.noisy_image: cor,
                                                 self.ground_truth: true,
                                                 self.mu: self.mu_default})
        self.writer.add_summary(summary, step)

    def picture_quality(self, true, cor):
        print('Starting quality: {}'.format(ut.l2_norm(true-cor)))
        guess = np.copy(cor)
        guess = self.net_output(guess, cor)
        print('End quality: {}'.format(ut.l2_norm(true - guess)))


    # train network
    def train(self, true, cor):
        # generate random distribution for rays
        epsilon = np.random.uniform(size=(self.batch_size))
        # optimize network
        self.sess.run(self.optimizer,
                      feed_dict={self.gen_im: cor, self.true_im: true, self.random_uint: epsilon})

    # update input
    def net_output(self, guess, cor):
        return self.update_pic(10, self.step_size, guess, cor, self.mu_default)




class stacked_denoiser(ar.Data_pip):
    model_name = 'Stacked_Denoiser'
    # The batch size
    batch_size = 64
    # relation between L2 error and regulariser
    mu = 13
    # weight on gradient norm regulariser for wasserstein network
    lmb = 20
    # learning rate for Adams
    learning_rate = 0.0003
    # step size for picture optimization
    step_size = 0.1

    # returns the used network architecture class. Can be overwritten in subclasses to change architecture
    def get_net(self):
        return convnet1()

    # sets up and loads the stacked denoiser architecture
    def __init__(self, amount_stacks):
        # save stack number
        self.amount_stacks = amount_stacks

        # call init of superclass to set up data pipeline
        super(stacked_denoiser, self).__init__()

        # the net object prescribes the design of the regularisation network
        net = self.get_net()

        # the array stack saves the different regularisers
        self.stacks = []

        # start a tensorflow session
        self.sess = tf.InteractiveSession()
        for k in range(amount_stacks):
            stack_name = 'Stack_' + str(k)
            current_stack = single_stack(net, self.sess, self.image_size, self.batch_size, self.mu, self.lmb,
                                         self.learning_rate, self.step_size, self.model_name, stack_name)
            self.stacks.append(current_stack)

        # initialize Variables
        tf.global_variables_initializer().run()

        # load latest saves
        for stack in self.stacks:
            stack.load()

    # optimizes the corrupted images with all stacked denoiser, stopping at finishing
    def optimize_until(self, cor, finishing):
        if finishing > self.amount_stacks:
            finishing = self.amount_stacks
        guess = np.copy(cor)
        if finishing>0:
            for k in range(finishing):
                guess = self.stacks[k].net_output(guess, cor)
        return guess

    # generates optimized images using the first 'stack_number' stacked denoisers.
    def generate_training_data(self, stack_number, training_data = True):
        true, cor = self.generate_images(64, training_data=training_data)
        guess = self.optimize_until(cor, stack_number)
        return true, guess

    # trains k-th layer
    def train_layer(self, stack_number, iterations):
        for k in range(iterations):
            true, guess = self.generate_training_data(stack_number)
            self.stacks[stack_number].train(true, guess)
            if k%25 == 0:
                true, guess = self.generate_training_data(stack_number, training_data=False)
                self.stacks[stack_number].evaluate_Network(true, guess)
                self.stacks[stack_number].picture_quality(true, guess)
        self.stacks[stack_number].save()


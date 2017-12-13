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



def lrelu(x):
    return (tf.nn.relu(x) - 0.1*tf.nn.relu(-x))

class Data_pip(object):
    model_name = 'default'
    image_size = (128,128)
    # setup methods

    # creates list of training data
    @staticmethod
    def find(pattern, path):
        result = []
        for root, dirs, files in os.walk(path):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    result.append(os.path.join(root, name).replace("\\", "/"))
        return result

    # check if folder structure is in place and creates folders if necessary
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

    # methode to create a single folder
    def create_single_folder(self, folder):
        if not os.path.exists(folder):
            try:
                os.makedirs(folder)
            except OSError:
                pass

    # methode to draw raw picture samples
    def load_data(self, training_data=True):
        if training_data:
            rand = random.randint(0, self.train_amount - 1)
            pic = mpimg.imread(self.train_list[rand])
        else:
            rand = random.randint(0, self.eval_amount - 1)
            pic = scipy.ndimage.imread(self.eval_list[rand])
        return pic/255.0

    # methode to cut image to [0,1] value range
    @staticmethod
    def cut_image(pic):
        pic = np.maximum(pic, 0.0)
        pic = np.minimum(pic, 1.0)
        return pic

    # visualize a single image
    @staticmethod
    def visualize_single_pic(pic, number):
        plt.figure()
        plt.imshow(Data_pip.cut_image(pic))
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
            noise = np.random.normal(0, 0.1 ,(self.image_size[0], self.image_size[1],3))
            image_cor  = image + noise
            true[i,...] = image
            cor[i,...] = image_cor
        return true, cor

    def __init__(self):
        self.create_folders()
        # set up the training data file system
        self.train_list = self.find('*.jpg', './BSDS/images/train')
        self.train_amount = len(self.train_list)
        print('Training Pictures found: ' + str(self.train_amount))
        self.eval_list = self.find('*.jpg', './BSDS/images/val')
        self.eval_amount = len(self.eval_list)
        print('Evaluation Pictures found: ' + str(self.eval_amount))

class denoiser(Data_pip):
    model_name = 'default'
    # The batch size
    batch_size = 64
    # relation between L2 error and regulariser
    mu_default = 50
    # weight on gradient norm regulariser for wasserstein network
    lmb = 20
    # learning rate for Adams
    learning_rate = 0.0003
    # step size for picture optimization
    step_size = 0.1

    def get_weights(self):
        return []

    def wasserstein_network(self, weights, input_pic):
        return input_pic

    def save(self):
        self.saver.save(self.sess, 'Saves/Data/' + self.model_name + '/model', global_step=self.global_step)
        print('Progress saved')

    def load(self):
        if os.listdir('Saves/Data/' + self.model_name + '/'):
            self.saver.restore(self.sess, tf.train.latest_checkpoint(os.path.join('Saves', 'Data', self.model_name, '')))
            print('Save restored')
        else:
            print('No save found')

    def __init__(self):
        super(denoiser, self).__init__()

        # start a tensorflow session
        self.sess = tf.InteractiveSession()

        # placeholders for NN
        self.gen_im = tf.placeholder(shape=[None, self.image_size[0], self.image_size[1], 3],
                                     dtype=tf.float32, name='gen_im')
        self.true_im = tf.placeholder(shape=[None, self.image_size[0], self.image_size[1], 3],
                                      dtype=tf.float32)
        self.random_uint = tf.placeholder(shape=[None], dtype=tf.float32)

        # the network outputs
        self.weights = self.get_weights()
        self.gen_was = self.wasserstein_network(self.weights, self.gen_im)
        self.data_was = self.wasserstein_network(self.weights, self.true_im)

        # Wasserstein loss
        self.wasserstein_loss = tf.reduce_mean(self.data_was - self.gen_was)
        # gradients for tracking
        self.g1 = tf.reduce_mean(tf.square(tf.gradients(self.wasserstein_loss, self.weights[0])[0]))

        # intermediate point
        random_uint_exp = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.random_uint, axis=1), axis=1), axis=1)
        self.inter = tf.multiply(self.gen_im, random_uint_exp) + \
                     tf.multiply(self.true_im, 1 - random_uint_exp)
        self.inter_was = self.wasserstein_network(self.weights, self.inter)

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
        self.data_error = tf.reduce_mean(tf.reduce_sum(data_mismatch, axis=(1,2,3)))

        # the loss functional
        self.was_output = tf.reduce_mean(self.wasserstein_network(self.weights, self.reconstruction))
        self.full_error = self.mu * self.was_output + self.data_error

        # get the batch size - all gradients have to be scaled by the batch size as they are taken over previously
        # averaged quantities already
        batch_s = tf.cast(tf.shape(self.reconstruction)[0],tf.float32)

        # Optimization for the picture
        self.pic_grad = tf.gradients(self.full_error*batch_s, self.reconstruction)

        # Gradients for logging
        self.g3 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.gradients(self.was_output*batch_s,
                                                                              self.reconstruction)[0]), axis=(1, 2, 3))))
        self.g4 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.gradients(self.data_error*batch_s,
                                                                              self.reconstruction)[0]), axis=(1, 2, 3))))


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
        self.writer = tf.summary.FileWriter('Saves/Logs/' + self.model_name + '/Network_Training/',
                                            self.sess.graph)
        # set up the logger for image optimization
        self.merged_pic = tf.summary.merge([data_loss, data_loss_grad, wasser_loss, wasser_loss_grad, quality_assesment])

        # set up variables saver
        self.saver = tf.train.Saver()

        # initialize Variables
        tf.global_variables_initializer().run()

        # load existing saves
        self.load()

    # Method to estimate a good value of the regularisation paramete.
    # This is done via estimation of 2 ||K^t (Kx-y)||_2 where x is the ground truth
    def find_good_lambda(self, sample = 256):
        ### for consistency, compute optimal lambda with graph as well
        true, cor = self.generate_images(sample)
        gradient_truth = self.sess.run(self.pic_grad, {self.reconstruction: true,
                                                 self.noisy_image: cor,
                                                 self.ground_truth: true,
                                                 self.mu: 0})
        print(np.sqrt(np.sum(np.square(gradient_truth[0]), axis=(1,2,3))))
        print(np.mean(np.sqrt(np.sum(np.square(gradient_truth[0]), axis=(1,2,3)))))


    # Method to determine the L2 noise level ||Kx-y|| where x is the ground truth
    def find_noise_level(self, sample= 256 ):
        true, cor = self.generate_images(sample)
        data_error = self.sess.run(self.data_error, {self.reconstruction: true,
                                                 self.noisy_image: cor,
                                                 self.ground_truth: true,
                                                 self.mu: 0})
        print(data_error)

    # visualization methode
    def visualize(self, true, fbp, recon, global_step, step, mu):
        quality = np.average(np.sqrt(np.sum(np.square(true - recon), axis=(1, 2, 3))))
        print('Quality of reconstructed image: ' + str(quality))
        self.create_single_folder('Saves/Pictures/' + self.model_name + '/' +str(global_step))
        plt.figure()
        plt.subplot(131)
        plt.imshow(true[-1, ...])
        plt.axis('off')
        plt.title('Original')
        plt.subplot(132)
        plt.imshow(fbp[-1,...])
        plt.axis('off')
        plt.title('Corrupted')
        plt.suptitle('L2 :' + str(quality))
        plt.subplot(133)
        plt.imshow(recon[-1,...])
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

    # visualization of Picture optimization
    def create_optimized_images(self, batch_size, steps = 30, step_s = step_size,
                                mu = mu_default):
        true, cor = self.generate_images(batch_size)
        guess = np.copy(cor)
        step = self.sess.run(self.global_step)
        writer = tf.summary.FileWriter('Saves/Logs/' + self.model_name + '/Picture_Opt/Iteration_' +
                                       str(step) + '/' + str(mu) + '/')
        for k in range(steps):
            summary = self.sess.run(self.merged_pic,
                                      feed_dict={self.reconstruction: guess,
                                                 self.noisy_image: cor,
                                                 self.ground_truth: true,
                                                 self.mu: mu})
            writer.add_summary(summary, k)
            if (k%5 == 0):
                self.visualize(true, cor, guess, step, k, mu)
            guess = self.update_pic(1, step_s, cor, guess, mu)
        writer.close()


    # evaluates and prints the network performance
    def evaluate_Network(self, mu):
        true, cor = self.generate_images(64)
        # generate random distribution for rays
        epsilon = np.random.uniform(size=(self.batch_size))
        step, Was_g, reg_g, Was, reg = self.sess.run([self.global_step, self.g1, self.g2,
                                                      self.wasserstein_loss, self.regulariser_was],
                                                     feed_dict={self.gen_im: cor, self.true_im: true,
                                                                self.random_uint: epsilon})
        print('Iteration: ' + str(step) + ', Was: ' + str(Was) + ', Reg: ' + str(reg) +
              ', Was grad: ' + str(Was_g) + ', Reg grad: ' + str(reg_g))

        # tensorflow logging
        true, cor = self.generate_images(64)
        guess = self.update_pic(30, self.step_size, cor, cor, mu)
        summary, step = self.sess.run([self.merged, self.global_step],
                                      feed_dict={self.gen_im: cor,
                                                 self.true_im: true,
                                                 self.random_uint: epsilon,
                                                 self.reconstruction: guess,
                                                 self.noisy_image: cor,
                                                 self.ground_truth: true,
                                                 self.mu: mu})
        self.writer.add_summary(summary, step)

    def generate_training_images(self, batch_size, amount_steps, mu = mu_default):
        true_im = np.zeros(shape=(batch_size, self.image_size[0],self.image_size[1], 3))
        output_im = np.zeros(shape=(batch_size, self.image_size[0], self.image_size[1], 3))
        output_cor = np.zeros(shape=(batch_size, self.image_size[0], self.image_size[1], 3))

        #create remaining samples
        for j in range(batch_size):
            true, cor = self.generate_images(1)
            guess = np.copy(cor)
            s = random.randint(1,amount_steps)
            guess = self.update_pic(s, self.step_size, true, guess, mu)
            true_im[j,...] = true[0,...]
            output_cor[j, ...] = cor[0,...]
            output_im[j, ...] = guess[0,...]
        return true_im, output_cor, output_im

    # control methode to check that generate_training_images works as it should
    def control(self, steps):
        true, fbp, recon = self.generate_training_images(4,steps)
        for k in range(4):
            plt.figure()
            plt.subplot(131)
            plt.imshow(true[k, ..., 0])
            plt.axis('off')
            plt.title('Original')
            plt.subplot(132)
            plt.imshow(fbp[k, ..., 0])
            plt.axis('off')
            plt.title('Corrupted')
            plt.subplot(133)
            plt.imshow(recon[k, ..., 0])
            plt.axis('off')
            plt.title('Reconstruction')
            plt.savefig('Saves/Pictures/' + self.model_name + '/test/' + str(k) + '.jpg')
            plt.close()

    # optimize network on initial guess input only, with initial guess being fbp
    def pretrain_Wasser_ini(self, steps, mu = mu_default):
        for k in range(steps):
            if k%20 == 0:
                self.evaluate_Network(mu)
            if k%100 == 0:
                self.create_optimized_images(64)
            true, cor = self.generate_images(self.batch_size)
            # generate random distribution for rays
            epsilon = np.random.uniform(size=(self.batch_size))
            # optimize network
            self.sess.run(self.optimizer,
                          feed_dict={self.gen_im: cor, self.true_im: true, self.random_uint: epsilon})
        self.save()


    # iterative training methode, using actual output distribtion instead of initial guess distribution
    def train(self, steps, amount_steps, mu = mu_default):
        for k in range(steps):
            if k % 20 == 0:
                self.evaluate_Network(mu)
            if k % 100 == 0:
                self.create_optimized_images(512)
            true, cor, gen = self.generate_training_images(self.batch_size, amount_steps=amount_steps,
                                                           mu = mu)
            # generate random distribution for rays
            epsilon = np.random.uniform(size=(self.batch_size))
            # optimize network
            self.sess.run(self.optimizer,
                          feed_dict={self.gen_im: gen, self.true_im: true, self.random_uint: epsilon})
        self.save()

class Denoiser1(denoiser):
    model_name = 'Denoiser1'

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
        conv_pre = lrelu(tf.nn.conv2d(input_pic, weights[10], strides=[1, 1, 1, 1], padding='SAME') + weights[11])

        # 1st convolutional layer (pic size 128)
        conv1 = lrelu(tf.nn.conv2d(conv_pre, weights[0], strides=[1, 2, 2, 1], padding='SAME') + weights[1])

        # 2nd conv layer (pic size 64)
        conv2 = lrelu(tf.nn.conv2d(conv1, weights[2], strides=[1, 2, 2, 1], padding='SAME') + weights[3])

        # 3rd conv layer (pic size 32)
        conv3 = lrelu(tf.nn.conv2d(conv2, weights[4], strides=[1, 2, 2, 1], padding='SAME') + weights[5])

        # 4th conv layer (pic size 16)
        conv4 = lrelu(tf.nn.conv2d(conv3, weights[6], strides=[1, 2, 2, 1], padding='SAME') + weights[7])

        # reshape (pic size 8)
        p2resh = tf.reshape(conv4, [-1, 128 * 8 * 8])

        # dropout layer
        # drop = tf.layers.dropout(p2resh, 0.3, training=True)

        # logits
        output = tf.matmul(p2resh, weights[8]) + weights[9]
        return output


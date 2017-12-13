import tensorflow as tf
import os, fnmatch
import random
import numpy as np
import scipy.ndimage
import math
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Data_pip(object):
    model_name = None
    # setup methods
    # takes float picture and cuts and transforms it to uint8
    @staticmethod
    def to_uint(pic):
        pic = pic * 255
        pic = np.maximum(pic, 0)
        pic = np.minimum(pic, 255)
        pic = np.floor(pic)
        return pic.astype(np.uint8)

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

    # internal methode for batch generation
    def generate_local_input(self, batch_size, training_data=True):
        width = 5
        length = 25

        w_d = int((width - 1) / 2)
        res_d = width - 2 * w_d
        l_d = int((length - 1) / 2)
        res_l = length - 2 * l_d
        ul, lr = self.edgepoint(40, 40)
        filter = np.ones(shape=[40, 40])
        filter[21 - l_d:21 + l_d + res_l, 21 - w_d:21 + w_d + res_d] = 0
        batch = self.load_data(batch_size, training_data)
        corrupted_image = np.empty(shape=[batch_size, 40, 40, self.image_size[2]])
        true_im = np.empty(shape=[batch_size, 40, 40, self.image_size[2]])
        fil = np.empty(shape=[batch_size, 40, 40, 1])

        #random_filling = np.random.randint(0, 255, (40, 40, 3))/255
        # fill the region with whatever was next to it.
        filling= np.zeros(shape=(batch_size, length,width,3))


        for i in range(batch_size):
            true_im[i, ...] = batch[i, ul[0]:lr[0], ul[1]:lr[1], : ]

            filling[i, ...] = true_im[i,21 - l_d:21 + l_d + res_l,21 - w_d - width-1:21 - w_d -1,:]

            fil[i, :, :, 0] = filter
            for j in range(self.image_size[2]):
                corrupted_image[i, :, :, j] = np.multiply(true_im[i, :, :, j], filter)
                #corrupted_image[i,21 - l_d:21 + l_d + res_l, 21 - w_d:21 + w_d + res_d,j] = filling[i,...,j]
        return true_im, corrupted_image, fil

    # methode to draw raw picture samples
    def load_data(self, batch_size, training_data=True):
        batch = np.empty(shape=[batch_size, self.image_size[0], self.image_size[1], self.image_size[2]])
        for i in range(batch_size):
            if training_data:
                rand = random.randint(0, self.train_amount - 1)
                pic = mpimg.imread(self.train_list[rand])
                # pic = scipy.ndimage.imread(self.train_list[rand])
            else:
                rand = random.randint(0, self.eval_amount - 1)
                pic = scipy.ndimage.imread(self.eval_list[rand])
            batch[i, ...] = pic/255.0
        return batch

    # define input with cut out areas
    def create_filter(self, upper_left, lower_right):
        filter = np.ones(shape=[self.image_size[0], self.image_size[1]])
        filter[upper_left[0]:(lower_right[0]), upper_left[1]:(lower_right[1])] = 0
        return filter

    # Draw random edgepoint
    def edgepoint(self, x_size, y_size):
        x_vary = self.image_size[0] - x_size
        x_coor = random.randint(0, x_vary)
        y_vary = self.image_size[1] - y_size
        y_coor = random.randint(0, y_vary)
        upper_left = [x_coor, y_coor]
        lower_right = [x_coor + x_size, y_coor + y_size]
        return upper_left, lower_right

    def __init__(self):
        # set up the training data file system
        self.train_list = self.find('*.jpg', './Train_Data')
        self.train_amount = len(self.train_list)
        print('Training Pictures found: ' + str(self.train_amount))
        self.eval_list = self.find('*.jpg', './Eval_Data')
        self.eval_amount = len(self.eval_list)
        print('Evaluation Pictures found: ' + str(self.eval_amount))
        self.image_size = (scipy.ndimage.imread(self.train_list[0])).shape


class Inpainter(object):
    model_name = 'default'
    # The batch size
    batch_size = 32
    # where the logarithms are cut to prevent problems with machine precession
    log_cut = 0.0
    adv_log_cut = 0.0
    # a value of L2CE of 1 corresponds to L2 Loss only, 0 corresponds to adv loss only
    L2CE = 0.3
    # scaling of loss for image optimization
    loss_scaling = 1
    # step size Image Optimization
    step_size = 0.5
    # Learning Rate for Network Optimization
    learning_rate_n = 0.0007
    # weight penalization factor of L2 square of network parameters
    L2_reg = 1

# setup methods
    # takes float picture and cuts and transforms it to uint8
    @staticmethod
    def to_uint(pic):
        pic = pic*255
        pic = np.maximum(pic, 0)
        pic = np.minimum(pic, 255)
        pic = np.floor(pic)
        return pic.astype(np.uint8)

    # creates list of training data
    @staticmethod
    def find(pattern, path):
        result = []
        for root, dirs, files in os.walk(path):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    result.append(os.path.join(root, name).replace("\\","/"))
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

    def get_weights(self):
        return []

    def adverserial_network(self, weights, input_pic, input_filter):
        return input_pic

    def __init__(self, finish_setup = True):
        # set up necessary folders
        self.create_folders()

        # set up the training data file system
        self.train_list = self.find('*.jpg', './Train_Data')
        self.train_amount = len(self.train_list)
        print('Training Pictures found: ' + str(self.train_amount))
        self.eval_list = self.find('*.jpg', './Eval_Data')
        self.eval_amount = len(self.eval_list)
        print('Evaluation Pictures found: ' + str(self.eval_amount))
        self.image_size = (scipy.ndimage.imread(self.train_list[0])).shape

        # start a tensorflow session
        self.sess = tf.InteractiveSession()

        # placeholders for NN
        self.gen_im = tf.placeholder(shape=[None, None, None, self.image_size[2]],
                                dtype=tf.float32, name='gen_im')
        self.filter = tf.placeholder(shape=[self.batch_size, 40, 40, 1], dtype=tf.float32, name='filter')
        self.true_im = tf.placeholder(shape=[None, None, None, self.image_size[2]],
                                 dtype=tf.float32)
        self.mu = tf.placeholder(tf.float32, shape=(), name='weightL2_CE')

        # the forward network
        self.weights = self.get_weights()
        self.gen_clas = self.adverserial_network(self.weights, self.gen_im, self.filter)
        self.true_clas = self.adverserial_network(self.weights, self.true_im, self.filter)

        # L2 regularisation on the network weights
        reg = [tf.reduce_sum(tf.square(w)) for w in self.weights]
        self.total_weight = tf.add_n(reg)

        # loss of adverserial Network during training given by misclassification loss of true data and network data
        self.loss = -tf.reduce_mean(tf.log(tf.maximum(self.true_clas, self.adv_log_cut)) +
                                    tf.log(tf.maximum(1. - self.gen_clas, self.adv_log_cut))) \
                    + self.L2_reg * self.total_weight



        # evaluation metric for classification
        self.acc = (tf.reduce_mean(tf.cast(tf.greater(0.5, self.gen_clas), tf.float32)) +
                   tf.reduce_mean(tf.cast(tf.greater(self.true_clas, 0.5), tf.float32))) / 2

        # gradients for tracking
        grad_training = tf.gradients(self.loss, self.weights[0])
        grad_reg = tf.gradients(self.total_weight, self.weights[0])
        self.grad_norm = tf.reduce_mean(tf.reduce_sum(tf.square(grad_training[0]), axis=(1,2,3)))
        self.weights_norm = tf.reduce_mean(tf.reduce_sum(tf.square(self.weights[0]), axis=(1,2,3)))
        self.grad_reg_norm = self.L2_reg * tf.reduce_mean(tf.reduce_sum(tf.square(grad_reg[0]), axis=(1, 2, 3)))


        # optimizer for adverserial network
        self.learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        # The reconstruction network
        self.inpainted_im = tf.placeholder(shape=[None, 40, 40, self.image_size[2]], dtype=tf.float32)
        self.recon_filter = tf.placeholder(shape=[None, None, None, 1], dtype=tf.float32)
        self.true_image = tf.placeholder(shape=[None, None, None, self.image_size[2]],
                                 dtype=tf.float32)

        # the loss functional
        self.adv_output = self.adverserial_network(self.weights, self.inpainted_im, self.recon_filter)
        self.adv_loss = self.loss_scaling*(tf.reduce_mean(- tf.log(tf.maximum(self.adv_output, self.log_cut))))
        mismatch = tf.square(tf.multiply(self.recon_filter, (self.inpainted_im- self.true_image)))
        self.data_error = self.loss_scaling*(tf.reduce_mean(tf.reduce_sum(mismatch, axis=(1,2,3))))
        self.full_error = (1-self.mu) * self.adv_loss + self.mu * self.data_error

        # Gradients for logging
        self.adv_grad = tf.gradients(self.adv_loss, self.inpainted_im)
        self.adv_grad_N = tf.reduce_mean(tf.reduce_sum(tf.square(self.adv_grad[0]), axis=(1,2,3)))
        self.L2_grad = tf.gradients(self.data_error, self.inpainted_im)
        self.L2_grad_N = tf.reduce_mean(tf.reduce_sum(tf.square(self.L2_grad[0]), axis=(1,2,3)))

        # Optimization for the picture
        self.pic_grad = tf.gradients(self.full_error, self.inpainted_im)

        # logging tools
        with tf.name_scope('Network_Optimization'):
            tf.summary.scalar('CE', self.loss)
            tf.summary.scalar('Acc', self.acc)
            tf.summary.scalar('Grad_norm', self.grad_norm)
            tf.summary.scalar('Grad_regulariser', self.grad_reg_norm)
        with tf.name_scope('Picture_Optimization'):
            tf.summary.scalar('CE_pic', self.adv_loss)
            tf.summary.scalar('L2_Error_pic', self.data_error)
            tf.summary.scalar('Grad_norm_adv', self.adv_grad_N)
            tf.summary.scalar('Out_Adv_First_Pic', self.adv_output[0,0])

        # set up the logger
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('Saves/Logs/' + self.model_name + '/',
                                             self.sess.graph)

        # set up variables saver
        self.saver = tf.train.Saver()

        # initialize Variables
        tf.global_variables_initializer().run()

        # load existing saves
        self.load()

        # variables for training
        self.last_added = 0
        true, inp, fil = self.generate_local_input(self.batch_size)
        self.true = true
        self.inp = inp
        self.fil = fil


    def train(self, training_steps):
        # optimize the initial pictures
        self.optimize_pics(50)
        for k in range(training_steps):
            # optimize the pictures
            self.optimize_pics(3)
            # update the weights
            self.optimize_network(0.9, 300)
            #go to next batch
            #self.update_batch(50)
            # find out current step
            step = self.sess.run(self.global_step)
            # log the progress
            summary, step = self.sess.run([self.merged, self.global_step],
                                          feed_dict={self.gen_im: self.inp,
                                                     self.true_im: self.true,
                                                     self.filter: self.fil,
                                                     self.learning_rate: self.learning_rate_n,
                                                     self.inpainted_im: self.inp,
                                                     self.recon_filter: self.fil,
                                                     self.true_image: self.true,
                                                     self.mu: self.L2CE
                                                     })
            self.writer.add_summary(summary, step)
            if k%50 == 0:
                self.visualize()
        self.save()


    # optimize network until a predefined CE level is reached
    def optimize_network(self, CE, max_iter, random_pics = False):
        h = CE + 1
        k = 0
        while ((h > CE) and (k<max_iter)):
            if random_pics:
                # get random input
                true_pics, _, _ = self.generate_local_input(self.batch_size)
                # optimize discriminator weights
                self.sess.run(self.optimizer, feed_dict={self.gen_im: self.inp,
                                                         self.true_im: true_pics,
                                                         self.filter: self.fil,
                                                         self.learning_rate: self.learning_rate_n})
                true_pics, _, _ = self.generate_local_input(256)
                h = self.sess.run(self.loss, feed_dict={self.gen_im: self.inp,
                                                         self.true_im: true_pics,
                                                         self.filter: self.fil,
                                                         self.learning_rate: self.learning_rate_n})
            else:
                # optimize discriminator weights
                self.sess.run(self.optimizer, feed_dict={self.gen_im: self.inp,
                                                         self.true_im: self.true,
                                                         self.filter: self.fil,
                                                         self.learning_rate: self.learning_rate_n})
                h = self.sess.run(self.loss, feed_dict={self.gen_im: self.inp,
                                                         self.true_im: self.true,
                                                         self.filter: self.fil,
                                                         self.learning_rate: self.learning_rate_n})
            k = k + 1

        # evaluate model performance
        step, loss, acc, true, gen, grad_N, weights_N, grad_reg = self.sess.run([self.global_step,
                                                                       self.loss,
                                                                       self.acc,
                                                                       self.true_clas,
                                                                       self.gen_clas,
                                                                       self.grad_norm,
                                                                       self.weights_norm,
                                                                       self.grad_reg_norm],
                                                              feed_dict={self.gen_im: self.inp,
                                                                         self.true_im: self.true,
                                                                         self.filter: self.fil,
                                                                         self.learning_rate: self.learning_rate_n})
        print('Iteration: ' + str(step) + ', CE: ' + str(loss) + ', Accuracy: ' + str(acc)
              + ', network true: ' + str(true[0, 0]) + ', network gen: ' + str(gen[0, 0]) +
              ', Norm weights: ' + str(weights_N) + ', Grad Norm: ' + str(grad_N) + ', Grad Reg: ' + str(grad_reg))

    # update pictures
    def optimize_pics(self, steps):
        # the optimization
        for k in range(steps):
            gradient = self.sess.run([self.pic_grad], feed_dict={self.inpainted_im: self.inp,
                                                                 self.recon_filter: self.fil,
                                                                 self.true_image: self.true,
                                                                 self.mu: self.L2CE})
            self.inp = self.inp - self.step_size * gradient[0][0]

        # evaluation
        l2, adv, all, adv_grad, l2_grad, out = self.sess.run([self.data_error,self.adv_loss,
                                                              self.full_error, self.adv_grad_N,
                                                              self.L2_grad_N, self.adv_output],
                                                                  feed_dict={self.inpainted_im: self.inp,
                                                                             self.recon_filter: self.fil,
                                                                             self.true_image: self.true,
                                                                             self.mu: self.L2CE})

        print('Optimizing Picture. L2Loss: ' + str(l2) + ', CE: ' + str(adv) + ', Overall Loss: ' +
              str(all) + ', Mu: ' + str(self.L2CE) + ', Grad Adv: ' + str(adv_grad) +
              ', Grad L2: ' + str(l2_grad) + ', Adv Out: ' + str(out[0, 0]))

    # update batch
    def update_batch(self, steps):
        n_true, n_cor, n_fil = self.generate_local_input(1)
        for k in range(steps):
            gradient = self.sess.run([self.pic_grad], feed_dict={self.inpainted_im: n_cor,
                                                                 self.recon_filter: n_fil,
                                                                 self.true_image: n_true,
                                                                 self.mu: self.L2CE})
            n_cor = n_cor - self.step_size * gradient[0][0]
        # evaluate new picture
        l2, adv, all, adv_grad, l2_grad = self.sess.run([self.data_error,self.adv_loss,
                                                              self.full_error, self.adv_grad_N,
                                                              self.L2_grad_N],
                                                                  feed_dict={self.inpainted_im: n_cor,
                                                                 self.recon_filter: n_fil,
                                                                 self.true_image: n_true,
                                                                 self.mu: self.L2CE})

        print('New Picture. L2Loss: ' + str(l2) + ', CE: ' + str(adv) + ', Overall Loss: ' +
              str(all) + ', Grad Adv: ' + str(adv_grad) +
              ', Grad L2: ' + str(l2_grad))
        # update training array
        self.true[self.last_added,...] = n_true[0,...]
        self.inp[self.last_added,...] = n_cor[0,...]
        self.fil[self.last_added,...] = n_fil[0,...]
        self.last_added = self.last_added + 1
        if self.last_added == self.batch_size:
            self.last_added = 0

    # internal methode for batch generation
    def generate_local_input(self, batch_size, training_data=True):

        width = 5
        length = 25

        w_d = int((width - 1) / 2)
        res_d = width - 2 * w_d
        l_d = int((length - 1) / 2)
        res_l = length - 2 * l_d
        ul, lr = self.edgepoint(40, 40)
        filter = np.ones(shape=[40, 40])
        filter[21 - l_d:21 + l_d + res_l, 21 - w_d:21 + w_d + res_d] = 0
        batch = self.load_data(training_data)
        corrupted_image = np.empty(shape=[batch_size, 40, 40, self.image_size[2]])
        true_im = np.empty(shape=[batch_size, 40, 40, self.image_size[2]])
        fil = np.empty(shape=[batch_size, 40, 40, 1])



        #random_filling = np.random.randint(0, 255, (40, 40, 3))/255
        # fill the region with whatever was next to it.
        filling= np.zeros(shape=(batch_size, length,width,3))


        for i in range(batch_size):
            true_im[i, ...] = batch[i, ul[0]:lr[0], ul[1]:lr[1], : ]

            filling[i, ...] = true_im[i,21 - l_d:21 + l_d + res_l,21 - w_d - width-1:21 - w_d -1,:]

            fil[i, :, :, 0] = filter
            for j in range(self.image_size[2]):
                corrupted_image[i, :, :, j] = np.multiply(true_im[i, :, :, j], filter)
                #corrupted_image[i,21 - l_d:21 + l_d + res_l, 21 - w_d:21 + w_d + res_d,j] = filling[i,...,j]
        return true_im, corrupted_image, fil

    # methode to draw raw picture samples
    def load_data(self, training_data=True):
        batch = np.empty(shape=[self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2]])
        for i in range(self.batch_size):
            if training_data:
                rand = random.randint(0, self.train_amount - 1)
                pic = mpimg.imread(self.train_list[rand])
                # pic = scipy.ndimage.imread(self.train_list[rand])
            else:
                rand = random.randint(0, self.eval_amount - 1)
                pic = scipy.ndimage.imread(self.eval_list[rand])
            batch[i, ...] = pic/255.0
        return batch

    # define input with cut out areas
    def create_filter(self, upper_left, lower_right):
        filter = np.ones(shape=[self.image_size[0], self.image_size[1]])
        filter[upper_left[0]:(lower_right[0]), upper_left[1]:(lower_right[1])] = 0
        return filter

    # Draw random edgepoint
    def edgepoint(self, x_size, y_size):
        x_vary = self.image_size[0] - x_size
        x_coor = random.randint(0, x_vary)
        y_vary = self.image_size[1] - y_size
        y_coor = random.randint(0, y_vary)
        upper_left = [x_coor, y_coor]
        lower_right = [x_coor + x_size, y_coor + y_size]
        return upper_left, lower_right


    # visualization methode
    def visualize(self):
        vis_index = self.last_added+1
        if vis_index == self.batch_size:
            vis_index = 0

        cor_im = np.zeros(shape=self.true.shape)
        for k in range(3):
            cor_im[vis_index,...,k] = np.multiply(self.true[vis_index,...,k], self.fil[vis_index,...,0])

        step = self.sess.run(self.global_step)
        plt.figure(step)
        plt.subplot(221)
        plt.imshow(Inpainter.to_uint(self.true[vis_index, ...]))
        plt.axis('off')
        plt.title('Original image')
        plt.subplot(222)
        plt.imshow(Inpainter.to_uint(cor_im[vis_index,...]))
        plt.axis('off')
        plt.title('Corrupted image')
        plt.subplot(223)
        plt.imshow(Inpainter.to_uint(self.inp[self.last_added,...]))
        plt.axis('off')
        plt.title('last added')
        plt.subplot(224)
        plt.imshow(Inpainter.to_uint(self.inp[vis_index,...]))
        plt.axis('off')
        plt.title('Reconstructed Image')
        plt.savefig('Saves/Pictures/' + self.model_name + '/iteration-' + str(step) + '.png')
        plt.close()

    def save(self):
        self.saver.save(self.sess, 'Saves/Data/' + self.model_name + '/model', global_step=self.global_step)
        print('Progress saved')

    def load(self):
        if os.listdir('Saves/Data/' + self.model_name + '/'):
            self.saver.restore(self.sess, tf.train.latest_checkpoint(os.path.join('Saves', 'Data', self.model_name, '')))
            print('Save restored')
        else:
            print('No save found')
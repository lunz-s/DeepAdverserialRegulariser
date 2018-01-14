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

class data_preprocessing(object):
    model_name = 'none'
    source = 'ellipses'

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
        paths['Visual Test Folder'] = 'Saves/Pictures/' + self.model_name + '/test'
        for key, value in paths.items():
            if not os.path.exists(value):
                try:
                    os.makedirs(value)
                except OSError:
                    pass
                print(key + ' created')

    def create_single_folder(self, folder):
        if not os.path.exists(folder):
            try:
                os.makedirs(folder)
            except OSError:
                pass


    def __init__(self):
        if self.source == 'LUNA':
            name = platform.node()
            Train_Path = ''
            Eval_Path = ''
            if name == 'LAPTOP-E6AJ1CPF':
                Train_Path = './LUNA/Train_Data'
                Eval_Path = './LUNA/Eval_Data'
            elif name == 'motel':
                Train_Path = '/local/scratch/public/sl767/LUNA/Training_Data'
                Eval_Path = '/local/scratch/public/sl767/LUNA/Evaluation_Data'
            # List the existing training data
            self.training_list = self.find('*.dcm', Train_Path)
            self.training_list_length = len(self.training_list)
            print('Training Data found: ' + str(self.training_list_length))
            self.eval_list = self.find('*.dcm', Eval_Path)
            self.eval_list_length = len(self.training_list)
            print('Evaluation Data found: ' + str(self.eval_list_length))

        # Create ODL data structures
        size = 128
        self.space = odl.uniform_discr([-64, -64], [64, 64], [size, size],
                                  dtype='float32')

        geometry = odl.tomo.parallel_beam_geometry(self.space, num_angles=30)
        op = odl.tomo.RayTransform(self.space, geometry)

        # Ensure operator has fixed operator norm for scale invariance
        opnorm = odl.power_method_opnorm(op)
        self.operator = (1 / opnorm) * op
        self.adjoint = (1 / opnorm) * op.adjoint
        self.fbp = (opnorm) * odl.tomo.fbp_op(op)

        # Create tensorflow layer from odl operator
        self.ray_transform = odl.contrib.tensorflow.as_tensorflow_layer(self.operator,
                                                                  'RayTransform')

        #self.ray_transform_adjoint = odl.contrib.tensorflow.as_tensorflow_layer(self.operator.adjoint,
        #                                                               'RayTransformAdjoint')

        # create needed folders
        self.create_folders()

    # returns simulated measurement, original pic and fbp
    def generate_data(self, batch_size, validation=False):
        """Generate a set of random data."""
        n_generate = 1 if (validation and self.source=='ellipses') else batch_size

        y = np.empty((n_generate, self.operator.range.shape[0], self.operator.range.shape[1], 1), dtype='float32')
        x_true = np.empty((n_generate, self.space.shape[0], self.space.shape[1], 1), dtype='float32')
        fbp = np.empty((n_generate, self.space.shape[0], self.space.shape[1], 1), dtype='float32')

        for i in range(n_generate):
            if validation:
                if self.source=='ellipses':
                    phantom = odl.phantom.shepp_logan(self.space, True)
                elif self.source == 'LUNA':
                    path = self.get_valid_path(validation=True)
                    phantom = self.space.element(self.get_pic(path))
            else:
                if self.source == 'LUNA':
                    path = self.get_valid_path(validation=False)
                    phantom = self.space.element(self.get_pic(path))
                else:
                    phantom = self.random_phantom(self.space)
            data = self.operator(phantom)
            noisy_data = data + odl.phantom.white_noise(self.operator.range) * np.mean(np.abs(data)) * 0.05

            fbp [i, ..., 0] = self.fbp(noisy_data)
            x_true[i, ..., 0] = phantom
            y[i, ..., 0] = noisy_data
        return y, x_true, fbp


    # generates one random ellipse
    def random_ellipse(self, interior=False):
        if interior:
            x_0 = np.random.rand() - 0.5
            y_0 = np.random.rand() - 0.5
        else:
            x_0 = 2 * np.random.rand() - 1.0
            y_0 = 2 * np.random.rand() - 1.0

        return ((np.random.rand() - 0.5) * np.random.exponential(0.4),
                np.random.exponential() * 0.2, np.random.exponential() * 0.2,
                x_0, y_0,
                np.random.rand() * 2 * np.pi)

    # generates odl space object with ellipses
    def random_phantom(self, spc, n_ellipse=50, interior=False):
        n = np.random.poisson(n_ellipse)
        ellipses = [self.random_ellipse(interior=interior) for _ in range(n)]
        return odl.phantom.ellipsoid_phantom(spc, ellipses)

    # check if path is valid
    def is_valid_path(self, path):
        valid = True
        f = ElementTree.parse(path).getroot()
        session = f.findall('{http://www.nih.gov}readingSession')
        if not session:
            valid = False
        return valid

    # get valid path
    def get_valid_path(self, validation = False):
        k = -1000
        path = ''
        while k < 0:
            path = self.get_random_path(validation=validation)
            if self.is_valid_path(path):
                k = 1
        return path

    # methodes for obtaining the medical data
    def get_random_path(self, validation =False):
        if not validation:
            path = self.training_list[random.randint(0, self.training_list_length-1)]
        else:
            path = self.eval_list[random.randint(0, self.eval_list_length - 1)]
        return path

    # checks if the xml file is a valid source (has readingSessions instead of CXRreadingSessions)
    def valid_xml(self, xml_path):
        valid = True
        f = ElementTree.parse(xml_path).getroot()
        session = f.findall('{http://www.nih.gov}readingSession')
        if not session:
            valid = False
        return valid

    def get_pic(self, path):
        dc_file = dc.read_file(path)
        pic = dc_file.pixel_array
        pic = imresize(pic, [128, 128])
        pic = pic - np.amin(pic)
        pic = pic/ np.amax(pic)
        return pic

    def tv_reconstruction(self, y, param=0.0013):
        # the operators
        gradients = odl.Gradient(self.space, method='forward')
        broad_op = odl.BroadcastOperator(self.operator, gradients)
        # define empty functional to fit the chambolle_pock framework
        g = odl.solvers.ZeroFunctional(broad_op.domain)

        # the norms
        l1_norm = param * odl.solvers.L1Norm(gradients.range)
        l2_norm_squared = odl.solvers.L2NormSquared(self.operator.range).translated(y)
        functional = odl.solvers.SeparableSum(l2_norm_squared, l1_norm)

        # Find parameters
        op_norm = 1.1 * odl.power_method_opnorm(broad_op)
        tau = 10.0 / op_norm
        sigma = 0.1 / op_norm
        niter = 500

        # find starting point
        x = self.fbp(y)

        # Run the optimization algoritm
        # odl.solvers.chambolle_pock_solver(x, functional, g, broad_op, tau = tau, sigma = sigma, niter=niter)
        odl.solvers.pdhg(x, functional, g, broad_op, tau=tau, sigma=sigma, niter=niter)
        return x

    def find_TV_lambda(self, lmd):
        amount_test_images = 1
        y, x_true, fbp = self.generate_data(amount_test_images)
        for l in lmd:
            error = np.zeros([amount_test_images])
            or_error = np.zeros([amount_test_images])
            for k in range(amount_test_images):
                recon = self.tv_reconstruction(y[k, ..., 0], l)
                error[k] = np.sum(np.square(recon - x_true[k, ..., 0]))
                or_error[k] = np.sum(np.square(fbp[k, ..., 0] - x_true[k, ..., 0]))
            total_e = np.mean(np.sqrt(error))
            total_o = np.mean(np.sqrt(or_error))
            print('Lambda: ' + str(l) + ', MSE: ' + str(total_e) + ', OriginalError: ' + str(total_o))


class ct_recon(data_preprocessing):
    model_name = 'default'
    # The batch size
    batch_size = 64
    # relation between L2 error and regulariser
    mu_default = 2
    # weight on gradient norm regulariser for wasserstein network
    lmb = 20
    # learning rate for Adams
    learning_rate = 0.0002
    # step size for picture optimization
    step_size = 0.1

    def get_weights(self):
        return []

    def end(self):
        tf.reset_default_graph()
        self.sess.close()

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
        super(ct_recon, self).__init__()

        # start a tensorflow session
        self.sess = tf.InteractiveSession()

        # placeholders for NN
        self.gen_im = tf.placeholder(shape=[None, 128, 128, 1],
                                     dtype=tf.float32, name='gen_im')
        self.true_im = tf.placeholder(shape=[None, 128, 128, 1],
                                      dtype=tf.float32)
        self.random_uint = tf.placeholder(shape=[None],
                                          dtype=tf.float32)

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
        self.reconstruction = tf.placeholder(shape=[None, 128, 128, 1], dtype=tf.float32)
        self.data_term = tf.placeholder(shape=[None, None, None, 1], dtype=tf.float32)
        self.mu = tf.placeholder(dtype=tf.float32)

        # data loss
        self.ray = self.ray_transform(self.reconstruction)
        data_mismatch = tf.square(self.ray - self.data_term)
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
        self.ground_truth = tf.placeholder(shape=[None, 128, 128, 1], dtype=tf.float32)
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
        estimation = np.zeros(shape=[sample])
        for k in range(sample):
            y, x_true, fbp = self.generate_data(1)
            data_mismatch = self.operator(x_true[0,...,0]) - y[0,...,0]
            adj_mismatch = self.adjoint(data_mismatch)
            estimation[k] = 2*np.sqrt(np.sum(np.square(adj_mismatch)))
        ### for consistency, compute optimal lambda with graph as well
        y, x_true, fbp = self.generate_data(sample)
        gradient_truth = self.sess.run(self.pic_grad, {self.reconstruction: x_true,
                                                 self.data_term: y,
                                                 self.ground_truth: x_true,
                                                 self.mu: 0})
        print(np.sqrt(np.sum(np.square(gradient_truth[0]), axis=(1,2,3))))
        print(np.mean(np.sqrt(np.sum(np.square(gradient_truth[0]), axis=(1,2,3)))))
        print(np.mean(estimation))

    # Method to determine the L2 noise level ||Kx-y|| where x is the ground truth
    def find_noise_level(self, sample= 256 ):
        y, x_true, fbp = self.generate_data(sample)
        data_error = self.sess.run(self.data_error, {self.reconstruction: x_true,
                                                 self.data_term: y,
                                                 self.ground_truth: x_true,
                                                 self.mu: 0})
        print(data_error)

    # visualization methode
    def visualize(self, true, fbp, tv, recon, global_step, step, mu):
        quality = np.average(np.sqrt(np.sum(np.square(true - recon), axis=(1, 2, 3))))
        print('Quality of reconstructed image: ' + str(quality))
        self.create_single_folder('Saves/Pictures/' + self.model_name + '/' +str(global_step))
        plt.figure()
        plt.subplot(221)
        plt.imshow(true[-1, ...,0])
        plt.axis('off')
        plt.title('Original')
        plt.subplot(222)
        plt.imshow(fbp[-1,...,0])
        plt.axis('off')
        plt.title('FBP')
        plt.suptitle('L2 :' + str(quality))
        plt.subplot(223)
        plt.imshow(tv)
        plt.axis('off')
        plt.title('TV')
        plt.subplot(224)
        plt.imshow(recon[-1,...,0])
        plt.title('Reconstruction')
        plt.axis('off')
        plt.savefig('Saves/Pictures/' + self.model_name + '/' +str(global_step) +'/mu-' + str(mu)+
                    'iteration-' + str(step) + '.png')
        plt.close()

    # picture update
    def update_pic(self, steps, stepsize, measurement, guess, mu):
        for k in range(steps):
            gradient = self.sess.run(self.pic_grad, feed_dict={self.reconstruction: guess,
                                                                self.data_term: measurement,
                                                               self.mu: mu})
            guess = guess - stepsize * gradient[0]
        return guess

    # unregularised minimization
    def unreg_mini(self, y, fbp):
        return self.update_pic(25, 0.1, y, fbp, 0)


    # visualization of Picture optimization
    def create_optimized_images(self, batch_size, steps = 30, step_s = step_size,
                                mu = mu_default, starting_point = 'FBP'):
        y, x_true, fbp = self.generate_data(batch_size)
        guess = np.copy(fbp)
        if starting_point == 'Mini':
            guess = self.unreg_mini(y, fbp)
        step = self.sess.run(self.global_step)
        writer = tf.summary.FileWriter('Saves/Logs/' + self.model_name + '/Picture_Opt/Iteration_' +
                                       str(step) + '/' + str(mu) + '/')
        tv = self.tv_reconstruction(y[-1, ..., 0])
        for k in range(steps):
            summary = self.sess.run(self.merged_pic,
                                      feed_dict={self.reconstruction: guess,
                                                 self.data_term: y,
                                                 self.ground_truth: x_true,
                                                 self.mu: mu})
            writer.add_summary(summary, k)
            if (k%5 == 0):
                self.visualize(x_true, fbp, tv, guess, step, k, mu)
            guess = self.update_pic(1, step_s, y, guess, mu)
        writer.close()


    # evaluates and prints the network performance
    def evaluate_Network(self, mu, starting_point = 'FBP'):
        y, true, fbp = self.generate_data(64)
        if starting_point == 'Mini':
            fbp = self.unreg_mini(y, fbp)
        # generate random distribution for rays
        epsilon = np.random.uniform(size=(self.batch_size))
        step, Was_g, reg_g, Was, reg = self.sess.run([self.global_step, self.g1, self.g2,
                                                      self.wasserstein_loss, self.regulariser_was],
                                                     feed_dict={self.gen_im: fbp, self.true_im: true,
                                                                self.random_uint: epsilon})
        print('Iteration: ' + str(step) + ', Was: ' + str(Was) + ', Reg: ' + str(reg) +
              ', Was grad: ' + str(Was_g) + ', Reg grad: ' + str(reg_g))

        # tensorflow logging
        y, x_true, fbp_p = self.generate_data(64)
        guess = np.copy(fbp_p)
        if starting_point == 'Mini':
            guess = self.unreg_mini(y, fbp)
        guess = self.update_pic(30, self.step_size, y, guess, mu)
        summary, step = self.sess.run([self.merged, self.global_step],
                                      feed_dict={self.gen_im: fbp,
                                                 self.true_im: true,
                                                 self.random_uint: epsilon,
                                                 self.reconstruction: guess,
                                                 self.data_term: y,
                                                 self.ground_truth: x_true,
                                                 self.mu: mu})
        self.writer.add_summary(summary, step)

    def generate_training_images(self, batch_size, amount_steps, mu = mu_default, starting_point = 'FBP'):
        true_im = np.zeros(shape=(batch_size, 128,128,1))
        output_im = np.zeros(shape=(batch_size, 128, 128, 1))
        output_fbp = np.zeros(shape=(batch_size, 128, 128, 1))

        #create remaining samples
        for j in range(batch_size):
            y, x_true, fbp = self.generate_data(1)
            guess = np.copy(fbp)
            if starting_point == 'Mini':
                guess = self.unreg_mini(y, fbp)
            s = random.randint(1,amount_steps)
            guess = self.update_pic(s, self.step_size, y, guess, mu)
            true_im[j,...] = x_true[0,...]
            output_fbp[j, ...] = fbp[0,...]
            output_im[j, ...] = guess[0,...]
        return true_im, output_fbp, output_im

    # control methode to check that generate_training_images works as it should
    def control(self, steps):
        true, fbp, recon = self.generate_training_images(4,steps, starting_point= 'Mini')
        for k in range(4):
            plt.figure()
            plt.subplot(131)
            plt.imshow(true[k, ..., 0])
            plt.axis('off')
            plt.title('Original')
            plt.subplot(132)
            plt.imshow(fbp[k, ..., 0])
            plt.axis('off')
            plt.title('FBP')
            plt.subplot(133)
            plt.imshow(recon[k, ..., 0])
            plt.axis('off')
            plt.title('Reconstruction')
            plt.savefig('Saves/Pictures/' + self.model_name + '/test/' + str(k) + '.jpg')
            plt.close()

    # optimize network on initial guess input only, with initial guess being fbp
    def pretrain_Wasser_FBP(self, steps, mu = mu_default):
        for k in range(steps):
            if k%20 == 0:
                self.evaluate_Network(mu)
            if k%100 == 0:
                self.create_optimized_images(64)
            y, x_true, fbp = self.generate_data(self.batch_size)
            # generate random distribution for rays
            epsilon = np.random.uniform(size=(self.batch_size))
            # optimize network
            self.sess.run(self.optimizer,
                          feed_dict={self.gen_im: fbp, self.true_im: x_true, self.random_uint: epsilon})
        self.save()

    # optimize network on initial guess input only, with initial guess being minimizer of ||Kx - y||
    def pretrain_Wasser_DataMinimizer(self, steps, mu = mu_default):
        for k in range(steps):
            if k%20 == 0:
                self.evaluate_Network(mu, starting_point='Mini')
            if k%100 == 0:
                self.create_optimized_images(64, starting_point='Mini')
            y, x_true, fbp = self.generate_data(self.batch_size)
            # optimize the fbp to fit the data term
            mini = self.unreg_mini(y, fbp)
            # generate random distribution for rays
            epsilon = np.random.uniform(size=(self.batch_size))
            # optimize network
            self.sess.run(self.optimizer,
                          feed_dict={self.gen_im: mini, self.true_im: x_true, self.random_uint: epsilon})
        self.save()

    # iterative training methode, using actual output distribtion instead of initial guess distribution
    def train(self, steps, amount_steps, starting_point, mu = mu_default):
        for k in range(steps):
            if k % 20 == 0:
                self.evaluate_Network(mu, starting_point= starting_point)
            if k % 200 == 0:
                self.create_optimized_images(256, steps = amount_steps, starting_point= starting_point)
            true, fbp, gen = self.generate_training_images(self.batch_size, amount_steps=amount_steps,
                                                           mu = mu, starting_point= starting_point)
            # generate random distribution for rays
            epsilon = np.random.uniform(size=(self.batch_size))
            # optimize network
            self.sess.run(self.optimizer,
                          feed_dict={self.gen_im: gen, self.true_im: true, self.random_uint: epsilon})
        self.save()

def lrelu(x):
    return (tf.nn.relu(x) - 0.1*tf.nn.relu(-x))

class Recon1(ct_recon):
    model_name = 'Recon1'
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
        logits_W = tf.get_variable(name="logits_W_ad", shape=[128*8*8, 1],
                                   initializer=(
                                   tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32)))
        logits_bias = tf.Variable(tf.constant(0.0, shape=[1, 1]), name='logits_bias_ad')
        con_pre = tf.get_variable(name="conv_pre", shape=[5, 5, 1, 16],
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

class Recon_LUNA(Recon1):
    model_name = 'Recon_LUNA'
    source = 'LUNA'





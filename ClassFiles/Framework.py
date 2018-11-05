import numpy as np
import os
import odl
import odl.contrib.tensorflow
from abc import ABC, abstractmethod
import tensorflow as tf
from ClassFiles import util as ut


class GenericFramework(ABC):
    model_name = 'no_model'
    experiment_name = 'default_experiment'

    # set the noise level used for experiments
    noise_level = 0.02

    @abstractmethod
    def get_network(self, size, colors):
        # returns an object of the network class. Used to set the network used
        pass

    @abstractmethod
    def get_Data_pip(self, path):
        # returns an object of the data_pip class.
        pass

    @abstractmethod
    def get_model(self, size):
        # Returns an object of the forward_model class.
        pass

    def __init__(self, data_path, saves_path):
        self.data_pip = self.get_Data_pip(data_path)
        self.colors = self.data_pip.colors
        self.image_size = self.data_pip.image_size
        self.network = self.get_network(self.image_size, self.colors)
        self.model = self.get_model(self.image_size)
        self.image_space = self.model.get_image_size()
        self.measurement_space = self.model.get_measurement_size()

        # finding the correct path for saving models
        self.path = saves_path+'Saves/{}/{}/{}/{}/'.format(self.model.name, self.data_pip.name,
                                                           self.model_name, self.experiment_name)
        # start tensorflow sesssion
        self.sess = tf.InteractiveSession()

        # generate needed folder structure
        ut.create_single_folder(self.path+'Data')
        ut.create_single_folder(self.path + 'Logs')

    def generate_training_data(self, batch_size, training_data=True):
        # method to generate training data given the current model type
        y = np.empty((batch_size, self.measurement_space[0], self.measurement_space[1], self.colors), dtype='float32')
        x_true = np.empty((batch_size, self.image_space[0], self.image_space[1], self.colors), dtype='float32')
        fbp = np.empty((batch_size, self.image_space[0], self.image_space[1], self.colors), dtype='float32')

        for i in range(batch_size):
            if training_data:
                image = self.data_pip.load_data(training_data=True)
            else:
                image = self.data_pip.load_data(training_data=False)
            data = self.model.forward_operator(image)

            # add white Gaussian noise
            noisy_data = data + self.noise_level*np.random.normal(size=(self.measurement_space[0],
                                                                        self.measurement_space[1],
                                                                        self.colors))
            print(noisy_data.shape)
            fbp[i, ...] = self.model.inverse(noisy_data)
            x_true[i, ...] = image
            y[i, ...] = noisy_data
        return y, x_true, fbp

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

    @abstractmethod
    def evaluate(self, guess, measurement):
        # apply the model to data
        pass


class AdversarialRegulariser(GenericFramework):
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
    # default step size for picture optimization
    step_size = 1
    # the amount of steps of gradient descent taken on loss functional
    total_steps_default = 30
    # default sampling pattern
    starting_point = 'fbp'

    def set_total_steps(self, steps):
        self.total_steps = steps

    # sets up the network architecture
    def __init__(self, data_path, saves_path):
        # call superclass init
        super(AdversarialRegulariser, self).__init__(data_path, saves_path)
        self.total_steps = self.total_steps_default

        ### Training the regulariser ###

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

        ### The reconstruction network ###

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
        # averaged quantities already. Makes gradients scaling batch size inveriant
        batch_s = tf.cast(tf.shape(self.reconstruction)[0], tf.float32)

        # Optimization for the picture
        self.pic_grad = tf.gradients(self.full_error * batch_s, self.reconstruction)

        # Measure quality of reconstruction
        self.ground_truth = tf.placeholder(shape=[None, self.image_space[0], self.image_space[0], self.colors], dtype=tf.float32)
        self.quality = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(self.ground_truth - self.reconstruction),
                                                            axis=(1, 2, 3))))

        # logging tools
        with tf.name_scope('Network_Optimization'):
            dd = tf.summary.scalar('Data_Difference', self.wasserstein_loss)
            lr = tf.summary.scalar('Lipschitz_Regulariser', self.regulariser_was)
            ol = tf.summary.scalar('Overall_Net_Loss', self.loss_was)
            self.merged_network = tf.summary.merge([dd, lr, ol])
        with tf.name_scope('Picture_Optimization'):
            data_loss = tf.summary.scalar('Data_Loss', self.data_error)
            wasser_loss = tf.summary.scalar('Wasserstein_Loss', self.was_output)
            recon = tf.summary.image('Reconstruction', self.reconstruction, max_outputs=1)
            ground_truth = tf.summary.image('Ground_truth', self.ground_truth, max_outputs=1)
            quality_assesment = tf.summary.scalar('L2_to_ground_truth', self.quality)
            self.merged_pic = tf.summary.merge([data_loss, wasser_loss, quality_assesment, recon, ground_truth])
        with tf.name_scope('Reconstruction_Quality'):
            data_loss = tf.summary.scalar('Data_Loss', self.data_error)
            wasser_loss = tf.summary.scalar('Wasserstein_Loss', self.was_output)
            recon = tf.summary.image('Reconstruction', self.reconstruction, max_outputs=1)
            ground_truth = tf.summary.image('Ground_truth', self.ground_truth, max_outputs=1)
            quality_assesment = tf.summary.scalar('L2_to_ground_truth', self.quality)
            self.training_eval = tf.summary.merge([data_loss, wasser_loss, quality_assesment, recon, ground_truth])

        # set up the logger
        self.writer = tf.summary.FileWriter(self.path + 'Logs/Network_Optimization/',
                                            self.sess.graph)

        # initialize Variables
        tf.global_variables_initializer().run()

        # load existing saves
        self.load()

    def update_pic(self, steps, stepsize, measurement, guess, mu):
        # updates the guess to come closer to the solution of the variational problem.
        for k in range(steps):
            gradient = self.sess.run(self.pic_grad, feed_dict={self.reconstruction: guess,
                                                               self.data_term: measurement,
                                                               self.mu: mu})
            guess = guess - stepsize * gradient[0]
        return guess

    def unreg_mini(self, y, fbp):
        # unregularised minimization. In case the method to compute the pseudo inverse returns images that
        # are far from the data minimizer, it might be benificial to do some steps of gradient descent on the data
        # term before applying the adversarial regularizer algorithm. Set starting_point to 'Mini' and define the amount
        # of steps and step size to be performed before any training or reconstruction on the data term here.
        return self.update_pic(10, 0.1, y, fbp, 0)

    def log_minimum(self):
        # Finds the optimumt of reconstruction and logs the values of this point only
        # This method is meant for quality evaluation during training
        # It uses the default values set in the class file for step size, total steps and regularisation
        # parameter mu.
        y, x_true, fbp = self.generate_training_data(self.batch_size, training_data=False)
        guess = np.copy(fbp)
        if self.starting_point == 'Mini':
            guess = self.unreg_mini(y, fbp)
        k = 0
        minimum = False
        while k <= self.total_steps and minimum == False:
            guess_update = self.update_pic(1, self.step_size, y, guess, self.mu_default)
            if ut.l2_norm(guess_update - x_true) >= ut.l2_norm(guess-x_true):
                minimum = True
            else:
                guess = guess_update
        logs, step = self.sess.run([self.merged_pic, self.global_step], feed_dict={self.reconstruction: guess,
                                                         self.data_term: y,
                                                         self.ground_truth: x_true,
                                                         self.mu: self.mu_default})
        self.writer.add_summary(logs, step)

    def log_network_training(self):
        # evaluates and prints the network performance.
        y, x_true, fbp = self.generate_training_data(batch_size=self.batch_size, training_data=False)
        guess = np.copy(fbp)
        if self.starting_point == 'Mini':
            guess = self.unreg_mini(y, fbp=fbp)
        # generate random distribution for rays
        epsilon = np.random.uniform(size=self.batch_size)
        logs, step = self.sess.run([self.merged_network, self.global_step],
                                   feed_dict={self.gen_im: guess, self.true_im: x_true,
                                              self.random_uint: epsilon})
        self.writer.add_summary(logs, step)

    def log_optimization(self, batch_size=None, steps=None, step_s=None,
                                    mu=None, starting_point=None):
        # Logs every step of picture optimization.
        # Can be used to play with the variational formulation once training is complete
        if batch_size is None:
            batch_size = self.batch_size
        if steps is None:
            steps = self.total_steps
        if step_s is None:
            step_s = self.step_size
        if mu is None:
            mu = self.mu_default
        if starting_point is None:
            starting_point = self.starting_point
        y, x_true, fbp = self.generate_training_data(batch_size, training_data=False)
        guess = np.copy(fbp)
        if starting_point == 'Mini':
            guess = self.unreg_mini(y, fbp)
        writer = tf.summary.FileWriter(self.path + '/Logs/Picture_Opt/mu_{}_step_s_{}'.format(mu, step_s))
        for k in range(steps+1):
            summary = self.sess.run(self.merged_pic,
                                    feed_dict={self.reconstruction: guess,
                                               self.data_term: y,
                                               self.ground_truth: x_true,
                                               self.mu: mu})
            writer.add_summary(summary, k)
            guess = self.update_pic(1, step_s, y, guess, mu)
        writer.close()

    def train(self, steps):
        # the training routine
        for k in range(steps):
            if k % 50 == 0:
                self.log_network_training()
                self.log_minimum()
            y, x_true, fbp = self.generate_training_data(self.batch_size)
            # generate random distribution for rays
            epsilon = np.random.uniform(size=self.batch_size)
            # optimize network
            self.sess.run(self.optimizer,
                          feed_dict={self.gen_im: fbp, self.true_im: x_true, self.random_uint: epsilon})
        self.save(self.global_step)

    def find_good_lambda(self, sample=64):
        # Method to estimate a good value of the regularisation paramete.
        # This is done via estimation of 2 ||K^t (Kx-y)||_2 where x is the ground truth
        y, x_true, fbp = self.generate_training_data(sample)
        gradient_truth = self.sess.run(self.pic_grad, {self.reconstruction: x_true,
                                       self.data_term: y,
                                       self.ground_truth: x_true,
                                       self.mu: 0})
        print(np.mean(np.sqrt(np.sum(np.square(gradient_truth[0]), axis=(1, 2, 3)))))

    def evaluate(self, guess, measurement):
        fbp = np.copy(guess)
        if self.starting_point == 'Mini':
            fbp = self.unreg_mini(y, fbp)
        return self.update_pic(steps=self.total_steps, measurement=measurement, guess=fbp,
                               stepsize=self.step_size, mu=self.mu_default)


class PostProcessing(GenericFramework):
    # Framework for postprocessing
    model_name = 'PostProcessing'
    # learning rate for Adams
    learning_rate = 0.001
    # The batch size
    batch_size = 16
    # noise level
    noise_level = 0.02

    def __init__(self, data_path, saves_path):
        # call superclass init
        super(PostProcessing, self).__init__(data_path, saves_path)

        # set placeholder for input and correct output
        self.true = tf.placeholder(shape=[None, self.image_space[0], self.image_space[1], self.data_pip.colors], dtype=tf.float32)
        self.fbp = tf.placeholder(shape=[None, self.image_space[0], self.image_space[1], self.data_pip.colors], dtype=tf.float32)
        # network output
        self.out = self.network.net(self.fbp)
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
        tf.summary.image('Reconstruction', self.out, max_outputs=1)
        tf.summary.image('Input', self.fbp, max_outputs=1)
        tf.summary.image('GroundTruth', self.true, max_outputs=1)

        # set up the logger
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.path + 'Logs/',
                                            self.sess.graph)

        # initialize Variables
        tf.global_variables_initializer().run()

        # load existing saves
        self.load()

    def log(self, x_true, fbp):
        summary, step = self.sess.run([self.merged, self.global_step],
                                      feed_dict={self.true: x_true,
                                                 self.fbp: fbp})
        self.writer.add_summary(summary, step)

    def train(self, steps):
        for k in range(steps):
            y, x_true, fbp = self.generate_training_data(self.batch_size)
            self.sess.run(self.optimizer, feed_dict={self.true: x_true,
                                                    self.fbp: fbp})
            if k % 50 == 0:
                y, x_true, fbp = self.generate_training_data(self.batch_size, training_data=True)
                self.log(x_true, fbp)
        self.save(self.global_step)

    def evaluate(self, guess, measurement):
        output = self.sess.run(self.out, feed_dict={self.fbp: guess})
        return output

    def evaluate_red(self, y, initial_guess, step_size, reg_para, steps):
        # implements the RED method with the denoising neural network as denoising model.
        guess = initial_guess
        for j in range(steps):
            gradient_data = np.zeros(shape=(guess.shape[0], self.image_size[0], self.image_space[1], self.colors))
            for k in range(guess.shape[0]):
                data_misfit = self.model.forward_operator(guess[k, ...]) - y[k, ...]
                gradient_data[k, ...] = self.model.forward_operator_adjoint(data_misfit)
            gradient_reg = guess - self.evaluate(y, guess)
            gradient = gradient_data + reg_para*gradient_reg
            guess = guess - step_size * gradient
        return guess


class TotalVariation(GenericFramework):
    # TV reconstruction
    model_name = 'TV'

    # TV hyperparameters
    noise_level = 0.02
    def_lambda = 0.003

    def __init__(self, data_path, saves_path):
        # call superclass init
        super(TotalVariation, self).__init__(data_path, saves_path)
        self.operator = self.model.get_odl_operator()
        self.space = self.operator.domain
        self.range = self.operator.range

    def get_network(self, size, colors):
        # Blank overwrite, as this method is not needed for TV
        pass

    def tv_reconstruction(self, y, param=def_lambda):
        # internal method to evaluate tv on a single element y with shape [width, height]

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
        niter = 200

        # find starting point
        x = self.space.element(self.model.inverse(np.expand_dims(y, axis=-1))[...,0])

        # Run the optimization algoritm
        # odl.solvers.chambolle_pock_solver(x, functional, g, broad_op, tau = tau, sigma = sigma, niter=niter)
        odl.solvers.pdhg(x, functional, g, broad_op, tau=tau, sigma=sigma, niter=niter)
        return x

    def find_TV_lambda(self, lmd):
        amount_test_images = 32
        y, true, fbp = self.generate_training_data(amount_test_images)
        for l in lmd:
            error = np.zeros((amount_test_images, self.colors))
            or_error = np.zeros((amount_test_images, self.colors))
            guess = np.copy(fbp)
            for j in range(self.colors):
                for k in range(amount_test_images):
                    recon = self.tv_reconstruction(y[k, ..., j], l)
                    guess[k,...,j] = recon
                    error[k,j] = np.sum(np.square(recon - true[k, ..., j]))
                    or_error[k,j] = np.sum(np.square(fbp[k, ..., j] - true[k, ..., j]))
            total_e = np.mean(np.sqrt(np.sum(error, axis=1)))
            total_o = np.mean(np.sqrt(np.sum(or_error, axis=1)))
            print('Lambda: ' + str(l) + ', MSE: ' + str(total_e) + ', OriginalError: ' + str(total_o))

    def evaluate(self, guess, measurement):
        amount_images = measurement.shape[0]
        for j in range(self.colors):
            for k in range(amount_images):
                recon = self.tv_reconstruction(measurement[k, ..., j], self.def_lambda)
                guess[k, ..., j] = recon
        return guess





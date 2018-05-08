from data_pips import ellipses
from data_pips import LUNA
from data_pips import BSDS

from forward_models import ct
from forward_models import denoising

from Framework import adversarial_regulariser
from Framework import positiv_adversarial_regulariser
from Framework import postprocessing
from Framework import iterative_scheme
from Framework import total_variation

from networks import multiscale_l1_classifier
from networks import resnet_classifier

number = input("Please enter number of experiment you want to run: ")

nl1 = 0.1

# Denoiser with dilated l1 architecture
if number == 1:
    print('Running denoiser with dilated l1 architecture')
    class l1_denoiser(adversarial_regulariser):
        # weight on gradient norm regulariser for wasserstein network
        lmb = 50
        default_sampling_pattern = 'startend'
        experiment_name = 'l1_arch'

        noise_level = nl1
        mu_default = 60
        learning_rate = 0.0002
        step_size = .05
        total_steps_default = 15

        def get_network(self, size, colors):
            return multiscale_l1_classifier(size=size, colors=colors)

        def get_Data_pip(self):
            return BSDS()

        def get_model(self, size):
            return denoising(size=size)

    adv_reg = l1_denoiser()

    ex_number = input('Number of experiment to run')
    if ex_number == 1:
        adv_reg.find_good_lambda()
        adv_reg.end()

    if ex_number == 2:
        repeat = 1
        while repeat == 1:
            ss = input('Please insert desired steps size: ')
            a_s = input('Please insert amount of steps: ')
            mu = input('Please insert regularisation parameter mu: ')
            adv_reg.evaluate_image_optimization(batch_size=64, mu=mu, step_s=ss,
                                                steps=a_s, starting_point='FBP')
            repeat = input('Repeat experiment?')
        adv_reg.end()

    if ex_number == 3:
        for k in range(3):
            adv_reg.pretrain_Wasser_FBP(300)

    if ex_number == 4:
        for k in range(3):
            adv_reg.train(300, starting_point='FBP')

# Denoiser with ResNet
if number == 2:
    print('Running denoiser with resNet')
    class l1_denoiser(adversarial_regulariser):
        # weight on gradient norm regulariser for wasserstein network
        lmb = 20

        experiment_name = 'resNet'

        noise_level = nl1
        mu_default = 60
        learning_rate = 0.0002
        step_size = .05
        total_steps_default = 30
        default_sampling_pattern = 'startend'

        def get_network(self, size, colors):
            return resnet_classifier(size=size, colors=colors)

        def get_Data_pip(self):
            return BSDS()

        def get_model(self, size):
            return denoising(size=size)



    adv_reg = l1_denoiser()

    ex_number = input('Number of experiment to run')
    if ex_number == 1:
        adv_reg.find_good_lambda()
        adv_reg.end()

    if ex_number == 2:
        repeat = 1
        while repeat == 1:
            ss = input('Please insert desired steps size: ')
            a_s = input('Please insert amount of steps: ')
            mu = input('Please insert regularisation parameter mu: ')
            adv_reg.evaluate_image_optimization(batch_size=64, mu=mu, step_s=ss,
                                                steps=a_s, starting_point='FBP')
            repeat = input('Repeat experiment?')
        adv_reg.end()

    if ex_number == 3:
        print('Pretraining')
        for k in range(3):
            adv_reg.pretrain_Wasser_FBP(300)

    if ex_number ==4:
        print('Iteratives Training')
        for k in range(3):
            adv_reg.train(300, starting_point='FBP')

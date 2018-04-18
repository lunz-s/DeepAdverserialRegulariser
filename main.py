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

### CT experiments
number = input("Please enter number of experiment you want to run: ")

# Experiments 0.0: Step size checks for solving variational problem later
if number == 0:
    print('Start unregularised optimisation experiments.')
    repeat = 1
    nl = input('Please insert desired noise level: ')
    class unreg_exp(adversarial_regulariser):
        experiment_name = 'Unregularised_mini'
        noise_level = nl
        mu_default = 1.5
        learning_rate = 0.0005
        step_size = 0.1
        total_steps_default = 30
    ur = unreg_exp()

    while repeat == 1:
        ss = input('Please insert desired steps size: ')
        a_s = input('Please insert amount of steps: ')
        ur.evaluate_image_optimization(batch_size=16, mu=0, step_s=ss,
                                              steps = a_s, starting_point='FBP')
        repeat = input('Repeat experiment?')

class exp1(adversarial_regulariser):
    experiment_name = 'LowNoiseExp1'
    noise_level = 0.01
    mu_default = .3
    learning_rate = 0.0005
    step_size = 1
    total_steps_default = 30

    def unreg_mini(self, y, fbp):
        return self.update_pic(15, 1, y, fbp, 0)

class pos_AR(positiv_adversarial_regulariser):
    experiment_name = 'positivAR'
    noise_level = 0.01
    mu_default = .3
    learning_rate = 0.0005
    step_size = 1
    total_steps_default = 30
    eps = 0.1

    def unreg_mini(self, y, fbp):
        return self.update_pic(15, 1, y, fbp, 0)


class exp2(exp1):
    experiment_name = 'OverregularisedRecursiveTraining'
    mu_default = 2.5

# Experiment 1.0: AR with noise level 0.01, standard classifier network, LUNA data set
if number == 1:
    print('Run AR algorithm, low noise, standard architecture')

    # create object of type experiment1
    adv_reg = exp1()
    adv_reg.set_total_steps(15)
    # adv_reg.find_good_lambda()
    for k in range(2):
        adv_reg.train(500)
    adv_reg.evaluate_image_optimization(steps=70)

    #adv_reg.train(500)
    adv_reg.end()

# Experiment to check good regularization level mu
if number == 1.1:
    class find_mu(adversarial_regulariser):
        experiment_name = 'Check_Reg_Level'
        noise_level = 0.01
        mu_default = 2
        learning_rate = 0.0005
        step_size = 0.7
        total_steps_default = 50

    adv_reg = find_mu()
    adv_reg.find_good_lambda()
    adv_reg.end()

# Experiment to check how quickly variational problem can be solved
if number == 1.2:
    net = input('Plesae insert type of network to be used: ')
    if net ==1:
        adv_reg = exp1()
    else:
        adv_reg = exp2()
    repeat = 1
    while repeat == 1:
        ss = input('Please insert desired steps size: ')
        a_s = input('Please insert amount of steps: ')
        mu = input('Please insert regularisation parameter mu: ')
        adv_reg.evaluate_image_optimization(batch_size=32, mu=mu, step_s=ss,
                                       steps=a_s, starting_point='Mini')
        repeat = input('Repeat experiment?')
    adv_reg.end()

# Overregularised rekursive training
if number == 1.3:
    print('Running overregularised rekursive training')
    adv_reg = exp2()
    adv_reg.set_total_steps(3)
    for k in range(2):
        adv_reg.train(500)

#positive AR
if number == 1.4:
    print('Running positiv AR')
    adv_reg = pos_AR()
    for k in range(3):
        adv_reg.train(500)


# Experiment 2.0 post-processing with noise level 0.01, standard UNet, LUNA data set
if number==2:
    print('Run Postprocessing, low noise, standard UNet')
    class exp2(postprocessing):
        experiment_name = 'Noise_0.01_SmallUNet'
        noise_level = 0.01
        learning_rate = 0.001

    pp = exp2()
    pp.train(500)
    pp.end()

# Experiment 3.0 iterative scheme with noise level 0.01, fully convolutional nn, LUNA data set
if number == 3:
    print('Run iterative scheme, low noise, standard convNet')
    class exp3(iterative_scheme):
        experiment_name = 'Noise_0.01_'
        noise_level = 0.01
        learning_rate = 0.0003

    it = exp3()
    it.train(500)
    it.end()

# Denoiser with dilated l1 architecture
if number == 4:
    print('Running denoiser with dilated l1 architecture')
    class l1_denoiser(adversarial_regulariser):
        # weight on gradient norm regulariser for wasserstein network
        lmb = 50
        default_sampling_pattern = 'startend'
        experiment_name = 'l1_arch'

        noise_level = 0.1
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

if number == 5:
    print('Running denoiser with resNet')
    class l1_denoiser(adversarial_regulariser):
        # weight on gradient norm regulariser for wasserstein network
        lmb = 20

        experiment_name = 'resNet'

        noise_level = 0.1
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





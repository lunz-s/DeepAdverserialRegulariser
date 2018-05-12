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
from networks import improved_binary_classifier

number = input("Please enter number of experiment you want to run: ")

nl1 = 0.02

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
    experiment_name = 'ResNet'
    noise_level = nl1
    mu_default = .3
    learning_rate = 0.0002
    step_size = 1
    total_steps_default = 30

    def get_network(self, size, colors):
        return resnet_classifier(size=size, colors=colors)

    def unreg_mini(self, y, fbp):
        return self.update_pic(15, 1, y, fbp, 0)

# Experiment 1.0: AR with noise level 0.01, standard classifier network, LUNA data set
if number == 1:
    print('Run AR algorithm, ResNet')

    # create object of type experiment1
    adv_reg = exp1()
    adv_reg.set_total_steps(30)
    # adv_reg.find_good_lambda()
    for k in range(2):
        adv_reg.pretrain_Wasser_DataMinimizer(500)
    adv_reg.evaluate_image_optimization(steps=70)

    #adv_reg.train(500)
    adv_reg.end()

# Experiment to check good regularization level mu
if number == 1.1:
    class find_mu(adversarial_regulariser):
        experiment_name = 'Check_Reg_Level'
        noise_level = nl1
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

class l1_exp(adversarial_regulariser):
    experiment_name = 'dilatedL1'
    noise_level = nl1
    mu_default = .3
    learning_rate = 0.0005
    step_size = 1
    total_steps_default = 30

    def get_network(self, size, colors):
        return multiscale_l1_classifier(size=size, colors=colors)

    def unreg_mini(self, y, fbp):
        return self.update_pic(15, 1, y, fbp, 0)

if number == 2:
    # create object of type experiment1
    adv_reg = l1_exp()
    adv_reg.set_total_steps(30)
    # adv_reg.find_good_lambda()
    for k in range(10):
        adv_reg.pretrain_Wasser_DataMinimizer(500)
    adv_reg.evaluate_image_optimization(steps=70)

### Comparison experiments: Standard architecture
class reference(adversarial_regulariser):
    experiment_name = 'ConvNet'
    noise_level = nl1
    mu_default = .7
    learning_rate = 0.0001
    step_size = 1
    total_steps_default = 15
    default_sampling_pattern = 'startend'

    def get_network(self, size, colors):
        return improved_binary_classifier(size=size, colors=colors)

    def unreg_mini(self, y, fbp):
        return self.update_pic(15, 1, y, fbp, 0)

if number == 3:
    # create object of type experiment1
    adv_reg = reference()
    adv_reg.set_total_steps(30)
    # adv_reg.find_good_lambda()
    for k in range(5):
        adv_reg.pretrain_Wasser_DataMinimizer(500)
    adv_reg.evaluate_image_optimization(steps=70)

if number == 3.1:
    adv_reg = reference()
    adv_reg.find_good_lambda()

    repeat = 1
    while repeat == 1:
        ss = input('Please insert desired steps size: ')
        a_s = input('Please insert amount of steps: ')
        mu = input('Please insert regularisation parameter mu: ')
        adv_reg.evaluate_image_optimization(batch_size=32, mu=mu, step_s=ss,
                                       steps=a_s, starting_point='Mini')
        repeat = input('Repeat experiment?')
    adv_reg.end()

if number == 3.2:
    adv_reg = reference()
    adv_reg.set_total_steps(20)
    adv_reg.set_sampling_pattern('uniform')
    adv_reg.check_recursive_patter()
    for k in range(5):
        adv_reg.train(500)
    adv_reg.evaluate_image_optimization(50)

if number == 4.0:
    tv = total_variation()
    print(tv.noise_level)
    lmb = []
    for k in range(10):
        lmb.append(0.001*(k+1))
    tv.find_TV_lambda(lmb)

if number == 5.0:
    recon = postprocessing()
    print(recon.noise_level)
    for k in range(5):
        recon.train(500)

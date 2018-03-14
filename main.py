from data_pips import ellipses
from data_pips import LUNA
from data_pips import BSDS

from forward_models import ct
from forward_models import denoising

from Framework import adversarial_regulariser
from Framework import postprocessing
from Framework import iterative_scheme
from Framework import total_variation

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


# Experiment 1.0: AR with noise level 0.01, standard classifier network, LUNA data set
if number == 1:
    print('Run AR algorithm, low noise, standard architecture')
    class exp1(adversarial_regulariser):
        experiment_name = 'Noise_0.01_StandardNet'
        noise_level = 0.01
        mu_default = .5
        learning_rate = 0.0005
        step_size = 0.7
        total_steps_default = 50

        def unreg_mini(self, y, fbp):
            return self.update_pic(15, 1, y, fbp, 0)

    adv_reg = exp1()
    adv_reg.find_good_lambda()
    #adv_reg.pretrain_Wasser_DataMinimizer(500)
    adv_reg.train(500)
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

# Experiment 4.0: AR with noise level 0.1, standard classifier network, LUNA data set
if number == 4:
    print('Run AR algorithm, high noise, standard architecture')
    class exp1(adversarial_regulariser):
        experiment_name = 'Noise_0.1_StandardNet'
        noise_level = 0.1
        mu_default = 1.5
        learning_rate = 0.0005
        step_size = 0.8
        total_steps_default = 30

    adv_reg = exp1()
    adv_reg.find_good_lambda()
    adv_reg.pretrain_Wasser_DataMinimizer(500)
    adv_reg.end()

# Experiment 5.0 post-processing with noise level 0.1, standard UNet, LUNA data set
if number==5:
    print('Run Postprocessing, high noise, standard UNet')
    class exp2(postprocessing):
        experiment_name = 'Noise_0.1_SmallUNet'
        noise_level = 0.1
        learning_rate = 0.001

    pp = exp2()
    pp.train(500)
    pp.end()

# Experiment 6.0 iterative scheme with noise level 0.1, fully convolutional nn, LUNA data set
if number == 6:
    print('Run iterative scheme, high noise, standard convNet')
    class exp3(iterative_scheme):
        experiment_name = 'Noise_0.1_'
        noise_level = 0.1
        learning_rate = 0.0003

    it = exp3()
    it.train(500)
    it.end()
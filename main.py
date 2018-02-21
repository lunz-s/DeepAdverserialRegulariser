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

# Experiment 1.0: AR with noise level 0.01, standard classifier network, LUNA data set
if 1:
    class exp1(adversarial_regulariser):
        experiment_name = 'Noise_0.01_StandardNet'
        noise_level = 0.01
        mu_default = 1.5
        learning_rate = 0.0003
        step_size = 0.1
        total_steps = 30

    adv_reg = exp1()
    adv_reg.evaluate_image_optimization(step_s=0.1, starting_point='FBP')
    adv_reg.evaluate_image_optimization(step_s=0.15, starting_point='FBP')
    adv_reg.evaluate_image_optimization(step_s=0.2, starting_point='FBP')
    adv_reg.pretrain_Wasser_FBP(10)
    adv_reg.pretrain_Wasser_DataMinimizer(10)
    adv_reg.train(10)
    adv_reg.end()

# Experiment 2.0 post-processing with noise level 0.01, standard UNet, LUNA data set
if 1:
    class exp2(postprocessing):
        experiment_name = 'Noise_0.01_SmallUNet'
        noise_level = 0.01
        learning_rate = 0.001

    pp = exp2()
    pp.train(10)

# Experiment 3.0 iterative scheme with noise level 0.01, fully convolutional ne, LUNA data set
if 1:
    class exp3(iterative_scheme):
        experiment_name = 'Noise_0.01_'
        noise_level = 0.01

    it = exp3()
    it.train(10)

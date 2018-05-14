from util import quality
from data_pips import ellipses
from data_pips import LUNA
from data_pips import BSDS

import numpy as np
import util as ut
from skimage.measure import compare_ssim as ssim

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

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

nl_el = 0.02

### Comparison experiments: Standard architecture
class ar(adversarial_regulariser):
    experiment_name = 'ConvNet'
    noise_level = nl_el
    mu_default = .2
    learning_rate = 0.0001
    step_size = 1
    total_steps_default = 25
    default_sampling_pattern = 'startend'

    def get_network(self, size, colors):
        return improved_binary_classifier(size=size, colors=colors)

    def unreg_mini(self, y, fbp):
        return self.update_pic(15, 1, y, fbp, 0)

    def get_Data_pip(self):
        return ellipses()


class tv(total_variation):
    experiment_name = 'Standard'
    noise_level = nl_el

    def get_Data_pip(self):
        return ellipses()


class pp(postprocessing):
    experiment_name = 'Standard'
    noise_level = nl_el
    def_lambda = 0.0007

    def get_Data_pip(self):
        return ellipses()


n = input('exp: ')

if n == 1:
    # create object of type experiment1
    adv_reg = ar()
    adv_reg.set_total_steps(30)
    # adv_reg.find_good_lambda()
    for k in range(5):
        adv_reg.pretrain_Wasser_DataMinimizer(500)
    adv_reg.evaluate_image_optimization(steps=70)

if n == 2:
    recon = pp()
    print(recon.noise_level)
    for k in range(5):
        recon.train(500)

if n == 3:
    tv = tv()
    print(tv.noise_level)
    lmb = []
    for k in range(10):
        lmb.append(0.0002 * (k + 1))
    tv.find_TV_lambda(lmb)

if n == 4:
    batch_size = 32
    ar = ar()
    y, x_true, fbp = ar.generate_training_data(batch_size=batch_size, training_data=False)
    ar_results = ar.evaluate(y, fbp)
    for res in ar_results:
        print('AR: ' + str(quality(x_true, res)))
    ar.end()
    pp = pp()
    pp_results = pp.evaluate(y, fbp)
    print('PP: ' + str(quality(x_true, pp_results)))
    pp.end()
    tv = tv()
    tv_results = tv.evaluate(y, fbp)
    print('TV: ' + str(quality(x_true, tv_results)))
    tv.end()
    print('FBP: ' + str(quality(x_true, fbp)))

    for k in range(10):
        plt.figure()
        plt.subplot(151)
        plt.imshow(ut.cut_image(x_true[k, ..., 0]), cmap='Greys')
        plt.axis('off')
        plt.title('Ground_truth')
        plt.subplot(152)
        plt.imshow(ut.cut_image(fbp[k, ..., 0]), cmap='Greys')
        plt.axis('off')
        plt.title('FBP')
        plt.subplot(153)
        plt.imshow(ut.cut_image(pp_results[k, ..., 0]), cmap='Greys')
        plt.title('PostProcessing')
        plt.axis('off')
        plt.subplot(154)
        plt.imshow(ut.cut_image((ar_results[20])[k, ..., 0]), cmap='Greys')
        plt.title('Adv. Reg')
        plt.axis('off')
        plt.subplot(155)
        plt.imshow(ut.cut_image(tv_results[k, ..., 0]), cmap='Greys')
        plt.title('TV')
        plt.axis('off')
        path = '/local/scratch/public/sl767/DeepAdversarialRegulariser/Saves/Computed_Tomography/ellipses/Comparison/'
        ut.create_single_folder(path)
        plt.savefig(path + str(k) + '.png')
        plt.close()

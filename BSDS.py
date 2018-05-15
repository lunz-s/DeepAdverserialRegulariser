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

nl = 0.1

def quality(truth, recon):
    recon = ut.cut_image(recon)
    l2 = np.average(np.sqrt(np.sum(np.square(truth - recon), axis = (1,2,3))))
    psnr = - 10 * np.log10(np.average(np.square(truth - recon)))
    amount_images = truth.shape[0]
    ssi = 0
    for j in range(3):
        for k in range(amount_images):
            ssi = ssi + ssim(truth[k,...,j], ut.cut_image(recon[k,...,j]),)
    ssi = ssi/amount_images
    ssi = ssi/3
    return [l2, psnr, ssi]


### Comparison experiments: Standard architecture
class ar(adversarial_regulariser):
    experiment_name = 'ConvNet'
    noise_level = nl
    mu_default = 50
    learning_rate = 0.0001
    step_size = .01
    total_steps_default = 150
    default_sampling_pattern = 'startend'

    def get_network(self, size, colors):
        return improved_binary_classifier(size=size, colors=colors)

    def unreg_mini(self, y, fbp):
        return self.update_pic(0, 1, y, fbp, 0)

    def get_Data_pip(self):
        return BSDS()

    def get_model(self, size):
        return denoising(size=size)


class tv(total_variation):
    experiment_name = 'Standard'
    noise_level = nl
    def_lambda = 0.1

    def get_Data_pip(self):
        return BSDS()

    def get_model(self, size):
        return denoising(size=size)


class pp(postprocessing):
    experiment_name = 'Standard'
    noise_level = nl
    def_lambda = 0.0007

    def get_Data_pip(self):
        return BSDS()

    def get_model(self, size):
        return denoising(size=size)


n = input('exp: ')

if n == 1:
    # create object of type experiment1
    adv_reg = ar()
    adv_reg.set_total_steps(30)
    # adv_reg.find_good_lambda()
    for k in range(5):
        adv_reg.pretrain_Wasser_FBP(500)
    adv_reg.evaluate_image_optimization(steps=70)

if n == 1.1:
    adv_reg = ar()
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
        lmb.append(3 ** (k -10))
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
        plt.imshow(ut.cut_image(x_true[k, ...]),vmin=0, vmax=1)
        plt.axis('off')
        plt.title('Ground_truth')
        plt.subplot(152)
        plt.imshow(ut.cut_image(fbp[k, ...]),vmin=0, vmax=1)
        plt.axis('off')
        plt.title('FBP')
        plt.subplot(153)
        plt.imshow(ut.cut_image(pp_results[k, ...]),vmin=0, vmax=1)
        plt.title('PostProcessing')
        plt.axis('off')
        plt.subplot(154)
        plt.imshow(ut.cut_image((ar_results[25])[k, ...]),vmin=0, vmax=1)
        plt.title('Adv. Reg')
        plt.axis('off')
        plt.subplot(155)
        plt.imshow(ut.cut_image(tv_results[k, ...]),vmin=0, vmax=1)
        plt.title('TV')
        plt.axis('off')
        path = '/local/scratch/public/sl767/DeepAdversarialRegulariser/Saves/Denoising/BSDS/Comparison/'
        ut.create_single_folder(path)
        plt.savefig(path + str(k) + '.png')
        plt.close()


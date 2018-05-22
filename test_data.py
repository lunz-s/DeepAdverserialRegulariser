import matplotlib.pyplot as plt
import numpy as np

from forward_models import ct
from forward_models import denoising

from data_pips import BSDS

from Framework import adversarial_regulariser
from Framework import positiv_adversarial_regulariser
from Framework import postprocessing
from Framework import iterative_scheme
from Framework import total_variation
from Framework import generic_framework
from Framework import total_variation


def l2(pic):
    return np.sqrt(np.sum(np.square(pic)))


data_gen = generic_framework()
for i in range(32):
    image = data_gen.data_pip.load_data(training_data=False)
    print(image.shape)
    data = data_gen.model.forward_operator(image[..., 0])

    # add white Gaussian noise
    noisy_data = data + np.random.normal(size=data_gen.measurement_space) * 0.02

    # percentual noise:
    norm = l2(data)
    error = l2(noisy_data - data)
    perc = error / norm

    fbp = data_gen.model.inverse(noisy_data)
    norm_fbp = l2(fbp)
    error_fbp = l2(fbp - image[...,0])
    perc_fbp = error_fbp / norm_fbp

    print('Data: {}, Noise: {}, perc: {}'.format(norm, error, perc))
    print('FBP: {}, Noise: {}, perc: {}'.format(norm_fbp, error_fbp, perc_fbp))
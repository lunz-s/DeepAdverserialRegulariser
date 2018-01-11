import AR_for_denoising as ar
import numpy as np

denoiser = ar.Denoiser1()

# testing sequence to check methods
if 0:
    denoiser.evaluate_Network(0.5)
    denoiser.create_optimized_images(32)
    denoiser.pretrain_Wasser_ini(2)
    denoiser.train(2, 30)

# pretraining
if 0:
    for k in range(5):
        denoiser.pretrain_Wasser_ini(500)

# try out different regularisation parameters
if 1:
    denoiser.find_noise_level()
    denoiser.find_good_lambda()
    denoiser.create_optimized_images(64, mu = 10)
    denoiser.create_optimized_images(64, mu=16)
    denoiser.create_optimized_images(64, mu=20)

# iterative training
if 0:
    for k in range(5):
        denoiser.train(200, 7)

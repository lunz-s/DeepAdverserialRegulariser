from iterative_denoising import stacked_denoiser
from iterative_denoising import bregmann_denoiser


if 1:
    sd = stacked_denoiser(2, mu=[44,5])
    sd.independant_layer(1, 10)
    sd.independant_layer(1, 15)
    sd.independant_layer(1, 25)
    sd.end()

if 0:
    sd = bregmann_denoiser(2, mu=[15, 15])
    sd.train_layer(1, 500)
    sd.independant_layer(1, 10)
    sd.independant_layer(1, 20)
    sd.independant_layer(1,30)
    sd.end()

from iterative_denoising import stacked_denoiser
from iterative_denoising import bregmann_denoiser


if 0:
    sd = stacked_denoiser(2, mu=[44,5])
    sd.train_layer(1, 500)
    sd.independant_layer(1, 10)
    sd.independant_layer(1, 15)
    sd.independant_layer(1, 25)
    sd.end()

if 1:
    sd = bregmann_denoiser(3, mu=[25, 25, 25])
    sd.train_layer(2, 500)
    sd.end()

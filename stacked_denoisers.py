from iterative_denoising import stacked_denoiser

sd = stacked_denoiser(2)
sd.train_layer(0, 500)
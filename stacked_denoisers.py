from iterative_denoising import stacked_denoiser

sd = stacked_denoiser(2)
# sd.train_layer(1, 500)

sd.track_layer(1, 20)
sd.track_layer(1, 50)
sd.track_layer(1, 80)
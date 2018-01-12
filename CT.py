import AR_for_CT as ar
import numpy as np

if 0:
    recon = ar.Recon1()
    # recon.pretrain_Wasser_FBP(500)
    # recon.pretrain_Wasser_DataMinimizer(500)
    # recon.find_good_lambda()
    # recon.find_noise_level()
    # recon.create_optimized_images(1, mu=2.5, steps=30, starting_point='Mini')
    #recon.create_optimized_images(64, mu=2, step_s=0.1, steps=200, starting_point='Mini')
    if 0:
        for k in range(2):
            recon.train(500, 250, starting_point='Mini')


    if 0:
        starting_point = 'Mini'
        recon.create_optimized_images(512, mu=2, step_s=0.1, steps=250, starting_point=starting_point)

if 1:
    recon = ar.Recon_LUNA()
    for k in range(4):
        recon.train(500, 125, starting_point='Mini')





import AR_for_denoising as ar
import postprocessing as pp
import numpy as np

def compare_methods(amount_test_data):
    denoiser = ar.Denoiser1()
    true, cor = denoiser.generate_images(amount_test_data, training_data=False)
    results = {}
    results['Adversarial Regulariser'] = denoiser.evaluate_AR(cor)
    results['TV'] = denoiser.evaluate_TV(cor)
    denoiser.end()
    post = pp.postDenoising()
    results['Post-Processing'] = post.evaluate_pp(true, cor)
    for methode, res in results.items():
        error = np.mean(np.sqrt(np.sum(np.square(true - res), axis=(1,2,3))))
        print('Methode: ' + methode + ', MSE: ' + str(error))


compare_methods(256)

# train postprocessing
if 0:
    postpro = pp.postDenoising()  #
    postpro.train(200)
    postpro.end()

if 0:
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

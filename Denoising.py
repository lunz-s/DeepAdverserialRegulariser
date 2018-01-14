import AR_for_denoising as ar
import postprocessing as pp
import numpy as np

def compare_methods(amount_test_data):
    denoiser = ar.Denoiser2()
    true, cor = denoiser.generate_images(amount_test_data, training_data=False)
    results = {}
    results['Adversarial Regulariser'] = denoiser.evaluate_AR(cor)
    results['TV'] = denoiser.evaluate_TV(cor)
    denoiser.end()
    post = pp.postDenoising2()
    results['Post-Processing'] = post.evaluate_pp(true, cor)
    for methode, res in results.items():
        error = np.mean(np.sqrt(np.sum(np.square(true - res), axis=(1,2,3))))
        print('Methode: ' + methode + ', MSE: ' + str(error))

def visual_comparison():
    pass
compare_methods(64)

if 0:
    denoiser = ar.Denoiser2()
    lmb = []
    for k in range(10):
        lmb.append(3**(k-10))
    denoiser.find_TV_lambda(lmb)

# train postprocessing
if 0:
    postpro = pp.postDenoising2()  #
    for k in range(5):
        postpro.train(300)
    postpro.end()

if 0:
    denoiser = ar.Denoiser2()

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
        denoiser.create_optimized_images(64, mu = 20)
        denoiser.create_optimized_images(64, mu=40)
        denoiser.create_optimized_images(64, mu=60)

    # iterative training
    if 0:
        for k in range(5):
            denoiser.train(200, 7)

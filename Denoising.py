import AR_for_denoising as ar
import postprocessing as pp
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

def visual_comparison(k):
    denoiser = ar.Denoiser2()
    true, cor = denoiser.generate_images(1, training_data=False)
    advR = denoiser.evaluate_AR(cor)
    tv = denoiser.evaluate_TV(cor)
    denoiser.end()
    post = pp.postDenoising2()
    postP = post.evaluate_pp(true, cor)
    post.end()
    plt.figure()
    plt.subplot(2,3,1)
    plt.imshow(true[0,...])
    plt.axis('off')
    plt.title('Original')
    plt.subplot(2,3,2)
    plt.imshow(cor[0,...])
    plt.axis('off')
    plt.title('Noisy')
    plt.subplot(2,3,4)
    plt.imshow(tv[0,...])
    plt.axis('off')
    plt.title('TV')
    plt.subplot(2,3,5)
    plt.imshow(postP[0,...])
    plt.axis('off')
    plt.title('PostProc.')
    plt.subplot(2,3,1)
    plt.imshow(advR[0,...])
    plt.axis('off')
    plt.title('Adv. Reg.')
    plt.savefig('Saves/Evaluations/' + str(k) + '.png')
    plt.close()


# compare_methods(64)

if 1:
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

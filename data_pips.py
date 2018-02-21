import random
import numpy as np
import scipy.ndimage
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import platform
import odl
import dicom as dc
from scipy.misc import imresize
import util as ut

# Objects of the data_pip type require to admit a method load_data that
# returns a single image from the dataset the method is linked to, in the
# format image_size.
# The image is always normalized between (0,1).
class data_pip(object):
    image_size = (128,128)
    name = 'default'
    colors = 1

    # load data outputs single image in format (image_size, colors)
    def load_data(self, training_data= True):
        pass

# returns 128x128 image from the BSDS dataset
class BSDS(data_pip):
    name = 'BSDS'
    colors = 3

    def __init__(self):
        # set up the training data file system
        self.train_list = ut.find('*.jpg', './BSDS/images/train')
        self.train_amount = len(self.train_list)
        print('Training Pictures found: ' + str(self.train_amount))
        self.eval_list = ut.find('*.jpg', './BSDS/images/val')
        self.eval_amount = len(self.eval_list)
        print('Evaluation Pictures found: ' + str(self.eval_amount))

    # method to draw raw picture samples
    def single_image(self, training_data=True):
        if training_data:
            rand = random.randint(0, self.train_amount - 1)
            pic = mpimg.imread(self.train_list[rand])
        else:
            rand = random.randint(0, self.eval_amount - 1)
            pic = scipy.ndimage.imread(self.eval_list[rand])
        return pic/255.0

    # Draw random edgepoint
    def edgepoint(self, x_size, y_size):
        x_vary = x_size - self.image_size[0]
        x_coor = random.randint(0, x_vary)
        y_vary = y_size - self.image_size[1]
        y_coor = random.randint(0, y_vary)
        upper_left = [x_coor, y_coor]
        lower_right = [x_coor + self.image_size[0], y_coor + self.image_size[1]]
        return upper_left, lower_right

    # methode to cut a image_size area out of the training images
    def load_data(self, training_data= True):
        pic = self.single_image(training_data=training_data)
        size = pic.shape
        ul, lr = self.edgepoint(size[0], size[1])
        image = pic[ul[0]:lr[0], ul[1]:lr[1],:]
        return image

# returns 128x128 image from the LUNA challenge dataset
class LUNA(data_pip):
    name = 'LUNA'
    colors = 1

    def __init__(self):
        name = platform.node()
        Train_Path = ''
        Eval_Path = ''
        if name == 'LAPTOP-E6AJ1CPF':
            Train_Path = './LUNA/Train_Data'
            Eval_Path = './LUNA/Eval_Data'
        elif name == 'motel':
            Train_Path = '/local/scratch/public/sl767/LUNA/Training_Data'
            Eval_Path = '/local/scratch/public/sl767/LUNA/Evaluation_Data'
        # List the existing training data
        self.training_list = ut.find('*.dcm', Train_Path)
        self.training_list_length = len(self.training_list)
        print('Training Data found: ' + str(self.training_list_length))
        self.eval_list = ut.find('*.dcm', Eval_Path)
        self.eval_list_length = len(self.eval_list)
        print('Evaluation Data found: ' + str(self.eval_list_length))

    # methodes for obtaining the medical data
    def get_random_path(self, training_data= True):
        if training_data:
            path = self.training_list[random.randint(0, self.training_list_length-1)]
        else:
            path = self.eval_list[random.randint(0, self.eval_list_length - 1)]
        return path

    # resizes image to format 128x128
    def reshape_pic(self, pic):
        pic = ut.normalize_image(pic)
        pic = imresize(pic, [128, 128])
        pic = ut.scale_to_unit_intervall(pic)
        return pic

    # the data method
    def load_data(self, training_data= True):
        k = -10000
        pic = np.zeros((128,128))
        while k < 0:
            try:
                path = self.get_random_path(training_data=training_data)
                dc_file = dc.read_file(path)
                pic = dc_file.pixel_array
                if pic.shape == (512,512):
                    pic = self.reshape_pic(pic)
                    k = 1
            except UnboundLocalError:
                k = - 10000
                print('UnboundLocalError caught')
        output = np.zeros((128,128,1))
        output[...,0] = pic
        return output

# returns 128x128 image of randomly sampled ellipses
class ellipses(data_pip):
    name = 'ellipses'
    colors = 1

    def __init__(self):
        self.space = odl.uniform_discr([-64, -64], [64, 64], [self.image_size[0], self.image_size[1]],
                                  dtype='float32')

    # generates one random ellipse
    def random_ellipse(self, interior=False):
        if interior:
            x_0 = np.random.rand() - 0.5
            y_0 = np.random.rand() - 0.5
        else:
            x_0 = 2 * np.random.rand() - 1.0
            y_0 = 2 * np.random.rand() - 1.0

        return ((np.random.rand() - 0.5) * np.random.exponential(0.4),
                np.random.exponential() * 0.2, np.random.exponential() * 0.2,
                x_0, y_0,
                np.random.rand() * 2 * np.pi)

    # generates odl space object with ellipses
    def random_phantom(self, spc, n_ellipse=50, interior=False):
        n = np.random.poisson(n_ellipse)
        ellipses = [self.random_ellipse(interior=interior) for _ in range(n)]
        return odl.phantom.ellipsoid_phantom(spc, ellipses)

    def load_data(self, training_data= True):
        pic = self.random_phantom(spc= self.space)
        output = np.zeros((128, 128, 1))
        output[..., 0] = ut.scale_to_unit_intervall(pic)
        return output

# methode to test the data generation pipelines
if __name__ == '__main__':
    models = []
    models.append(BSDS())
    models.append(LUNA())
    models.append(ellipses())
    plt.figure()
    amount_models = len(models)
    for k in range(amount_models):
        plt.subplot(1, amount_models, k+1)
        image = models[k].load_data()
        plt.imshow(image)
        max = round(np.amax(image), 2)
        min = round(np.amin(image), 2)
        av = round(np.average(image), 2)
        plt.title('Max: {} \n Min: {} \n Av.: {}'.format(max, min, av))
        plt.axis('off')
    plt.savefig('Saves/Test/test_preprocessing.jpg')
    plt.close()

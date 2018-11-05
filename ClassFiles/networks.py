import tensorflow as tf
from ClassFiles.util import lrelu
from ClassFiles import util as ut
from abc import ABC, abstractmethod

# The format the networks for reconstruction are written in. Size in format (width, height) gives the shape
# of an input image. colors specifies the amount of output channels for image to image architectures.
class network(ABC):
    def __init__(self, size, colors):
        self.size = size
        self.colors = colors

    # Method defining the neural network architecture, returns computation result. Use reuse=tf.AUTO_REUSE.
    @abstractmethod
    def net(self, input):
        pass

### basic network architectures ###
# A couple of small network architectures for computationally light comparison experiments.
# No dropout, no batch_norm, no skip-connections


class MultiscaleL1Classifier(network):
    # uses strided convolutions for a multiscale classifier

    def net(self, input):
        # fine scale
        loc1 = tf.layers.conv2d(inputs=input, filters=16, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=tf.AUTO_REUSE, name='loc1')
        loc2 = tf.layers.conv2d(inputs= loc1, filters=16, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=tf.AUTO_REUSE, name='loc2')
        loc3 = tf.layers.conv2d(inputs= loc2, filters=16, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=tf.AUTO_REUSE, name='loc3')
        loc_l1 = ut.image_l1(loc3)

        # medium scale
        med1 = tf.layers.conv2d(inputs=input, filters=16, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=tf.AUTO_REUSE, name='med1')
        med2 = tf.layers.conv2d(inputs= med1, filters=16, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=tf.AUTO_REUSE, name='med2')
        med3 = ut.dilated_conv_layer(inputs= med2, name='med3', filters=16, kernel_size=[5, 5], padding="SAME",
                                 activation=lrelu, reuse=tf.AUTO_REUSE, rate=4)
        med_l1 = ut.image_l1(med3)

        # global scale
        glob1 = tf.layers.conv2d(inputs=input, filters=16, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=tf.AUTO_REUSE, name='glob1')
        glob2 = ut.dilated_conv_layer(inputs= glob1, name='glob2', filters=16, kernel_size=[5, 5], padding="SAME",
                                 activation=lrelu, reuse=tf.AUTO_REUSE, rate = 4)
        glob3 = ut.dilated_conv_layer(inputs= glob2, name='glob3', filters=16, kernel_size=[5, 5], padding="SAME",
                                 activation=lrelu, reuse=tf.AUTO_REUSE, rate=24)
        glob_l1 = ut.image_l1(glob3)

        # linear classifier on l1 norms
        results = tf.concat([loc_l1, med_l1, glob_l1], axis=1)
        dense = tf.layers.dense(inputs = results, units = 48, activation=lrelu, reuse=tf.AUTO_REUSE, name='dense1')
        output = tf.layers.dense(inputs=dense, units=1, reuse=tf.AUTO_REUSE, name='dense2')

        return output


class LocalClassifier(network):
    # convolutional layers followed by global average pooling only

    def net(self, input):
        # convolutional network for feature extraction
        conv1 = tf.layers.conv2d(inputs=input, filters=32, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=tf.AUTO_REUSE, name='conv1')

        conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[3, 3], padding="same",
                                 activation=lrelu, reuse=tf.AUTO_REUSE, name='conv2')

        conv3 = tf.layers.conv2d(inputs=conv2, filters=32, kernel_size=[3, 3], padding="same",
                                 activation=lrelu, reuse=tf.AUTO_REUSE, name='conv3')

        conv4 = tf.layers.conv2d(inputs=conv3, filters=32, kernel_size=[3, 3], padding="same",
                                 reuse=tf.AUTO_REUSE, name='conv4')

        output = tf.reduce_mean(conv4, axis=(1,2,3))
        tf.expand_dims(output, axis=1)

        # Output network results
        return output


class ConvNetClassifier(network):
    # classical classifier with convolutional layers with strided convolutions and two dense layers at the end

    def net(self, input):
        # convolutional network for feature extraction
        conv1 = tf.layers.conv2d(inputs=input, filters=16, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=tf.AUTO_REUSE, name='conv1')
        conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=tf.AUTO_REUSE, name='conv2')
        conv3 = tf.layers.conv2d(inputs=conv2, filters=32, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=tf.AUTO_REUSE, name='conv3', strides=2)
        # image size is now size/2
        conv4 = tf.layers.conv2d(inputs=conv3, filters=64, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=tf.AUTO_REUSE, name='conv4', strides=2)
        # image size is now size/4
        conv5 = tf.layers.conv2d(inputs=conv4, filters=64, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=tf.AUTO_REUSE, name='conv5', strides=2)
        # image size is now size/8
        conv6 = tf.layers.conv2d(inputs=conv5, filters=128, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=tf.AUTO_REUSE, name='conv6', strides=2)

        # reshape for classification - assumes image size is multiple of 32
        finishing_size = int(self.size[0] * self.size[1]/(16*16))
        dimensionality = finishing_size * 128
        reshaped = tf.reshape(conv6, [-1, dimensionality])

        # dense layer for classification
        dense = tf.layers.dense(inputs=reshaped, units=256, activation=lrelu, reuse=tf.AUTO_REUSE, name='dense1')
        output = tf.layers.dense(inputs=dense, units=1, reuse=tf.AUTO_REUSE, name='dense2')

        # Output network results
        return output


class PrunedUNet(network):
    # a small version of the UNet architecture for image-to-image computations

    def net(self, input):
        conv1 = tf.layers.conv2d(inputs=input, filters=32, kernel_size=[5, 5],
                                 padding="same", name='conv1', reuse=tf.AUTO_REUSE, activation=tf.nn.relu)
        # 64
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5],
                                 padding="same", name='conv2', reuse=tf.AUTO_REUSE, activation=tf.nn.relu)
        # 32
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        conv3 = tf.layers.conv2d(inputs=pool2, filters=64, kernel_size=[3, 3],
                                 padding="same", name='conv3', reuse=tf.AUTO_REUSE, activation=tf.nn.relu)
        # 64
        conv4 = tf.layers.conv2d_transpose(inputs=conv3, filters=32, kernel_size=[5, 5],
                                           strides=(2, 2), padding="same", name='deconv1',
                                           reuse=tf.AUTO_REUSE, activation=tf.nn.relu)
        concat1 = tf.concat([conv4, pool1], axis=3)
        # 128
        conv5 = tf.layers.conv2d_transpose(inputs=concat1, filters=32, kernel_size=[5, 5],
                                           strides=(2, 2), padding="same", name='deconv2',
                                           reuse=tf.AUTO_REUSE, activation=tf.nn.relu)
        concat2 = tf.concat([conv5, input], axis=3)
        output = tf.layers.conv2d(inputs=concat2, filters=self.colors, kernel_size=[5, 5],
                                  padding="same",name='deconv3',
                                  reuse=tf.AUTO_REUSE,  activation=tf.nn.relu)
        return output


class FullyConvolutional(network):
    # a stack of convolutional layers for image-to-image functionality

    def net(self, input):
        # 128
        conv1 = tf.layers.conv2d(inputs=input, filters=32, kernel_size=[5, 5],
                                 padding="same", name='conv1', reuse=tf.AUTO_REUSE, activation=tf.nn.relu)
        # 64
        conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[5, 5],
                                 padding="same", name='conv2', reuse=tf.AUTO_REUSE, activation=tf.nn.relu)
        # 32
        conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=[3, 3],
                                 padding="same", name='conv3', reuse=tf.AUTO_REUSE, activation=tf.nn.relu)
        # 64
        conv4 = tf.layers.conv2d(inputs=conv3, filters=32, kernel_size=[5, 5],
                                 padding="same", name='conv4',reuse=tf.AUTO_REUSE, activation=tf.nn.relu)
        # 128
        conv5 = tf.layers.conv2d(inputs=conv4, filters=32, kernel_size=[5, 5],
                                 padding="same", name='conv5', reuse=tf.AUTO_REUSE, activation=tf.nn.relu)
        output = tf.layers.conv2d(inputs=conv5, filters=self.colors, kernel_size=[5, 5],
                                  padding="same", name='conv6', reuse=tf.AUTO_REUSE)
        return output

### deeper network architectures ###
# Skip-connections. No batch-norm, as common for adversarial nets.

# A ResNet architecture, used as adversarial network
def apply_conv(x, filters=32, kernel_size=3, strides=1):
    return tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, padding='SAME', strides=strides,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                            activation=lrelu)


def resblock(x, filters):
    with tf.name_scope('resblock'):
        x = tf.identity(x)
        update = apply_conv(x, filters=filters)
        update = apply_conv(update, filters=filters)

        skip = tf.layers.conv2d(x, filters=filters, kernel_size=1, padding='SAME',
                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        return skip + update


def resblock_downsample(x, filters):
    with tf.name_scope('resblock'):
        x = tf.identity(x)
        update = apply_conv(x, filters=filters)
        strided = apply_conv(update, filters=filters, strides=2)

        skip = tf.layers.conv2d(x, filters=filters, kernel_size=1, padding='SAME', strides=2,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        return skip + strided


class ResnetClassifier(network):
    # A residual network with a singel neuron as output

    def net(self, input):
        with tf.name_scope('pre_process'):
            x = apply_conv(input, filters=16, kernel_size=3)

        with tf.name_scope('Level1'):
            x = resblock(x, 16)

        with tf.name_scope('Level2'):
            x = resblock_downsample(x, filters=32) # 1/2

        with tf.name_scope('Level3'):
            x = resblock_downsample(x, filters=32) # 1/4

        with tf.name_scope('Level4'):
            x = resblock_downsample(x, filters=32)  # 1/8

        with tf.name_scope('Level5'):
            x = resblock_downsample(x, filters=32)  # 1/16

        with tf.name_scope('post_process'):
            flat = tf.contrib.layers.flatten(x)
            flat = tf.layers.dense(flat, 1)

        return flat

# The UNet architecture for denoising
def downsampling_block(tensor, name, filters, kernel = (5,5)):
    # applies strided convolution, sampling down by factor 2
    with tf.variable_scope(name):
        conv1 = tf.layers.conv2d(inputs=tensor, filters=filters, kernel_size=kernel,
                                          padding="same", name='conv1', reuse=tf.AUTO_REUSE, activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(inputs=conv1, filters=filters, kernel_size=kernel,
                                          padding="same", name='conv2', reuse=tf.AUTO_REUSE, activation=tf.nn.relu)
        pool = tf.layers.average_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        return pool


def upsampling_block(tensor, name, filters, kernel = (5,5)):
    # applies deconvolution, sampling up by factor 2
    with tf.variable_scope(name):
        conv1 = tf.layers.conv2d(inputs=tensor, filters=filters, kernel_size=kernel,
                                          padding="same", name='conv1', reuse=tf.AUTO_REUSE, activation=tf.nn.relu)
        upsample = tf.layers.conv2d_transpose(inputs=conv1, filters=filters, kernel_size=[5, 5],
                                           strides= (2,2), padding="same", name='deconv1',
                                           reuse=tf.AUTO_REUSE, activation=tf.nn.relu)
        return upsample


class UNet(network):
    # Unet, totally sampling down by factor 16 at the maximum, 64 channels
    def net(self, input):
        # same shape conv
        pre1 = tf.layers.conv2d(inputs=input, filters=16, kernel_size=[5, 5],
                                padding="same", name='pre1', reuse=tf.AUTO_REUSE, activation=tf.nn.relu)
        # downsampling 1
        down1 = downsampling_block(tensor=pre1, name='down1', filters=32)

        # downsampling 2
        down2 = downsampling_block(tensor=down1, name='down2', filters=64)

        # downsampling 3
        down3 = downsampling_block(tensor=down2, name='down3', filters=64)

        # downsampling 4
        down4 = downsampling_block(tensor=down3, name='down4', filters=128)

        # upsampling 1
        up1 = upsampling_block(tensor=down4, name='up1', filters=64)
        con1 = tf.concat([up1, down3], axis=3)

        # upsampling 2
        up2 = upsampling_block(tensor=con1, name='up2', filters=64)
        con2 = tf.concat([up2, down2], axis=3)

        # upsampling 3
        up3 = upsampling_block(tensor=con2, name='up3', filters=64)
        con3 = tf.concat([up3, down1], axis=3)

        # upsampling 4
        up4 = upsampling_block(tensor=con3, name='up4', filters=32)
        con4 = tf.concat([up4, pre1], axis=3)

        post1 = tf.layers.conv2d(inputs=con4, filters=16, kernel_size=[5, 5],
                                 padding="same", name='post1',
                                 reuse=tf.AUTO_REUSE, activation=tf.nn.relu)
        output = tf.layers.conv2d(inputs=post1, filters=self.colors, kernel_size=[5, 5],
                                 padding="same", name='post2',
                                 reuse=tf.AUTO_REUSE)

        return output

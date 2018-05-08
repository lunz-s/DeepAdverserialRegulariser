import tensorflow as tf
from util import lrelu
import util as ut

class dilated_con_classifier(object):
    def net(self, input):
        pass

class multiscale_l1_classifier(object):
    def __init__(self, size, colors):
        self.size = size
        self.reuse = False
        self.colors = colors

    def net(self, input):
        # fine scale
        loc1 = tf.layers.conv2d(inputs=input, filters=16, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=self.reuse, name='loc1')
        loc2 = tf.layers.conv2d(inputs= loc1, filters=16, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=self.reuse, name='loc2')
        loc3 = tf.layers.conv2d(inputs= loc2, filters=16, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=self.reuse, name='loc3')
        loc_l1 = ut.image_l1(loc3)

        # medium scale
        med1 = tf.layers.conv2d(inputs=input, filters=16, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=self.reuse, name='med1')
        med2 = tf.layers.conv2d(inputs= med1, filters=16, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=self.reuse, name='med2')
        med3 = ut.dilated_conv_layer(inputs= med2, name='med3', filters=16, kernel_size=[5, 5], padding="SAME",
                                 activation=lrelu, reuse=self.reuse, rate=4)
        med_l1 = ut.image_l1(med3)

        # global scale
        glob1 = tf.layers.conv2d(inputs=input, filters=16, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=self.reuse, name='glob1')
        glob2 = ut.dilated_conv_layer(inputs= glob1, name='glob2', filters=16, kernel_size=[5, 5], padding="SAME",
                                 activation=lrelu, reuse=self.reuse, rate = 4)
        glob3 = ut.dilated_conv_layer(inputs= glob2, name='glob3', filters=16, kernel_size=[5, 5], padding="SAME",
                                 activation=lrelu, reuse=self.reuse, rate=24)
        glob_l1 = ut.image_l1(glob3)

        # linear classifier on l1 norms
        results = tf.concat([loc_l1, med_l1, glob_l1], axis=1)
        dense = tf.layers.dense(inputs = results, units = 48, activation=lrelu, reuse=self.reuse, name='dense1')
        output = tf.layers.dense(inputs=dense, units=1, reuse=self.reuse, name='dense2')

        # change reuse variable for next call of network method
        self.reuse = True

        return output

class binary_classifier(object):
    def __init__(self, size, colors):
        self.size = size
        self.reuse = False
        self.colors = colors

    def net(self, input):
        # convolutional network for feature extraction
        conv1 = tf.layers.conv2d(inputs=input, filters=16, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=self.reuse, name='conv1')
        # begin convolutional/pooling architecture
        conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=self.reuse, name='conv2')
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        # image size is now size/2
        conv3 = tf.layers.conv2d(inputs=pool2, filters=32, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=self.reuse, name='conv3')
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
        # image size is now size/4
        conv4 = tf.layers.conv2d(inputs=pool3, filters=64, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=self.reuse, name='conv4')
        pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
        # image size is now size/8
        conv5 = tf.layers.conv2d(inputs=pool4, filters=64, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=self.reuse, name='conv5')
        pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)
        # image size is now size/16
        conv6 = tf.layers.conv2d(inputs=pool5, filters=128, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=self.reuse, name='conv6')

        # reshape for classification - assumes image size is multiple of 32
        finishing_size = int(self.size[0]* self.size[1]/(16*16))
        dimensionality = finishing_size * 128
        reshaped = tf.reshape(conv6, [-1, dimensionality])

        # dense layer for classification
        dense = tf.layers.dense(inputs = reshaped, units = 256, activation=tf.nn.relu, reuse=self.reuse, name='dense1')
        output = tf.layers.dense(inputs=dense, units=1, reuse=self.reuse, name='dense2')

        # change reuse variable for next call of network method
        self.reuse = True

        # Output network results
        return output

class improved_binary_classifier(object):
    def __init__(self, size, colors):
        self.size = size
        self.reuse = False
        self.colors = colors

    def net(self, input):
        # convolutional network for feature extraction
        conv1 = tf.layers.conv2d(inputs=input, filters=16, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=self.reuse, name='conv1')
        conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=self.reuse, name='conv2')
        conv3 = tf.layers.conv2d(inputs=conv2, filters=32, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=self.reuse, name='conv3', strides=2)
        # image size is now size/2
        conv4 = tf.layers.conv2d(inputs=conv3, filters=64, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=self.reuse, name='conv4', strides=2)
        # image size is now size/4
        conv5 = tf.layers.conv2d(inputs=conv4, filters=64, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=self.reuse, name='conv5', strides=2)
        # image size is now size/8
        conv6 = tf.layers.conv2d(inputs=conv5, filters=128, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=self.reuse, name='conv6', strides=2)

        # reshape for classification - assumes image size is multiple of 32
        finishing_size = int(self.size[0]* self.size[1]/(16*16))
        dimensionality = finishing_size * 128
        reshaped = tf.reshape(conv6, [-1, dimensionality])

        # dense layer for classification
        dense = tf.layers.dense(inputs = reshaped, units = 256, activation=lrelu, reuse=self.reuse, name='dense1')
        output = tf.layers.dense(inputs=dense, units=1, reuse=self.reuse, name='dense2')

        # change reuse variable for next call of network method
        self.reuse = True

        # Output network results
        return output

class UNet(object):
    def __init__(self, size, colors, parameter_sharing = True):
        self.colors = colors
        self.size = size
        self.parameter_sharing = parameter_sharing
        self.used = False

    def raw_net(self, input, reuse):
        # 128
        conv1 = tf.layers.conv2d(inputs=input, filters=32, kernel_size=[5, 5],
                                      padding="same", name='conv1', reuse=reuse, activation=tf.nn.relu)
        # 64
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5],
                                      padding="same", name='conv2', reuse=reuse, activation=tf.nn.relu)
        # 32
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        conv3 = tf.layers.conv2d(inputs=pool2, filters=64, kernel_size=[3, 3],
                                      padding="same", name='conv3', reuse=reuse, activation=tf.nn.relu)
        # 64
        conv4 = tf.layers.conv2d_transpose(inputs=conv3, filters=32, kernel_size=[5, 5],
                                           strides= (2,2), padding="same", name='deconv1',
                                           reuse=reuse, activation=tf.nn.relu)
        concat1 = tf.concat([conv4, pool1], axis= 3)
        # 128
        conv5 =  tf.layers.conv2d_transpose(inputs=concat1, filters=32, kernel_size=[5, 5],
                                           strides= (2,2), padding="same", name='deconv2',
                                            reuse=reuse, activation=tf.nn.relu)
        concat2 = tf.concat([conv5, input], axis= 3)
        output = tf.layers.conv2d(inputs=concat2, filters=self.colors, kernel_size=[5, 5],
                                  padding="same",name='deconv3',
                                  reuse=reuse,  activation=tf.nn.relu)
        return output

    def net(self, input):
        output = self.raw_net(input, reuse=self.used)
        if self.parameter_sharing:
            self.used = True
        return output

class fully_convolutional(UNet):

    def raw_net(self, input, reuse):
        # 128
        conv1 = tf.layers.conv2d(inputs=input, filters=32, kernel_size=[5, 5],
                                 padding="same", name='conv1', reuse=reuse, activation=tf.nn.relu)
        # 64
        conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[5, 5],
                                 padding="same", name='conv2', reuse=reuse, activation=tf.nn.relu)
        # 32
        conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=[3, 3],
                                 padding="same", name='conv3', reuse=reuse, activation=tf.nn.relu)
        # 64
        conv4 = tf.layers.conv2d(inputs=conv3, filters=32, kernel_size=[5, 5],
                                 padding="same", name='conv4',reuse=reuse, activation=tf.nn.relu)
        # 128
        conv5 = tf.layers.conv2d(inputs=conv4, filters=32, kernel_size=[5, 5],
                                 padding="same", name='conv5', reuse=reuse, activation=tf.nn.relu)
        output = tf.layers.conv2d(inputs=conv5, filters=self.colors, kernel_size=[5, 5],
                                  padding="same", name='conv6', reuse=reuse)
        return output

### resnet architectures
def apply_conv(x, filters=32, kernel_size=3):
    return tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, padding='SAME',
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                            activation=lrelu)

def resblock(x, filters):
    with tf.name_scope('resblock_bn'):
        x = tf.identity(x)
        update = apply_conv(x, filters=filters)
        update = apply_conv(update, filters=filters)

        skip = tf.layers.conv2d(x, filters=filters, kernel_size=1, padding='SAME',
                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        return skip + update

def meanpool(x):
    with tf.name_scope('meanpool'):
        x = tf.identity(x)
        return tf.add_n([x[:, ::2, ::2, :], x[:, 1::2, ::2, :],
                         x[:, ::2, 1::2, :], x[:, 1::2, 1::2, :]]) / 4.0

class resnet_classifier(object):
    def __init__(self, size, colors):
        self.size = size
        self.reuse = False
        self.colors = colors

    def net(self, input):
        with tf.name_scope('pre_process'):
            x = apply_conv(input, filters=64, kernel_size=3)

        with tf.name_scope('x1'):
            x = resblock(x, 64)

        with tf.name_scope('x2'):
            x = resblock(meanpool(x), filters=64)  # 1/2

        with tf.name_scope('x3'):
            x = resblock(meanpool(x), filters=128)  # 1/4

        with tf.name_scope('x4'):
            x = resblock(meanpool(x), filters=256)  # 1/8

        with tf.name_scope('x5'):
            x = resblock(meanpool(x), filters=256)  # 1/16

        with tf.name_scope('post_process'):
            flat = tf.contrib.layers.flatten(x)
            flat = tf.layers.dense(flat, 1)

        # change reuse variable for next call of network method
        self.reuse = True

        return flat


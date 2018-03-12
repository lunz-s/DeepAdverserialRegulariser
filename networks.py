import tensorflow as tf

class binary_classifier(object):
    def __init__(self, size, colors):
        self.size = size
        self.reuse = False
        self.colors = colors

    def net(self, input):
        # convolutional network for feature extraction
        conv1 = tf.layers.conv2d(inputs=input, filters=16, kernel_size=[5, 5], padding="same",
                                 activation=tf.nn.relu, reuse=self.reuse, name='conv1')
        # begin convolutional/pooling architecture
        conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[5, 5], padding="same",
                                 activation=tf.nn.relu, reuse=self.reuse, name='conv2')
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        # image size is now size/2
        conv3 = tf.layers.conv2d(inputs=pool2, filters=32, kernel_size=[5, 5], padding="same",
                                 activation=tf.nn.relu, reuse=self.reuse, name='conv3')
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
        # image size is now size/4
        conv4 = tf.layers.conv2d(inputs=pool3, filters=64, kernel_size=[5, 5], padding="same",
                                 activation=tf.nn.relu, reuse=self.reuse, name='conv4')
        pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
        # image size is now size/8
        conv5 = tf.layers.conv2d(inputs=pool4, filters=64, kernel_size=[5, 5], padding="same",
                                 activation=tf.nn.relu, reuse=self.reuse, name='conv5')
        pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)
        # image size is now size/16
        conv6 = tf.layers.conv2d(inputs=pool5, filters=128, kernel_size=[5, 5], padding="same",
                                 activation=tf.nn.relu, reuse=self.reuse, name='conv6')

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

import tensorflow as tf

from old import AdverserialRegulariser as ar


def lrelu(x):
    return (tf.nn.relu(x) - 0.1*tf.nn.relu(-x))

class advReg1(ar.Inpainter):
    model_name = 'Layout_strided_conv'

    def get_weights(self):
        con1 = tf.get_variable(name="conv1_ad", shape=[7, 7, 16, 32],
                               initializer=(
                               tf.contrib.layers.xavier_initializer_conv2d(uniform=False, dtype=tf.float32)))
        bias1 = tf.Variable(tf.constant(0.1, shape=[1, 1, 1, 32]), name="bias1_ad")
        con2 = tf.get_variable(name="conv2_ad", shape=[5, 5, 32, 64],
                               initializer=(
                               tf.contrib.layers.xavier_initializer_conv2d(uniform=False, dtype=tf.float32)))
        bias2 = tf.Variable(tf.constant(0.1, shape=[1, 1, 1, 64]), name="bias2_ad")
        con3 = tf.get_variable(name="conv3_ad", shape=[5, 5, 64, 128],
                               initializer=(
                               tf.contrib.layers.xavier_initializer_conv2d(uniform=False, dtype=tf.float32)))
        bias3 = tf.Variable(tf.constant(0.1, shape=[1, 1, 1, 128]), name="bias3_ad")
        logits_W = tf.get_variable(name="logits_W_ad", shape=[128*5*5, 1],
                                   initializer=(
                                   tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32)))
        logits_bias = tf.Variable(tf.constant(0.0, shape=[1, 1]), name='logits_bias_ad')
        con_pre = tf.get_variable(name="conv_pre", shape=[7, 7, 4, 16],
                               initializer=(
                               tf.contrib.layers.xavier_initializer_conv2d(uniform=False, dtype=tf.float32)))
        bias_pre = tf.Variable(tf.constant(0.1, shape=[1, 1, 1, 16]), name="bias_pre")

        return [con1, bias1, con2, bias2, con3, bias3, logits_W, logits_bias, con_pre, bias_pre]


    def adverserial_network(self, weights, input_pic, input_filter):

        input_layer = tf.concat([input_pic, input_filter], axis = 3)

        # convolutional layer without downsamling
        conv_pre = lrelu(tf.nn.conv2d(input_layer, weights[8], strides=[1, 1, 1, 1], padding='SAME') + weights[9])

        # 1st convolutional layer (pic size 40)
        conv1 = lrelu(tf.nn.conv2d(conv_pre, weights[0], strides=[1, 2, 2, 1], padding='SAME') + weights[1])

        # 2nd conv layer (pic size 20)
        conv2 = lrelu(tf.nn.conv2d(conv1, weights[2], strides=[1, 2, 2, 1], padding='SAME') + weights[3])

        # 3rd conv layer (pic size 10)
        conv3 = lrelu(tf.nn.conv2d(conv2, weights[4], strides=[1, 2, 2, 1], padding='SAME') + weights[5])

        # reshape (pic size 5)
        p2resh = tf.reshape(conv3, [-1, 128 * 5 * 5])

        # dropout layer
        drop = tf.layers.dropout(p2resh, 0.3, training=True)

        # logits
        output = tf.nn.sigmoid(tf.matmul(drop, weights[6]) + weights[7])
        return output



if __name__ == '__main__':
    r = advReg1()
    r.train(300)
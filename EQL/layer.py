import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers


class EqlLayer(keras.layers.Layer):
    def __init__(self, weights, biases, kernel_regularizer=None):
        super(EqlLayer, self).__init__()
        self.w = weights
        self.b = biases
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
                                      shape=[int(input_shape[-1]), 5],
                                      regularizer=self.kernel_regularizer)

    def call(self, inputs):
        input1 = tf.matmul(inputs, self.w['h1']) + self.b['b1']
        input2 = tf.matmul(inputs, self.w['h2']) + self.b['b2']
        input3 = tf.matmul(inputs, self.w['h3']) + self.b['b3']
        input4 = tf.matmul(inputs, self.w['h4']) + self.b['b4']
        input5 = tf.matmul(inputs, self.w['h5']) + self.b['b5']
        input6 = tf.matmul(inputs, self.w['h6']) + self.b['b6']

        identity_output = tf.identity(input1, name='identity_output')
        sin_output = tf.sin(input2, name='sin_output')
        cos_output = tf.cos(input3, name='cos_output')
        sigmoid_output = tf.sigmoid(input4, name='sig_output')

        sum_input = tf.add(input5, input6)
        mult_output = tf.multiply(sum_input, sum_input, name='mult_output')

        out_layer = tf.concat([identity_output, sin_output, cos_output, sigmoid_output, mult_output], axis=1)

        return out_layer

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers, initializers


class EqlLayer(keras.layers.Layer):
    def __init__(self, regularizer = None, initializer='random_normal'):
        super(EqlLayer, self).__init__()
        self.regularizer = regularizers.get(regularizer)
        self.initializer = initializers.get(initializer)

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], 6),
            initializer=self.initializer,
            trainable=True, regularizer=self.regularizer
        )
        self.b = self.add_weight(
            shape=(6,), initializer=self.initializer, trainable=True
        )

    def call(self, inputs):
        out = tf.matmul(inputs, self.w) + self.b
        identity = tf.identity(tf.gather(out, [0], axis=1), name='identity_output')
        sin = tf.sin(tf.gather(out, [1], axis=1), name='sin_output')
        cos = tf.cos(tf.gather(out, [2], axis=1), name='cos_output')
        sigmoid = tf.sigmoid(tf.gather(out, [3], axis=1), name='sig_output')
        sum1 = tf.gather(out, [4], axis=1)
        sum2 = tf.gather(out, [5], axis=1)
        sum_input = tf.add(sum1, sum2)
        mult = tf.multiply(sum_input, sum_input, name='mult_output')

        output = tf.concat([identity, sin, cos, sigmoid, mult], axis=1)
        return output


class DenseLayer(keras.layers.Layer):
    def __init__(self, initializer='random_normal'):
        super(DenseLayer, self).__init__()
        self.initializer = initializer

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(5, 1),
            initializer=self.initializer,
            trainable=True
        )
        self.b = self.add_weight(
            shape=(1,), initializer=self.initializer, trainable=True
        )

    def call(self, inputs):
        out = tf.matmul(inputs, self.w) + self.b
        return out

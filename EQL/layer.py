import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers, initializers


class EqlLayer(keras.layers.Layer):
    def __init__(self, lmbda=0, w_initializer='random_normal', b_initializer='random_normal', mask=None):
        super(EqlLayer, self).__init__()
        self.regularizer = regularizers.L1(l1=lmbda)
        self.w_initializer = initializers.get(w_initializer)
        self.b_initializer = initializers.get(b_initializer)
        self.mask = mask

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], 6),
            initializer=self.w_initializer,
            trainable=True, regularizer=self.regularizer
        )
        self.b = self.add_weight(
            shape=(6,), initializer=self.b_initializer, trainable=True, regularizer=self.regularizer
        )

    def call(self, inputs):
        if self.mask:
            for i in range(self.w.shape[0]):
                w_mask = tf.matmul([self.w[i]], self.mask[0][i])[0]
                self.w[i].assign(w_mask)
            b_mask = tf.matmul([self.b], self.mask[1])[0]
            self.b.assign(b_mask)
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
    def __init__(self, lmbda=0, w_initializer='random_normal', b_initializer='random_normal', mask=None):
        super(DenseLayer, self).__init__()
        self.regularizer = regularizers.L1(l1=lmbda)
        self.w_initializer = initializers.get(w_initializer)
        self.b_initializer = initializers.get(b_initializer)
        self.mask = mask

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(5, 1),
            initializer=self.w_initializer,
            trainable=True, regularizer=self.regularizer
        )
        self.b = self.add_weight(
            shape=(1,), initializer=self.b_initializer, trainable=True, regularizer=self.regularizer
        )

    def call(self, inputs):
        if self.mask:
            for i in range(self.w.shape[0]):
                w_mask = tf.matmul([self.w[i]], self.mask[0][i])[0]
                self.w[i].assign(w_mask)
            b_mask = tf.matmul([self.b], self.mask[1])[0]
            self.b.assign(b_mask)
        out = tf.matmul(inputs, self.w) + self.b
        return out

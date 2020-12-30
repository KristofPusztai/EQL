import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers, initializers


def identity(out, index):
    return tf.identity(tf.gather(out, [index], axis=1), name='identity_output')


def sin(out, index):
    return tf.sin(tf.gather(out, [index], axis=1), name='sin_output')


def cos(out, index):
    return tf.cos(tf.gather(out, [index], axis=1), name='cos_output')


def sigmoid(out, index):
    return tf.sigmoid(tf.gather(out, [index], axis=1), name='sig_output')


def mult(out, index):
    sum1 = tf.gather(out, [index], axis=1)
    sum2 = tf.gather(out, [index + 1], axis=1)
    sum_input = tf.add(sum1, sum2)
    return tf.multiply(sum_input, sum_input, name='mult_output')


class EqlLayer(keras.layers.Layer):
    def __init__(self, w_initializer, b_initializer, v, lmbda=0, mask=None, exclude=None):
        super(EqlLayer, self).__init__()
        if exclude is None:
            exclude = []
        self.regularizer = regularizers.L1(l1=lmbda)
        self.w_initializer = initializers.get(w_initializer)
        self.b_initializer = initializers.get(b_initializer)
        self.mask = mask
        self.v = v
        self.activations = [identity, sin, cos, sigmoid, mult]

        self.exclusion = 0
        if 'id' in exclude:
            self.exclusion += 1
            self.activations.remove(identity)
        if 'sin' in exclude:
            self.exclusion += 1
            self.activations.remove(sin)
        if 'cos' in exclude:
            self.exclusion += 1
            self.activations.remove(cos)
        if 'sig' in exclude:
            self.exclusion += 1
            self.activations.remove(sigmoid)
        if 'mult' in exclude:
            self.exclusion += 2
            self.activations.remove(mult)

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], 6 * self.v - self.v * self.exclusion),
            initializer=self.w_initializer,
            trainable=True, regularizer=self.regularizer
        )
        self.b = self.add_weight(
            shape=(6 * self.v - self.v * self.exclusion,), initializer=self.b_initializer,
            trainable=True, regularizer=self.regularizer
        )

    def call(self, inputs):
        if self.mask:
            for i in range(self.w.shape[0]):
                w_mask = tf.matmul([self.w[i]], self.mask[0][i])[0]
                self.w[i].assign(w_mask)
            b_mask = tf.matmul([self.b], self.mask[1])[0]
            self.b.assign(b_mask)

        out = tf.matmul(inputs, self.w) + self.b
        output_batches = []
        for i in range(self.v):
            v = (6 - self.exclusion) * i
            for a in range(len(self.activations)):
                activation = self.activations[a](out, a + v)
                output_batches.append(activation)
        output = tf.concat(output_batches, axis=1)
        return output


class DenseLayer(keras.layers.Layer):
    def __init__(self, w_initializer, b_initializer, lmbda=0, mask=None):
        super(DenseLayer, self).__init__()
        self.regularizer = regularizers.L1(l1=lmbda)
        self.w_initializer = initializers.get(w_initializer)
        self.b_initializer = initializers.get(b_initializer)
        self.mask = mask

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], 1),
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

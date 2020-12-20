import tensorflow as tf

from EQL.layer import EqlLayer
from EQL.model import EQL


def test1():  # Tests layer construction
    w_init = tf.random_normal_initializer()
    weights = {
        'h1': tf.Variable(w_init(shape=(5, 1)), name='w_hidden1'),
        'h2': tf.Variable(w_init(shape=(5, 1)), name='w_hidden2'),
        'h3': tf.Variable(w_init(shape=(5, 1)), name='w_hidden3'),
        'h4': tf.Variable(w_init(shape=(5, 1)), name='w_hidden4'),
        'h5': tf.Variable(w_init(shape=(5, 1)), name='w_hidden5'),
        'h6': tf.Variable(w_init(shape=(5, 1)), name='w_hidden6'),
    }

    b_init = tf.zeros_initializer()
    biases = {
        'b1': tf.Variable(b_init(shape=(1,)), name='bias_hidden1'),
        'b2': tf.Variable(b_init(shape=(1,)), name='bias_hidden2'),
        'b3': tf.Variable(b_init(shape=(1,)), name='bias_hidden3'),
        'b4': tf.Variable(b_init(shape=(1,)), name='bias_hidden4'),
        'b5': tf.Variable(b_init(shape=(1,)), name='bias_hidden5'),
        'b6': tf.Variable(b_init(shape=(1,)), name='bias_hidden6'),
    }

    x = tf.ones((1, 5))
    linear_layer = EqlLayer(weights, biases)
    y = linear_layer(x)
    assert y.shape == (1, 5), 'Wrong shape of output!'
    print('test1 successful')


def test2():  # Tests layer dimension output
    w_init = tf.random_normal_initializer()
    weights = {
        'h1': tf.Variable(w_init(shape=(5, 1)), name='w_hidden1'),
        'h2': tf.Variable(w_init(shape=(5, 1)), name='w_hidden2'),
        'h3': tf.Variable(w_init(shape=(5, 1)), name='w_hidden3'),
        'h4': tf.Variable(w_init(shape=(5, 1)), name='w_hidden4'),
        'h5': tf.Variable(w_init(shape=(5, 1)), name='w_hidden5'),
        'h6': tf.Variable(w_init(shape=(5, 1)), name='w_hidden6'),
    }

    b_init = tf.zeros_initializer()
    biases = {
        'b1': tf.Variable(b_init(shape=(1,)), name='bias_hidden1'),
        'b2': tf.Variable(b_init(shape=(1,)), name='bias_hidden2'),
        'b3': tf.Variable(b_init(shape=(1,)), name='bias_hidden3'),
        'b4': tf.Variable(b_init(shape=(1,)), name='bias_hidden4'),
        'b5': tf.Variable(b_init(shape=(1,)), name='bias_hidden5'),
        'b6': tf.Variable(b_init(shape=(1,)), name='bias_hidden6'),
    }

    x = tf.ones((10, 5))
    linear_layer = EqlLayer(weights, biases)
    y = linear_layer(linear_layer(linear_layer(x)))
    assert y.shape == (10, 5), 'Wrong shape of output!'
    print('test2 successful')


def test3():  # Tests model dimensionality as a whole
    test = EQL(dim=5)
    x = tf.ones((100, 5))
    y = tf.random_normal_initializer()(shape=(100, 1))
    test.fit(x, y)
    params = test.count_params()
    assert params == 78, 'trainable parameter count is wrong'
    print('test3 successful')


test1()
test2()
test3()

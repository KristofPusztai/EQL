import tensorflow as tf

from EQL.layer import EqlLayer, DenseLayer
from EQL.model import EQL


def test1():  # Tests layer construction
    x = tf.ones((2, 2))
    linear_layer = EqlLayer()
    y = linear_layer(x)
    assert y.shape == (2,5), 'Wrong shape of output!'
    print('test1 successful')


def test2():  # Tests layer dimension output
    x = tf.ones((10, 5))
    linear_layer = EqlLayer()
    y = linear_layer(linear_layer(linear_layer(x)))
    assert y.shape == (10,5), 'Wrong shape of output!'
    print('test2 successful')


def test3(): # Tests custom DenseLayer implementation
    x = tf.ones((10, 5))
    linear_layer = DenseLayer()
    y = linear_layer(x)
    assert y.shape == (10, 1), 'Wrong shape of output!'
    print('test3 successful')


def test4():  # Tests model dimensionality as a whole
    test = EQL(dim=2)
    x = tf.ones((100, 2))
    y = tf.random_normal_initializer()(shape=(100, 1))
    test.build_and_compile_model()
    test.fit(x, y, 0)
    params = test.count_params()
    assert params == 60, 'trainable parameter count is wrong'
    print('test4 successful')


test1()
test2()
test3()
test4()

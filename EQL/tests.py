import tensorflow as tf

from EQL.layer import EqlLayer, DenseLayer
from EQL.model import EQL


def test1():  # Tests layer construction
    x = tf.ones((2, 2))
    linear_layer = EqlLayer('random_normal', 'random_normal', v=1)
    y = linear_layer(x)
    assert y.shape == (2, 5), 'Wrong shape of output!'
    print('test1 successful')


def test2():  # Tests layer dimension output
    x = tf.ones((10, 5))
    linear_layer = EqlLayer('random_normal', 'random_normal', v=1)
    y = linear_layer(linear_layer(linear_layer(x)))
    assert y.shape == (10, 5), 'Wrong shape of output!'
    print('test2 successful')


def test3():  # Tests custom DenseLayer implementation
    x = tf.ones((10, 5))
    linear_layer = DenseLayer('random_normal', 'random_normal')
    y = linear_layer(x)
    assert y.shape == (10, 1), 'Wrong shape of output!'
    print('test3 successful')


def test4():  # Tests masking output for layers
    x = tf.ones((10, 5))
    mask = [[

        [[0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]],

        [[1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 1]],

        [[1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 1]],

        [[1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1]],

    ],
        [[1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 1]]
    ]
    linear_layer = EqlLayer('random_normal', 'random_normal', mask=mask, v=1)
    linear_layer(x)  # necessary to initialize weights
    weights = linear_layer.get_weights()
    _1 = weights[0][0]
    _2 = weights[0][1]
    assert (_1 == [0., 0., 0., 0., 0., 0.]).all(), 'W_1 is not as expected (all 0)'
    assert _2[4] == 0, 'second to last value of W_2 is not 0'
    assert _2[5] == 0, 'last value of W_2 is not 0'
    print('test4 successful')


def test5():  # Tests model dimensionality as a whole
    test = EQL(dim=2)
    x = tf.ones((100, 2))
    y = tf.random_normal_initializer()(shape=(100, 1))
    test.build_and_compile_model()
    test.fit(x, y, 0, 1)
    params = test.count_params()
    assert params == 60, 'trainable parameter count is wrong'
    print('test5 successful')


def test6():  # Tests l1 regularization
    test = EQL(dim=2)
    x = tf.ones((100, 2))
    y = tf.random_normal_initializer()(shape=(100, 1))
    test.build_and_compile_model()
    test.fit(x, y, 0.01, t1=1)
    params = test.count_params()
    assert params == 60, 'trainable parameter count is wrong'
    print('test6 successful')


def test7():  # Tests l0 regularization
    test = EQL(dim=2)
    x = tf.ones((100, 2))
    y = tf.random_normal_initializer()(shape=(100, 1))
    test.build_and_compile_model()
    test.fit(x, y, 0.01, t1=1, t2=1)
    params = test.count_params()
    assert params == 60, 'trainable parameter count is wrong'
    print('test7 successful')


def test8():  # Tests layer width parameter
    x = tf.ones((2, 2))
    linear_layer = EqlLayer('random_normal', 'random_normal', v=2)
    y = linear_layer(x)
    assert y.shape == (2, 10), 'Wrong shape of output!'
    print('test8 successful')


def test9():    # Tests layer width parameter in model
    test = EQL(dim=2, v=[1, 2])
    x = tf.ones((100, 2))
    y = tf.random_normal_initializer()(shape=(100, 1))
    test.build_and_compile_model()
    test.fit(x, y, 0.01, t1=1, t2=1)
    params = test.count_params()
    assert params == 101, 'trainable parameter count is wrong'
    print('test9 successful')


def test10():   # Tests layer exclusion parameter
    x = tf.ones((2, 2))
    linear_layer = EqlLayer('random_normal', 'random_normal', v=2, exclude=['mult', 'sin'])
    y = linear_layer(x)
    assert y.shape == (2, 6), 'Wrong shape of output!'

    test = EQL(dim=2, v=[1, 2])
    x = tf.ones((100, 2))
    y = tf.random_normal_initializer()(shape=(100, 1))
    test.build_and_compile_model(exclude=[['sin','cos','sig','mult'], ['sin','cos','sig','mult']])
    test.fit(x, y, 0.0, t0=1)
    params = test.count_params()
    assert params == 10, 'trainable parameter count is wrong'
    print('test10 successful')


test1()
test2()
test3()
test4()
test5()
test6()
test7()
test8()
test9()
test10()

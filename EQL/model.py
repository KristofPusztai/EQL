import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from EQL.layer import EqlLayer


# TODO: Add regularization

class EQL:
    def __init__(self, num_layers=2, dim=1):
        self.b_init = tf.zeros_initializer()
        self.w_init = tf.random_normal_initializer()
        self.num_layers = num_layers
        self.model = None
        self.dim = dim

    def build_and_compile_model(self, metrics=None, loss_weights=None, weighted_metrics=None,
                                run_eagerly=None, kernel_regularizer=None):
        total_layers = []
        for i in range(self.num_layers):
            bias = {
                'b1': tf.Variable(self.b_init(shape=(1,)), name='bias_hidden1'),
                'b2': tf.Variable(self.b_init(shape=(1,)), name='bias_hidden2'),
                'b3': tf.Variable(self.b_init(shape=(1,)), name='bias_hidden3'),
                'b4': tf.Variable(self.b_init(shape=(1,)), name='bias_hidden4'),
                'b5': tf.Variable(self.b_init(shape=(1,)), name='bias_hidden5'),
                'b6': tf.Variable(self.b_init(shape=(1,)), name='bias_hidden6'),
            }
            weights = {
                'h1': tf.Variable(self.w_init(shape=(self.dim, 1)), name='w_hidden1'),
                'h2': tf.Variable(self.w_init(shape=(self.dim, 1)), name='w_hidden2'),
                'h3': tf.Variable(self.w_init(shape=(self.dim, 1)), name='w_hidden3'),
                'h4': tf.Variable(self.w_init(shape=(self.dim, 1)), name='w_hidden4'),
                'h5': tf.Variable(self.w_init(shape=(self.dim, 1)), name='w_hidden5'),
                'h6': tf.Variable(self.w_init(shape=(self.dim, 1)), name='w_hidden6'),
            }
            layer = EqlLayer(weights, bias, kernel_regularizer)
            total_layers.append(layer)

        total_layers.append(layers.Dense(1, name='output'))
        model = keras.Sequential(total_layers)

        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(0.001),
                      metrics=metrics, loss_weights=loss_weights,
                      weighted_metrics=weighted_metrics, run_eagerly=run_eagerly)
        self.model = model

    def fit(self, x, y, batch_size=None, epochs=100, verbose=0, callbacks=None,
            validation_split=0.0, validation_data=None, shuffle=True, class_weight=None,
            sample_weight=None, initial_epoch=0, steps_per_epoch=None,
            validation_steps=None, validation_batch_size=None, validation_freq=1,
            max_queue_size=10, workers=1, use_multiprocessing=False, t1=None, t2=None):
        assert self.model is not None, 'Must call build_and_compile method model before training'
        # TODO: Implement t1, t2 l1 reg.
        self.model.fit(x, y, batch_size, epochs, verbose, callbacks,
                       validation_split, validation_data, shuffle, class_weight,
                       sample_weight, initial_epoch, steps_per_epoch,
                       validation_steps, validation_batch_size, validation_freq,
                       max_queue_size, workers, use_multiprocessing)

    def predict(self, x, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10,
                workers=1, use_multiprocessing=False):
        return self.model.predict(x, batch_size=batch_size, verbose=verbose, steps=steps, callbacks=callbacks,
                                  max_queue_size=max_queue_size, workers=workers,
                                  use_multiprocessing=use_multiprocessing)

    def summary(self):
        return self.model.summary()

    def count_params(self):
        return self.model.count_params()

    def get_weights(self, layer):
        return self.model.layers[layer].get_weights()

    def set_weights(self, layer, weights):
        self.model.layers[layer].set_weights(weights)

    def evaluate(self, x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None,
                 callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False,
                 return_dict=False):
        return self.model.evaluate(x=x, y=y, batch_size=batch_size, verbose=verbose, sample_weight=sample_weight,
                                   steps=steps,
                                   callbacks=callbacks, max_queue_size=max_queue_size, workers=workers,
                                   use_multiprocessing=use_multiprocessing,
                                   return_dict=return_dict)

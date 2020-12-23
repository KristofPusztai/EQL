import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from EQL.layer import EqlLayer, DenseLayer


# TODO: Add regularization
class EQL:
    def __init__(self, num_layers=2, dim=1, metrics=None, loss_weights=None, weighted_metrics=None,
                 run_eagerly=None, optimizer=tf.keras.optimizers.Adam(0.001)):
        self.num_layers = num_layers
        self.model = None
        self.dim = dim

        self.metrics = metrics
        self.loss_weights = loss_weights
        self.weighted_metrics = weighted_metrics
        self.run_eagerly = run_eagerly
        self.optimizer = optimizer

    def build_and_compile_model(self):
        inputs = tf.keras.Input(shape=(self.dim,))
        x = inputs
        for i in range(self.num_layers):
            x = EqlLayer()(x)
        out_layer = DenseLayer()
        outputs = out_layer(x)
        model = keras.Model(inputs, outputs)

        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=self.optimizer,
                      metrics=self.metrics, loss_weights=self.loss_weights,
                      weighted_metrics=self.weighted_metrics, run_eagerly=self.run_eagerly)
        self.model = model

    def __rebuild(self, weights, biases, l0=False, lmbda=0):
        total_layers = []
        for i in range(self.num_layers):
            layer = EqlLayer()
            total_layers.append(layer)
        total_layers.append(layers.Dense(1, name='output'))
        model = keras.Sequential(total_layers)

        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=self.optimizer,
                      metrics=self.metrics, loss_weights=self.loss_weights,
                      weighted_metrics=self.weighted_metrics, run_eagerly=self.run_eagerly)
        self.model = model

    def fit(self, x, y, batch_size=None, t0=100, t1=0, t2=0, verbose=0, callbacks=None,
            validation_split=0.0, validation_data=None, shuffle=True, class_weight=None,
            sample_weight=None, initial_epoch=0, steps_per_epoch=None,
            validation_steps=None, validation_batch_size=None, validation_freq=1,
            max_queue_size=10, workers=1, use_multiprocessing=False):
        assert self.model is not None, 'Must call build_and_compile method model before training'
        self.model.fit(x, y, batch_size, t0, verbose, callbacks,
                       validation_split, validation_data, shuffle, class_weight,
                       sample_weight, initial_epoch, steps_per_epoch,
                       validation_steps, validation_batch_size, validation_freq,
                       max_queue_size, workers, use_multiprocessing)
        # TODO: Implement t1, t2 l1 reg.
        # if t1 != 0:
        # if t2 != 0:

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

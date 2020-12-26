import tensorflow as tf

from tensorflow import keras
import numpy as np
from EQL.layer import EqlLayer, DenseLayer


# TODO: Add regularization
class EQL:
    def __init__(self, num_layers=2, dim=1):
        self.num_layers = num_layers
        self.model = None
        self.dim = dim

    def replace_w_near_zero(self, data):
        for d in range(len(data)):
            for i in range(len(data[d])):
                if np.isclose(data[d][i], 0, atol=self.atol):
                    data[d][i] = 1
        return data

    def replace_b_near_zero(self, data):
        for i in range(len(data)):
            if np.isclose(data[i], 0, atol=self.atol):
                data[i] = 1
        return data

    def build_and_compile_model(self, metrics=None, loss_weights=None, weighted_metrics=None,
                                run_eagerly=None, optimizer=tf.keras.optimizers.Adam(0.001)):
        self.metrics = metrics
        self.loss_weights = loss_weights
        self.weighted_metrics = weighted_metrics
        self.run_eagerly = run_eagerly
        self.optimizer = optimizer

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

    def __rebuild(self, weights, biases, lmbda, l0=False):
        inputs = tf.keras.Input(shape=(self.dim,))
        x = inputs
        for i in range(self.num_layers):
            w_initializer = tf.constant_initializer(weights[i])
            b_initializer = tf.constant_initializer(biases[i])
            x = EqlLayer(lmbda, w_initializer=w_initializer, b_initializer=b_initializer)(x)
        w_initializer = tf.constant_initializer(weights[self.num_layers])
        b_initializer = tf.constant_initializer(biases[self.num_layers])
        out_layer = DenseLayer(lmbda, w_initializer=w_initializer, b_initializer=b_initializer)
        outputs = out_layer(x)
        model = keras.Model(inputs, outputs)

        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=self.optimizer,
                      metrics=self.metrics, loss_weights=self.loss_weights,
                      weighted_metrics=self.weighted_metrics, run_eagerly=self.run_eagerly)
        self.model = model

    def fit(self, x, y, lmbda, t0=100, t1=0, t2=0, verbose=0, batch_size=None, callbacks=None,
            validation_split=0.0, validation_data=None, shuffle=True, class_weight=None,
            sample_weight=None, initial_epoch=0, steps_per_epoch=None,
            validation_steps=None, validation_batch_size=None, validation_freq=1,
            max_queue_size=10, workers=1, use_multiprocessing=False, atol=0.01):
        self.atol=atol
        assert self.model is not None, 'Must call build_and_compile method model before training'
        if verbose != 0:
            print('Beginning Training, T0_epochs = ' + str(t0))
        self.model.fit(x, y, batch_size, t0, verbose, callbacks,
                       validation_split, validation_data, shuffle, class_weight,
                       sample_weight, initial_epoch, steps_per_epoch,
                       validation_steps, validation_batch_size, validation_freq,
                       max_queue_size, workers, use_multiprocessing)
        # TODO: Implement t1, t2 l1 reg.
        if t1 != 0:
            weights = []
            biases = []
            for i in range(1, self.num_layers + 1):
                weights.append(self.get_weights(i)[0])
                biases.append(self.get_weights(i)[1])
            self.__rebuild(weights, biases, lmbda=lmbda)
            if verbose != 0:
                print('Beginning LASSO Training, T1_epochs = ' + str(t1))
            self.model.fit(x, y, batch_size, t1, verbose, callbacks,
                           validation_split, validation_data, shuffle, class_weight,
                           sample_weight, initial_epoch, steps_per_epoch,
                           validation_steps, validation_batch_size, validation_freq,
                           max_queue_size, workers, use_multiprocessing)
        if t2 != 0:
            weights = []
            biases = []
            for i in range(1, self.num_layers + 1):
                w = self.replace_w_near_zero(self.get_weights(i)[0])
                weights.append(w)
                b = self.replace_b_near_zero(self.get_weights(i)[1])
                biases.append(b)
            self.__rebuild(weights, biases, l0=True, lmbda=0)
            if verbose != 0:
                print('Beginning L0 Training, T2_epochs = ' + str(t2))
            self.model.fit(x, y, batch_size, t2, verbose, callbacks,
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

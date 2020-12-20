import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from EQL.layer import EqlLayer


class EQL:
    def __init__(self, num_layers=2, dim=1):
        self.b_init = tf.zeros_initializer()
        self.w_init = tf.random_normal_initializer()
        self.num_layers = num_layers
        self.model = None
        self.dim = dim
        self.w = None
        self.b = None

    def __build_and_compile_model(self, loss, metrics, loss_weights,
                                  weighted_metrics, run_eagerly):
        total_layers = []
        init_layer = EqlLayer(self.w, self.b)
        total_layers.append(init_layer)
        for i in range(self.num_layers - 1):
            bias = {
                'b1': tf.Variable(self.b_init(shape=(1,)), name='bias_hidden1'),
                'b2': tf.Variable(self.b_init(shape=(1,)), name='bias_hidden2'),
                'b3': tf.Variable(self.b_init(shape=(1,)), name='bias_hidden3'),
                'b4': tf.Variable(self.b_init(shape=(1,)), name='bias_hidden4'),
                'b5': tf.Variable(self.b_init(shape=(1,)), name='bias_hidden5'),
                'b6': tf.Variable(self.b_init(shape=(1,)), name='bias_hidden6'),
            }
            weights = {
                'h1': tf.Variable(self.w_init(shape=(5, 1)), name='w_hidden1'),
                'h2': tf.Variable(self.w_init(shape=(5, 1)), name='w_hidden2'),
                'h3': tf.Variable(self.w_init(shape=(5, 1)), name='w_hidden3'),
                'h4': tf.Variable(self.w_init(shape=(5, 1)), name='w_hidden4'),
                'h5': tf.Variable(self.w_init(shape=(5, 1)), name='w_hidden5'),
                'h6': tf.Variable(self.w_init(shape=(5, 1)), name='w_hidden6'),
            }
            layer = EqlLayer(weights, bias)
            total_layers.append(layer)

        total_layers.append(layers.Dense(1, name='output'))
        model = keras.Sequential(total_layers)

        model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(0.001), metrics=metrics, loss_weights=loss_weights,
                      weighted_metrics=weighted_metrics, run_eagerly=run_eagerly)
        return model

    def fit(self, x, y, weights=None, biases=None, epochs=100, verbose=0, validation_split=0.2,
            loss=tf.keras.losses.MeanSquaredError(), metrics=None, loss_weights=None, weighted_metrics=None,
            run_eagerly=None):
        if weights is None:
            self.w = {
                'h1': tf.Variable(self.w_init(shape=(self.dim, 1)), name='w_hidden1'),
                'h2': tf.Variable(self.w_init(shape=(self.dim, 1)), name='w_hidden2'),
                'h3': tf.Variable(self.w_init(shape=(self.dim, 1)), name='w_hidden3'),
                'h4': tf.Variable(self.w_init(shape=(self.dim, 1)), name='w_hidden4'),
                'h5': tf.Variable(self.w_init(shape=(self.dim, 1)), name='w_hidden5'),
                'h6': tf.Variable(self.w_init(shape=(self.dim, 1)), name='w_hidden6'),
            }
            self.b = {
                'b1': tf.Variable(self.b_init(shape=(1,)), name='bias_hidden1'),
                'b2': tf.Variable(self.b_init(shape=(1,)), name='bias_hidden2'),
                'b3': tf.Variable(self.b_init(shape=(1,)), name='bias_hidden3'),
                'b4': tf.Variable(self.b_init(shape=(1,)), name='bias_hidden4'),
                'b5': tf.Variable(self.b_init(shape=(1,)), name='bias_hidden5'),
                'b6': tf.Variable(self.b_init(shape=(1,)), name='bias_hidden6'),
            }
        else:
            self.w = weights
            self.b = biases
        assert type(self.w) == dict, 'specified weights are incorrect, must be a dictionary of form:\n' \
                                     'biases = {\n' \
                                     "\t'h1': tensor(shape=(input_dim,1)), name='w_hidden1'),\n" \
                                     "\t'h2': tensor(shape=(input_dim,1)), name='w_hidden2'),\n" \
                                     "\t'h3': tensor(shape=(input_dim,1)), name='w_hidden3'),\n" \
                                     "\t'h4': tensor(shape=(input_dim,1)), name='w_hidden4'),\n" \
                                     "\t'h5': tensor(shape=(input_dim,1)), name='w_hidden5'),\n" \
                                     "\t'h6': tensor(shape=(input_dim,1)), name='w_hidden6'),\n" \
                                     '}'
        assert type(self.b) == dict, 'specified biases are incorrect, must be a dictionary of form:\n' \
                                     'biases = {\n' \
                                     "\t'b1': tensor(shape=(1,)), name='bias_hidden1'),\n" \
                                     "\t'b2': tensor(shape=(1,)), name='bias_hidden2'),\n" \
                                     "\t'b3': tensor(shape=(1,)), name='bias_hidden3'),\n" \
                                     "\t'b4': tensor(shape=(1,)), name='bias_hidden4'),\n" \
                                     "\t'b5': tensor(shape=(1,)), name='bias_hidden5'),\n" \
                                     "\t'b6': tensor(shape=(1,)), name='bias_hidden6'),\n" \
                                     '}'
        self.model = self.__build_and_compile_model(loss=loss, metrics=metrics, loss_weights=loss_weights,
                                                    weighted_metrics=weighted_metrics, run_eagerly=run_eagerly)
        self.model.fit(x, y, epochs=epochs,
                       # suppress logging
                       verbose=verbose,
                       # Calculate validation results on 20% of the training data
                       validation_split=validation_split)

    def predict(self, x, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10,
                workers=1, use_multiprocessing=False):
        return self.model.predict(x, batch_size=batch_size, verbose=verbose, steps=steps, callbacks=callbacks,
                                  max_queue_size=max_queue_size, workers=workers,
                                  use_multiprocessing=use_multiprocessing)

    def summary(self):
        return self.model.summary()

    def count_params(self):
        return self.model.count_params()

    def evaluate(self, x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None,
                 callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False,
                 return_dict=False):
        return self.model.evaluate(x=x, y=y, batch_size=batch_size, verbose=verbose, sample_weight=sample_weight,
                                   steps=steps,
                                   callbacks=callbacks, max_queue_size=max_queue_size, workers=workers,
                                   use_multiprocessing=use_multiprocessing,
                                   return_dict=return_dict)

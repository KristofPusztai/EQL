import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


class EQL:
    def __init__(self, num_layers=2):
        self.num_layers = num_layers
        self.model = None

    def __build_and_compile_model(self):
        total_layers = []
        for i in range(self.num_layers):
            id_out = tf.keras.layers.Dense(units=1, activation=tf.identity)
            sin_out = tf.keras.layers.Dense(units=1, activation=tf.math.sin)
            cos_out = tf.keras.layers.Dense(units=1, activation=tf.math.cos)
            sigmoid_out = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid)

            out = tf.concat([id_out, sin_out, cos_out, sigmoid_out...], axis=1)
            # TODO: Implement custom layer
            total_layers.append(tf.keras.layers.Dense(6, activation=tf.math.sin))
        total_layers.append(layers.Dense(1))
        model = keras.Sequential(total_layers)

        model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))
        return model

    def fit(self, x, y):
        self.model = self.__build_and_compile_model()
        self.model.fit(x, y, epochs=100,
                       # suppress logging
                       verbose=0,
                       # Calculate validation results on 20% of the training data
                       validation_split=0.2)

    def predict(self, x):
        return self.model.predict(x)

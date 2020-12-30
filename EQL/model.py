import tensorflow as tf

from tensorflow import keras
import numpy as np
from EQL.layer import EqlLayer, DenseLayer


class EQL:
    """
    EQL model class, here we build our model and specify training processes and architecture
    """

    def __init__(self, num_layers=2, dim=1, v=None):
        """
        EQL initializer

        :param num_layers: Number of hidden layers
        :type num_layers: integer
        :param dim: Dimension of input
        :type dim: integer
        :param v: Integer number of binary units in hidden layers, v = 1/4 * u, u -> unary units
        :type v: list
        """
        if v is None:
            v = np.ones(num_layers, dtype=int)
        self.num_layers = num_layers
        self.model = None
        self.dim = dim
        assert len(v) == self.num_layers, 'v array must have same dimensions as number of hidden layers param'
        self.v = v

    def __replace_w_near_zero(self, data):
        """
        :param data: weights
        :type data: list
        :return: weights with 0's replacing values close to or equal to 0
        :rtype: list
        """
        for d in range(len(data)):
            for i in range(len(data[d])):
                if np.isclose(data[d][i], 0, atol=self.atol):
                    data[d][i] = 0
        return data

    def __replace_b_near_zero(self, data):
        """
        :param data: biases
        :type data: list
        :return: biases with 0's replacing values close to or equal to 0
        :rtype: list
        """
        for i in range(len(data)):
            if np.isclose(data[i], 0, atol=self.atol):
                data[i] = 0
        return data

    def __build_mask(self, weights, biases):
        """
        Builds incomplete identity mask to define trainable weights

        :param weights: weight array
        :type weights: list
        :param biases: bias array
        :type biases: list
        :return: Array of incomplete identity matrices
        :rtype: list
        """
        mask = []
        b_mask = []
        weight_masks = []
        for w in weights:
            w_mask = []
            for i in w:
                if i == 0:
                    w_mask.append(0)
                else:
                    w_mask.append(1)
            w_mask = np.diag(w_mask)
            weight_masks.append(w_mask)
        for b in biases:
            if b == 0:
                b_mask.append(0)
            else:
                b_mask.append(1)
        b_mask = np.diag(b_mask)
        mask.append(weight_masks)
        mask.append(b_mask)
        return mask

    def build_and_compile_model(self, metrics=None, loss_weights=None, weighted_metrics=None,
                                run_eagerly=None, optimizer=tf.keras.optimizers.Adam(0.001),
                                w_init='random_normal', b_init='random_normal', exclude=None):
        """
        Configures the model for training.

        :param metrics: model performance metrics
        :type metrics: tf.keras.metrics
        :param loss_weights: weights for certain loss, check tensorflow keras api documentation for more info
        :type loss_weights: list
        :param weighted_metrics: weights for desired metrics
        :type weighted_metrics: list
        :param run_eagerly: Defaults to False. If True, this Model's logic will not be wrapped in a tf.function.
            Recommended to leave this as None unless your Model cannot be run inside a tf.function.
        :type run_eagerly: bool
        :param optimizer: name of optimizer or optimizer instance. See tf.keras.optimizers.
        :type optimizer: tf.keras.optimizers
        :param w_init: Initializer for weights in network, default is random normal distribution
        :type w_init: str or tf.keras.initializers
        :param b_init: Initializer for biases in network, default is random normal distribution
        :type b_init: str or tf.keras.initializers
        :param exclude: Exclude any activation functions in a layer, must be a multidimensional list of activation
            function names: ['id','sin', 'cos','sig','mult']
        :type exclude: list
        """
        if exclude:
            assert len(exclude) == self.num_layers, "exclude parameter wrong format, len(exclude) must equal " \
                                                    "num_layers, ex.: exclude=[['sin'],[]], num_layers = 2"
        else:
            exclude = [[] for i in range(self.num_layers)]
        self.metrics = metrics
        self.loss_weights = loss_weights
        self.weighted_metrics = weighted_metrics
        self.run_eagerly = run_eagerly
        self.optimizer = optimizer
        self.exclude = exclude

        inputs = tf.keras.Input(shape=(self.dim,))
        x = inputs
        for i in range(self.num_layers):
            x = EqlLayer(w_initializer=w_init, b_initializer=b_init, v=self.v[i], exclude=self.exclude[i])(x)
        out_layer = DenseLayer(w_initializer=w_init, b_initializer=b_init)
        outputs = out_layer(x)
        model = keras.Model(inputs, outputs)

        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=self.optimizer,
                      metrics=self.metrics, loss_weights=self.loss_weights,
                      weighted_metrics=self.weighted_metrics, run_eagerly=self.run_eagerly)
        self.model = model

    def __rebuild(self, weights, biases, lmbda, l0=False):
        """
        :param weights: Weights from previous training
        :type weights: list
        :param biases: Biases from previous training
        :type biases: list
        :param lmbda: l1 regularizer parameter
        :type lmbda: float
        :param l0: Determines whether function is in l0 state
        :type l0: bool
        """
        inputs = tf.keras.Input(shape=(self.dim,))
        x = inputs
        for i in range(self.num_layers):
            w_initializer = tf.constant_initializer(weights[i])
            b_initializer = tf.constant_initializer(biases[i])
            if l0:
                mask = self.__build_mask(weights[i], biases[i])
                x = EqlLayer(lmbda=lmbda, w_initializer=w_initializer, b_initializer=b_initializer,
                             mask=mask, v=self.v[i], exclude=self.exclude[i])(x)
            else:
                x = EqlLayer(lmbda=lmbda, w_initializer=w_initializer, b_initializer=b_initializer,
                             v=self.v[i], exclude=self.exclude[i])(x)
        w_initializer = tf.constant_initializer(weights[self.num_layers])
        b_initializer = tf.constant_initializer(biases[self.num_layers])
        if l0:
            mask = self.__build_mask(weights[self.num_layers], biases[self.num_layers])
            out_layer = DenseLayer(lmbda=lmbda, w_initializer=w_initializer, b_initializer=b_initializer, mask=mask)
        else:
            out_layer = DenseLayer(lmbda=lmbda, w_initializer=w_initializer, b_initializer=b_initializer)
        outputs = out_layer(x)
        model = keras.Model(inputs, outputs)

        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=self.optimizer,
                      metrics=self.metrics, loss_weights=self.loss_weights,
                      weighted_metrics=self.weighted_metrics, run_eagerly=self.run_eagerly)
        self.model = model

    def fit(self, x, y, lmbda, t0=100, t1=0, t2=0, initial_epoch=0, verbose=0, batch_size=None, callbacks=None,
            validation_split=0.0, validation_data=None, shuffle=True, class_weight=None,
            sample_weight=None, steps_per_epoch=None,
            validation_steps=None, validation_batch_size=None, validation_freq=1,
            max_queue_size=10, workers=1, use_multiprocessing=False, atol=0.01):
        """
        Trains the model

        :param x: X data matrix, preferably tensor but can be an array type
        :type x: tf.Tensor
        :param y: Y data matrix, preferably tensor but can be an array type
        :type y: tf.Tensor
        :param lmbda: l1 regularizer
        :type lmbda: float
        :param t0: T0 epochs as specified in paper, no regularization
        :type t0: int
        :param t1: T1 epochs as specified in paper, l1 regularization
        :type t1: int
        :param t2: T2 epochs as specified in paper, l0 regularization
        :type t2: int
        :param initial_epoch: Epoch at which to start training (useful for resuming a previous training run).
        :type initial_epoch: int
        :param verbose: Output of call, 0 silences training process
        :type verbose: int
        :param batch_size: Number of samples per gradient update. If unspecified, batch_size will default to 32.
            Do not specify the batch_size if your data is in the form of datasets
        :type batch_size: integer or None
        :param callbacks: List of callbacks to apply during training. See tf.keras.callbacks.
            Note tf.keras.callbacks.ProgbarLogger and tf.keras.callbacks.History callbacks are created automatically
            and need not be passed into model.fit. tf.keras.callbacks.ProgbarLogger is created or not based on verbose
            argument to model.fit.
        :type callbacks: keras.callbacks.Callback
        :param validation_split: Fraction of the training data to be used as validation data. The model will set apart
            this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics
            on this data at the end of each epoch. The validation data is selected from the last samples in the x and y
            data provided, before shuffling. This argument is not supported when x is a dataset, generator or
             keras.utils.Sequence instance.
        :type validation_split: int
        :param validation_data:Data on which to evaluate the loss and any model metrics at the end of each epoch.
            The model will not be trained on this data. Thus, note the fact that the validation loss of data provided
            using validation_split or validation_data is not affected by regularization layers like noise and dropout.
            validation_data will override validation_split. validation_data could be: - tuple (x_val, y_val) of Numpy
            arrays or tensors - tuple (x_val, y_val, val_sample_weights) of Numpy arrays - dataset For the first two
            cases, batch_size must be provided. For the last case, validation_steps could be provided. Note that
            validation_data does not support all the data types that are supported in x, eg, dict, generator or
            keras.utils.Sequence.
        :type validation_data: tf.Tensor
        :param shuffle:
        :type shuffle:
        :param class_weight: (whether to shuffle the training data before each epoch) or str (for 'batch').
            This argument is ignored when x is a generator. 'batch' is a special option for dealing with the
            limitations of HDF5 data; it shuffles in batch-sized chunks. Has no effect when steps_per_epoch is not None.
        :type class_weight: bool
        :param sample_weight:Optional Numpy array of weights for the training samples, used for weighting the loss
            function (during training only). You can either pass a flat (1D) Numpy array with the same length as the
            input samples (1:1 mapping between weights and samples), or in the case of temporal data, you can pass a 2D
            array with shape (samples, sequence_length), to apply a different weight to every timestep of every sample.
            This argument is not supported when x is a dataset, generator, or keras.utils.Sequence instance, instead
            provide the sample_weights as the third element of x.
        :type sample_weight: np.Array
        :param steps_per_epoch:Total number of steps (batches of samples) before declaring one epoch finished and
            starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default
            None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be
            determined. If x is a tf.data dataset, and 'steps_per_epoch' is None, the epoch will run until the input
            dataset is exhausted. When passing an infinitely repeating dataset, you must specify the steps_per_epoch
            argument. This argument is not supported with array inputs.
        :type steps_per_epoch: int
        :param validation_steps: Only relevant if validation_data is provided and is a tf.data dataset. Total number of
            steps (batches of samples) to draw before stopping when performing validation at the end of every epoch.
            If 'validation_steps' is None, validation will run until the validation_data dataset is exhausted.
            In the case of an infinitely repeated dataset, it will run into an infinite loop. If 'validation_steps'
            is specified and only part of the dataset will be consumed, the evaluation will start from the beginning
             of the dataset at each epoch. This ensures that the same validation samples are used every time.
        :type validation_steps: int
        :param validation_batch_size: Number of samples per validation batch. If unspecified, will default to
            batch_size. Do not specify the validation_batch_size if your data is in the form of datasets, generators,
            or keras.utils.Sequence instances (since they generate batches).
        :type validation_batch_size: int
        :param validation_freq: Only relevant if validation data is provided. Integer or collections_abc.Container
            instance (e.g. list, tuple, etc.). If an integer, specifies how many training epochs to run before a new
            validation run is performed, e.g. validation_freq=2 runs validation every 2 epochs. If a Container,
            specifies the epochs on which to run validation, e.g. validation_freq=[1, 2, 10] runs validation at the end
            of the 1st, 2nd, and 10th epochs.
        :type validation_freq: int
        :param max_queue_size: Used for generator or keras.utils.Sequence input only. Maximum size for the generator
            queue. If unspecified, max_queue_size will default to 10.
        :type max_queue_size: int
        :param workers: Used for generator or keras.utils.Sequence input only. Maximum number of processes to spin up
            when using process-based threading. If unspecified, workers will default to 1. If 0, will execute the
            generator on the main thread.
        :type workers: int
        :param use_multiprocessing: Used for generator or keras.utils.Sequence input only. If True, use process-based
            threading. If unspecified, use_multiprocessing will default to False. Note that because this
            implementation relies on multiprocessing, you should not pass non-picklable arguments to the
            generator as they can't be passed easily to children processes.
        :type use_multiprocessing: bool
        :param atol: Absolute tolerance for values near 0, used only for L0 regularization epochs
        :type atol: float
        """
        self.atol = atol
        assert self.model is not None, 'Must call build_and_compile method model before training'
        if verbose != 0:
            print('Beginning Training, T0_epochs = ' + str(t0))
        self.model.fit(x, y, batch_size, t0, verbose, callbacks,
                       validation_split, validation_data, shuffle, class_weight,
                       sample_weight, initial_epoch, steps_per_epoch,
                       validation_steps, validation_batch_size, validation_freq,
                       max_queue_size, workers, use_multiprocessing)
        if t1 != 0:
            weights = []
            biases = []
            for i in range(1, self.num_layers + 2):  # Plus 2 here because range goes from 1 -> n - 1, need n + 1
                weights.append(self.get_weights(i)[0])
                biases.append(self.get_weights(i)[1])
            self.__rebuild(weights, biases, lmbda)
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
            for i in range(1, self.num_layers + 2):
                w = self.__replace_w_near_zero(self.get_weights(i)[0])
                weights.append(w)
                b = self.__replace_b_near_zero(self.get_weights(i)[1])
                biases.append(b)
            self.__rebuild(weights, biases, 0, l0=True)
            if verbose != 0:
                print('Beginning L0 Training, T2_epochs = ' + str(t2))
            self.model.fit(x, y, batch_size, t2, verbose, callbacks,
                           validation_split, validation_data, shuffle, class_weight,
                           sample_weight, initial_epoch, steps_per_epoch,
                           validation_steps, validation_batch_size, validation_freq,
                           max_queue_size, workers, use_multiprocessing)

    def predict(self, x, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10,
                workers=1, use_multiprocessing=False):
        """
        Generates output predictions for the input samples.

        Computation is done in batches. This method is designed for performance in large scale inputs. For small amount
        of inputs that fit in one batch, directly using __call__ is recommended for faster execution, e.g., model(x), or
        model(x, training=False) if you have layers such as tf.keras.layers.BatchNormalization that behaves differently
        during  inference. Also, note the fact that test loss is not affected by regularization
        layers like noise and dropout.

        :param x: X data
        :type x: tf.Tensor
        :param batch_size: Number of samples per batch. If unspecified, batch_size will default to 32. Do not specify
            the batch_size if your data is in the form of dataset, generators, or keras.utils.Sequence instances
            (since they generate batches).
        :type batch_size: integer or None
        :param verbose: Verbosity mode, 0 or 1
        :type verbose: int
        :param steps: Total number of steps (batches of samples) before declaring the prediction round finished.
            Ignored with the default value of None. If x is a tf.data dataset and steps is None, predict will run
            until the input dataset is exhausted.
        :type steps: int
        :param callbacks: List of callbacks to apply during prediction. See keras callbacks.
        :type callbacks: tf.keras.callbacks.Callback
        :param max_queue_size: Used for generator or keras.utils.Sequence input only. Maximum size for the generator
            queue. If unspecified, max_queue_size will default to 10.
        :type max_queue_size: int
        :param workers: Used for generator or keras.utils.Sequence input only. Maximum number of processes to spin up
            when using process-based threading. If unspecified, workers will default to 1. If 0, will execute the
            generator on the main thread.
        :type workers: int
        :param use_multiprocessing: Used for generator or keras.utils.Sequence input only. If True, use process-based
            threading. If unspecified, use_multiprocessing will default to False. Note that because this implementation
            relies on multiprocessing, you should not pass non-picklable arguments to the generator as they can't be
            passed easily to children processes.
        :type use_multiprocessing: bool
        :return: Numpy array(s) of predictions.
        :rtype: np.ndarray
        """
        return self.model.predict(x, batch_size=batch_size, verbose=verbose, steps=steps, callbacks=callbacks,
                                  max_queue_size=max_queue_size, workers=workers,
                                  use_multiprocessing=use_multiprocessing)

    def summary(self):
        """
        :return: Prints a string summary of the network.
        :rtype: str
        """
        return self.model.summary()

    def count_params(self):
        """
        :return: Number of parameters in model
        :rtype: int
        """
        return self.model.count_params()

    def get_weights(self, layer):
        """
        :param layer: Specified layer number to get weights from
        :type layer: int
        :return: Array of weights and biases in certain layer
        :rtype: np.ndarray
        """
        return self.model.layers[layer].get_weights()

    def set_weights(self, layer, weights):
        """
        :param layer: Specified layer number to set weights in
        :type layer: int
        :param weights: Array of weights, must match dimensionality of specified model layer
        :type weights: np.ndarray
        """
        self.model.layers[layer].set_weights(weights)

    def evaluate(self, x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None,
                 callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False,
                 return_dict=False):
        """
        Returns the loss value & metrics values for the model in test mode.

        Computation is done in batches (see the batch_size arg.)

        :param x:Input data. It could be:
            -A Numpy array (or array-like), or a list of arrays (in case the model has multiple inputs).
            -A TensorFlow tensor, or a list of tensors (in case the model has multiple inputs).
            -A dict mapping input names to the corresponding array/tensors, if the model has named inputs.
            -A tf.data dataset. Should return a tuple of either (inputs, targets) or (inputs, targets, sample_weights).
            -A generator or keras.utils.Sequence returning (inputs, targets) or (inputs, targets, sample_weights).
            -A more detailed description of unpacking behavior for iterator types (Dataset, generator, Sequence) is
            given in the Unpacking behavior for iterator-like inputs section of Model.fit.
        :type x: tf.Tensor
        :param y: Target data. Like the input data x, it could be either Numpy array(s) or TensorFlow tensor(s).
            It should be consistent with x (you cannot have Numpy inputs and tensor targets, or inversely).
            If x is a dataset, generator or keras.utils.Sequence instance, y should not be specified (since targets
            will be obtained from the iterator/dataset).
        :type y: tf.Tensor
        :param batch_size: Number of samples per batch of computation. If unspecified, batch_size will default to 32.
            Do not specify the batch_size if your data is in the form of a dataset, generators, or keras.utils.Sequence
            instances (since they generate batches).
        :type batch_size: int
        :param verbose: Verbosity mode. 0 = silent, 1 = progress bar.
        :type verbose: int
        :param sample_weight: Optional Numpy array of weights for the test samples, used for weighting the loss
            function. You can either pass a flat (1D) Numpy array with the same length as the input samples
            (1:1 mapping between weights and samples), or in the case of temporal data, you can pass a 2D array with
            shape (samples,sequence_length), to apply a different weight to every timestep of every sample.
            This argument is not supported when x is a dataset, instead pass sample weights as the third element of x.
        :type sample_weight: np.ndarray
        :param steps: Total number of steps (batches of samples) before declaring the evaluation round finished.
            Ignored with the default value of None. If x is a tf.data dataset and steps is None, 'evaluate' will
            run until the dataset is exhausted. This argument is not supported with array inputs.
        :type steps: int
        :param callbacks: List of keras.callbacks.Callback instances. List of callbacks to apply during evaluation.
            See callbacks.
        :type callbacks: tf.keras.callbacks.Callback
        :param max_queue_size: Used for generator or keras.utils.Sequence input only. Maximum size for the generator
            queue. If unspecified, max_queue_size will default to 10.
        :type max_queue_size: int
        :param workers: Used for generator or keras.utils.Sequence input only. Maximum number of processes to
            spin up when using process-based threading. If unspecified, workers will default to 1. If 0, will execute
            the generator on the main thread.
        :type workers: int
        :param use_multiprocessing: Used for generator or keras.utils.Sequence input only. If True, use process-based
            threading. If unspecified, use_multiprocessing will default to False. Note that because this
            implementation relies on multiprocessing, you should not pass non-picklable arguments to the generator as
            they can't be passed easily to children processes.
        :type use_multiprocessing: bool
        :param return_dict: If True, loss and metric results are returned as a dict, with each key being the name of
            the metric. If False, they are returned as a list.
        :type return_dict: bool
        :return: Returns loss metric results either as dict or list depending on specified parameters
        :rtype: list
        """
        return self.model.evaluate(x=x, y=y, batch_size=batch_size, verbose=verbose, sample_weight=sample_weight,
                                   steps=steps,
                                   callbacks=callbacks, max_queue_size=max_queue_size, workers=workers,
                                   use_multiprocessing=use_multiprocessing,
                                   return_dict=return_dict)

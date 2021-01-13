![GitHub release (latest by date)](https://img.shields.io/github/v/release/KristofPusztai/EQL?style=plastic)
# Introduction:

A tensorflow implementation of the Equation Learner Neural Network based model:
![Imgur](https://i.imgur.com/L77pz3d.png)
- https://arxiv.org/pdf/1610.02995.pdf

# Functionality:
### Installation:
pip:

    pip install EQL-NN
or from github:
    
    git clone https://github.com/KristofPusztai/EQL.git
Navigate to cloned directory and run setup.py
    
    sudo python setup.py install

### Creating and training a model:

    from EQL.model import EQL
    EQLmodel = EQL(num_layers = 2, dim=5, v = [1,1]) # num_layers -> hidden_layers, dim -> dimension of input, v -> number of binary inputs
    x = tf.random_normal_initializer()(shape=(100, 5))
    y = tf.random_normal_initializer()(shape=(100, 1))
    
    EQLmodel.build_and_compile(self, metrics=None, loss_weights=None, weighted_metrics=None,
                               run_eagerly=None, kernel_regularizer=None,
                               w_init='random_normal', b_init='random_normal, exclude=None) # exclude specifies activation function exclusions in layers
                               
    EQLmodel.fit(x, y, lmbda, t0=100, t1=0, t2=0, initial_epoch=0, verbose=0, batch_size=None, callbacks=None,
            validation_split=0.0, validation_data=None, shuffle=True, class_weight=None,
            sample_weight=None, steps_per_epoch=None,
            validation_steps=None, validation_batch_size=None, validation_freq=1,
            max_queue_size=10, workers=1, use_multiprocessing=False, atol=0.01)
    
### Prediction:
    
    EQLmodel.predict(x, batch_size=None, verbose=0, steps=None, 
                     callbacks=None, max_queue_size=10,
                     workers=1, use_multiprocessing=False)
    
Note: use tensor input for maximum computational efficiency

### Model Information:
There are a variety of methods for understanding what is going on
in your model.

    EQLmodel.summary() # Provides tensorflow summary
    EQLmodel.count_params() # Provides # trainable params
    EQLmodel.get_weights(layer) #returns array of layer values
    EQLmodel.set_weights(layer, weights) #sets weights of specified layer
    
    EQLmodel.evaluate(x=None, y=None, batch_size=None, verbose=1,
                      sample_weight=None, steps=None,
                      callbacks=None, max_queue_size=10, 
                      workers=1, use_multiprocessing=False,
                      return_dict=False)
    # Returns the loss value & metrics values for the model
## Implementation Notes:
Training regiment is interpretted as debiased LASSO:
- T0 epochs are normal training, no regularization
- T1 epochs are L1 regularized training, continuing where T0 ended
- T2 epochs are L0 (weights with values close to 0, |w| < atol, are rounded to 0 and left untrained) regularized training, continuing from T1
# Introduction:

A tensorflow implementation of the Equation Learner Neural Network based model:
![Imgur](https://i.imgur.com/L77pz3d.png)
- https://arxiv.org/pdf/1610.02995.pdf

# Functionality:
Installation:

    pip install EQL
    
Creating and training a model:

    from EQL.model import EQL
    EQLmodel = EQL(num_layers = 2, dim=5)
    x = tf.random_normal_initializer()(shape=(100, 5))
    y = tf.random_normal_initializer()(shape=(100, 1))
    
    EQLmodel.build_and_compile(self, metrics=None, loss_weights=None, weighted_metrics=None,
                               run_eagerly=None, kernel_regularizer=None)
                               
    EQLmodel.fit(x, y, weights=None, biases=None, epochs=100, verbose=0, validation_split=0.2,
                 loss=tf.keras.losses.MeanSquaredError(), metrics=None, loss_weights=None, weighted_metrics=None,
                 run_eagerly=None, steps_per_execution=None)
    
Prediction:
    
    EQLmodel.predict(x, batch_size=None, verbose=0, steps=None, 
                     callbacks=None, max_queue_size=10,
                     workers=1, use_multiprocessing=False)
    
Note: use tensor input for maximum computational efficiency

Model Information:

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
# Introduction:

A tensorflow implementation of the Equation Learner Neural Network based model:
![Imgur](https://i.imgur.com/L77pz3d.png)
- https://arxiv.org/pdf/1610.02995.pdf

# Functionality:
Installation:

    pip install EQL
    
Creating and training a model:

    from EQL.model import EQL
    EQLmodel = EQL(dim=5)
    x = tf.random_normal_initializer()(shape=(100, 5))
    y = tf.random_normal_initializer()(shape=(100, 1))
    EQLmodel.fit(x,y,verbose = 1)
    
    #fit(self, x, y, weights=None, biases=None, epochs=100, verbose=0, validation_split=0.2,
         loss=tf.keras.losses.MeanSquaredError(), metrics=None, loss_weights=None, weighted_metrics=None,
         run_eagerly=None, steps_per_execution=None)
Prediction:
    
    EQLmodel.predict(x, verbose=1)
    
    #predict(x, batch_size=batch_size, verbose=verbose, 
            steps=steps, callbacks=callbacks,
            max_queue_size=max_queue_size, workers=workers,
            use_multiprocessing=use_multiprocessing)
Model Information:

    EQLmodel.summary() # Provides tensorflow summary
    EQLmodel.count_params() # Provides # trainable params
    EQLmodel.evaluate(x, y, verbose = 1) # Returns the loss value & metrics values for the model in test mode.
    
    #evaluate(x=x, y=y, batch_size=batch_size, verbose=verbose, sample_weight=sample_weight,
              steps=steps,
              callbacks=callbacks, max_queue_size=max_queue_size, workers=workers,
              use_multiprocessing=use_multiprocessing,
              return_dict=return_dict)
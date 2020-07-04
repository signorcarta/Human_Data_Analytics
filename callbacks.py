from tensorflow import keras

"Stop training when a monitored metric has stopped improving"
early_stopper = keras.callbacks.EarlyStopping(monitor="loss", # must be passed as last
                                           min_delta=1e-7,
                                           patience=2,
                                           verbose=0,
                                           mode="auto")

"Reduce learning rate when a metric has stopped improving"
lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor="loss",
                                                 factor=0.1,
                                                 patience=2,
                                                 verbose=0,
                                                 mode="auto",
                                                 min_delta=1e-4,
                                                 cooldown=1,
                                                 min_lr=1e-3, )

"Terminate training when a NaN loss is encountered"
nan_loss_stopper = keras.callbacks.TerminateOnNaN()


my_callbacks = [lr_reducer, nan_loss_stopper, early_stopper]

import tensorflow as tf

from t_Mod import plots_t

MAX_EPOCHS = 300

def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val, verbose=0
                      ,callbacks=[early_stopping])
  return history


 #%%
def compile_2(model):
 # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min')
  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])
  return


#%%

def fit_2(model, window,   patience=2, EPOCHS=None):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')
    if EPOCHS == 'True' :
        history = model.fit(window.train, epochs=plots_t.MAX_EPOCHS,
                      validation_data=window.val, verbose=0,
                      callbacks=[early_stopping])

    else :
        history = model.fit(window.train, epochs= MAX_EPOCHS,
                      validation_data=window.val, verbose=0,
                      callbacks=[early_stopping])

    return history


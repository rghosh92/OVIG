###### import sys
import sys
assert sys.version_info >= (3, 5)
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import tensorflow_probability as tfp
np.random.seed(42)
tf.random.set_seed(42)

randN_05 = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)


def get_network(dim_x,dim_y):
    tf.keras.backend.clear_session()
    
    input_A = keras.layers.Input(shape=[dim_x+dim_y])
    input_B = keras.layers.Input(shape=[dim_x+dim_y])

    transform = keras.models.Sequential([
    layers.Dense(30, kernel_initializer=randN_05, activation="relu"),
    keras.layers.Dropout(rate=0.3), # To regularize higher dimensionality
    layers.Dense(30, kernel_initializer=randN_05, activation="relu"),
    keras.layers.Dropout(rate=0.3), # To regularize higher dimensionality
    layers.Dense(1, kernel_initializer=randN_05, activation=None)])

    output_A = transform(input_A)
    output_B = transform(input_B)
    output_C = tf.reduce_mean(output_A) - tf.math.log(tf.reduce_mean(tf.exp(output_B))) # MINE
    #output_C = tf.reduce_mean(output_A) - tf.reduce_mean(tf.exp(output_B))+1 # MINE-f
    MI_mod = keras.models.Model(inputs=[input_A, input_B], outputs=output_C)
    MI_mod.compile(loss=loss_func, optimizer=keras.optimizers.Nadam(lr=0.001))
    return MI_mod

    

def loss_func(inp, outp):
    '''Calculate the loss: scaled negative estimated mutual information'''
    return -outp

def MINE_ready(x_sample, y_sample):
    x_sample1, x_sample2 = tf.split(x_sample, num_or_size_splits=2)
    y_sample1, y_sample2 = tf.split(y_sample, num_or_size_splits=2)
    
     # Ensure both tensors are of type float32
    x_sample1 = tf.cast(x_sample1, dtype=tf.float32)
    x_sample2 = tf.cast(x_sample2, dtype=tf.float32)
    y_sample1 = tf.cast(y_sample1, dtype=tf.float32)
    y_sample2 = tf.cast(y_sample2, dtype=tf.float32)
    
    joint_sample = tf.concat([x_sample1, y_sample1], axis=1)
    marg_sample = tf.concat([x_sample2, y_sample1], axis=1)
    return joint_sample,marg_sample

def MINE_MI(x_sample,y_sample,total_epochs=50):
    joint_sample,marg_sample = MINE_ready(x_sample,y_sample)
    MI_mod = get_network(x_sample.shape[-1],y_sample.shape[-1])
    MI_mod.compile(loss=loss_func, optimizer=keras.optimizers.Adam(lr=0.001,decay=5e-4))
    history_mi = MI_mod.fit((joint_sample, marg_sample), x_sample[0:int(x_sample.shape[0]/2)], epochs=total_epochs,batch_size=200,verbose=0)
    return -np.log2(np.exp(1))*history_mi.history['loss'][-1],history_mi

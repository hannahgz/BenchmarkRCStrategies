import keras_genomics
import tensorflow as tf
import keras 
import numpy as np
import os
import functools

from keras import backend as K 
from keras.layers.core import Dropout 
from keras.layers.core import Flatten
from keras.engine import Layer
from keras.models import Sequential 
import keras.layers as kl
from keras.engine.base_layer import InputSpec
from keras_genomics.layers import RevCompConv1D
from keras import initializers
 

from keras.layers import Input
from keras.models import Model
from keras.models import load_model
from numpy.random import seed
from tensorflow import set_random_seed
from keras.callbacks import EarlyStopping, History, ModelCheckpoint

#Used to preserve RC symmetry
class RevCompSumPool(Layer): 
    def __init__(self, **kwargs): 
        super(RevCompSumPool, self).__init__(**kwargs)

    def build(self, input_shape):
        self.num_input_chan = input_shape[2]
        super(RevCompSumPool, self).build(input_shape)

    def call(self, inputs): 
        inputs = (inputs[:,:,:int(self.num_input_chan/2)] + inputs[:,:,int(self.num_input_chan/2):][:,::-1,::-1])
        return inputs
      
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], int(input_shape[2]/2))
    
    
class WeightedSum1D(Layer):
    '''Learns a weight for each position, for each channel, and sums
    lengthwise.
    # Arguments
        symmetric: if want weights to be symmetric along length, set to True
        input_is_revcomp_conv: if the input is [RevCompConv1D], set to True for
            added weight sharing between reverse-complement pairs
        smoothness_penalty: penalty to be applied to absolute difference
            of adjacent weights in the length dimension
        bias: whether or not to have bias parameters
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    '''
    def __init__(self, symmetric, input_is_revcomp_conv,
                       smoothness_penalty=None, bias=False,
                       **kwargs):
        super(WeightedSum1D, self).__init__(**kwargs)
        self.symmetric = symmetric
        self.input_is_revcomp_conv = input_is_revcomp_conv
        self.smoothness_penalty = smoothness_penalty
        self.bias = bias
        self.input_spec = [InputSpec(ndim=3)]

    def build(self, input_shape): 
        #input_shape[0] is the batch index
        #input_shape[1] is length of input
        #input_shape[2] is number of filters

        #Equivalent to 'fanintimesfanouttimestwo' from the paper
        limit = np.sqrt(6.0/(input_shape[1]*input_shape[2]*2))  
        self.init = initializers.uniform(-1*limit, limit)

        if (self.symmetric == False):
            W_length = input_shape[1]
        else:
            self.odd_input_length = input_shape[1]%2.0 == 1
            #+0.5 below turns floor into ceil
            W_length = int(input_shape[1]/2.0 + 0.5)

        if (self.input_is_revcomp_conv == False):
            W_chan = input_shape[2]
        else:
            assert input_shape[2]%2==0,\
             "if input is revcomp conv, # incoming channels would be even"
            W_chan = int(input_shape[2]/2)

        self.W_shape = (W_length, W_chan)
        self.b_shape = (W_chan,)
        self.W = self.add_weight(self.W_shape,
             initializer=self.init,
             name='{}_W'.format(self.name),
             regularizer=(None if self.smoothness_penalty is None else
                         regularizers.SmoothnessRegularizer(
                          self.smoothness_penalty)))
        if (self.bias):
            assert False, "No bias was specified in original experiments"

        self.built = True

    #3D input -> 2D output (loses length dimension)
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def call(self, x, mask=None):
        if (self.symmetric == False):
            W = self.W
        else:
            W = K.concatenate(
                 tensors=[self.W,
                          #reverse along length, concat along length
                          self.W[::-1][(1 if self.odd_input_length else 0):]],
                 axis=0)
        if (self.bias):
            b = self.b
        if (self.input_is_revcomp_conv):
            #reverse along both length and channel dims, concat along chan
            #if symmetric=True, reversal along length here makes no diff
            W = K.concatenate(tensors=[W, W[::-1,::-1]], axis=1)
            if (self.bias):
                b = K.concatenate(tensors=[b, b[::-1]], axis=0)
        output = K.sum(x*K.expand_dims(W,0), axis=1)
        if (self.bias):
            output = output + K.expand_dims(b,0)
        return output 

    def get_config(self):
        config = {'symmetric': self.symmetric,
                  'input_is_revcomp_conv': self.input_is_revcomp_conv,
                  'smoothness_penalty': self.smoothness_penalty,
                  'bias': self.bias}
        base_config = super(WeightedSum1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
def get_rc_model(parameters, is_weighted_sum, use_bias = False):
    rc_model = keras.models.Sequential()
    rc_model.add(keras_genomics.layers.convolutional.RevCompConv1D(
        input_shape=(1000,4), nb_filter=16, filter_length=15))    
    rc_model.add(keras_genomics.layers.normalization.RevCompConv1DBatchNorm())
    rc_model.add(kl.core.Activation("relu"))
    rc_model.add(keras_genomics.layers.convolutional.RevCompConv1D(
        nb_filter=16, filter_length=14))
    rc_model.add(keras_genomics.layers.normalization.RevCompConv1DBatchNorm())
    rc_model.add(kl.core.Activation("relu"))
    rc_model.add(keras_genomics.layers.convolutional.RevCompConv1D(
        nb_filter=16, filter_length=14))
    rc_model.add(keras_genomics.layers.normalization.RevCompConv1DBatchNorm())
    rc_model.add(kl.core.Activation("relu"))
    rc_model.add(keras.layers.convolutional.MaxPooling1D(
        pool_length = parameters['pool_size'], strides = parameters['strides']))
    if is_weighted_sum:
        rc_model.add(WeightedSum1D(
            symmetric=False, input_is_revcomp_conv=True))
        rc_model.add(kl.Dense(output_dim=1, trainable=False,
                              init="ones"))
    else:                 
        rc_model.add(RevCompSumPool())
        rc_model.add(Flatten())
        rc_model.add(keras.layers.core.Dense(output_dim=1, trainable=True,
                                             init="glorot_uniform", use_bias = use_bias))
    rc_model.add(kl.core.Activation("sigmoid"))
    rc_model.compile(optimizer = keras.optimizers.Adam(lr=0.001), 
                     loss="binary_crossentropy", metrics=["accuracy"])
    return rc_model     


def get_reg_model(parameters):
    model = keras.models.Sequential()
    model.add(keras.layers.Convolution1D(
              input_shape=(1000,4), nb_filter=16, filter_length=15))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(keras.layers.core.Activation("relu"))
    model.add(keras.layers.convolutional.Convolution1D(
            nb_filter=16, filter_length=14))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.convolutional.Convolution1D(
            nb_filter=16, filter_length=14))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.convolutional.MaxPooling1D(pool_length=parameters['pool_size'],
                                                      strides= parameters['strides']))         
    model.add(Flatten())
    model.add(keras.layers.core.Dense(output_dim=1, trainable=True,
                                    init="glorot_uniform"))
    model.add(keras.layers.core.Activation("sigmoid"))
    model.compile(optimizer = keras.optimizers.Adam(lr=0.001), 
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


def get_siamese_model(parameters):
    main_input = Input(shape=(1000, 4))
    rev_input = kl.Lambda(lambda x: x[:,::-1, ::-1])(main_input)
    
    s_model = Sequential([
        keras.layers.Convolution1D(
              input_shape=(1000,4), nb_filter=16, filter_length=15),
        keras.layers.normalization.BatchNormalization(), 
        keras.layers.core.Activation("relu"),
        keras.layers.convolutional.Convolution1D(
            nb_filter=16, filter_length=14),
        keras.layers.normalization.BatchNormalization(), 
        keras.layers.core.Activation("relu"),
        keras.layers.convolutional.Convolution1D(
            nb_filter=16, filter_length=14),
        keras.layers.normalization.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.convolutional.MaxPooling1D(pool_length=parameters['pool_size'],
                                                      strides= parameters['strides']),
        Flatten(), 
        keras.layers.core.Dense(output_dim=1, trainable=True,
                                    init="glorot_uniform"),     
    ],  name = "shared_layers")

    main_output = s_model(main_input)
    rev_output = s_model(rev_input)
    
    avg = kl.average([main_output, rev_output])
    
    final_out = kl.core.Activation("sigmoid")(avg)
    siamese_model = Model(inputs = main_input, outputs = final_out)
    siamese_model.compile(optimizer = keras.optimizers.Adam(lr=0.001), 
                  loss="binary_crossentropy", metrics=["accuracy"])
    return siamese_model


#The difference between this siamese model and the one above is when the averaging takes place
def get_new_siamese_model(parameters):
    main_input = Input(shape=(1000, 4))
    rev_input = kl.Lambda(lambda x: x[:,::-1, ::-1])(main_input)
    
    s_model = Sequential([
        keras.layers.Convolution1D(
              input_shape=(1000,4), nb_filter=16, filter_length=15),
        keras.layers.normalization.BatchNormalization(), 
        keras.layers.core.Activation("relu"),
        keras.layers.convolutional.Convolution1D(
            nb_filter=16, filter_length=14),
        keras.layers.normalization.BatchNormalization(), 
        keras.layers.core.Activation("relu"),
        keras.layers.convolutional.Convolution1D(
            nb_filter=16, filter_length=14),
        keras.layers.normalization.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.convolutional.MaxPooling1D(pool_length=parameters['pool_size'],
                                                      strides= parameters['strides']),
        Flatten(), 
        keras.layers.core.Dense(output_dim=1, trainable=True,
                                    init="glorot_uniform"),     
    ],  name = "shared_layers")

    main_output = s_model(main_input)
    rev_output = s_model(rev_input)
    
    final_out_main = kl.core.Activation("sigmoid")(main_output)
    final_out_rev = kl.core.Activation("sigmoid")(rev_output)
    
    avg = kl.average([final_out_main, final_out_rev])
    
    siamese_model = Model(inputs = main_input, outputs = avg)
    siamese_model.compile(optimizer = keras.optimizers.Adam(lr=0.001), 
                  loss="binary_crossentropy", metrics=["accuracy"])
    return siamese_model
                     
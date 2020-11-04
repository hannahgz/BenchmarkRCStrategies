import keras_genomics
import tensorflow as tf
import keras 
import keras_genomics
import numpy as np
import os

from keras import backend as K 

import keras.layers as kl
from keras.layers import Input
from keras.layers.core import Dropout 
from keras.layers.core import Flatten

from keras.engine import Layer
from keras.engine.base_layer import InputSpec

from keras.models import Sequential 
from keras.models import Model
from keras.models import load_model
from keras_genomics.layers import RevCompConv1D

from numpy.random import seed
from tensorflow import set_random_seed
from keras.callbacks import EarlyStopping, History, ModelCheckpoint
  
    
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
    
    
class RegArch():
    def __init__(self, filters, kernel_size, 
                 input_length, pool_size, strides):
        self.filters = filters 
        self.kernel_size = kernel_size 
        self.input_length = input_length 
        self.pool_size = pool_size 
        self.strides = strides 
        self.name = "Standard"
        
    def get_model(self):
        reg_model = keras.models.Sequential()
        reg_model.add(kl.Conv1D(filters = self.filters,
                                kernel_size = self.kernel_size,
                                input_shape = (self.input_length,4),
                                strides = 1))
        reg_model.add(kl.BatchNormalization())
        reg_model.add(kl.core.Activation("relu"))
        reg_model.add(kl.pooling.MaxPooling1D(pool_size = self.pool_size,
                                              strides = self.strides))
        reg_model.add(Flatten())
        reg_model.add(kl.Dense(units = 3))
        reg_model.add(kl.core.Activation("sigmoid"))
        opt = keras.optimizers.Adam(lr=0.001)
        reg_model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
        return reg_model
        
        
class RCArch(RegArch): 
    def __init__(self, **kwargs): 
        self.name = "RC"
        super(RCArch, self).__init__(**kwargs)
                 
    def get_model(self): 
        rc_model = keras.models.Sequential()
        rc_model.add(keras_genomics.layers.RevCompConv1D(filters = self.filters,
                                                         kernel_size = self.kernel_size, 
                                                         input_shape = (self.input_length,4), 
                                                         strides = 1))    
        rc_model.add(keras_genomics.layers.RevCompConv1DBatchNorm())
        rc_model.add(kl.core.Activation("relu"))
        rc_model.add(kl.pooling.MaxPooling1D(pool_size = self.pool_size, 
                                             strides = self.strides))

        rc_model.add(RevCompSumPool())
        rc_model.add(Flatten())
        rc_model.add(kl.Dense(units = 3))
        rc_model.add(kl.core.Activation("sigmoid"))
        opt = keras.optimizers.Adam(lr=0.001)
        rc_model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
        return rc_model
         
        
class SiameseArch(RegArch): 
    def __init__(self, **kwargs): 
        self.name = "Siamese"
        super(SiameseArch, self).__init__(**kwargs) 
                 
    def get_model(self): 
        main_input = Input(shape=(self.input_length, 4, ))
        rev_input = kl.Lambda(lambda x: x[:,::-1, ::-1])(main_input)

        s_model = Sequential([
            kl.Conv1D(filters = self.filters,
                      kernel_size = self.kernel_size,
                      input_shape =(self.input_length,4),
                      strides = 1),
            kl.BatchNormalization(), 
            kl.core.Activation("relu"),
            kl.pooling.MaxPooling1D(pool_size = self.pool_size,
                                              strides = self.strides), 
            Flatten(), 
            kl.Dense(units = 3),     
        ],  name = "shared_layers")

        main_output = s_model(main_input)
        rev_output = s_model(rev_input)

        avg = kl.average([main_output, rev_output])

        final_out = kl.core.Activation("sigmoid")(avg)
        siamese_model = Model(inputs = main_input, outputs = final_out)
        opt = keras.optimizers.Adam(lr=0.001)
        siamese_model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
        return siamese_model
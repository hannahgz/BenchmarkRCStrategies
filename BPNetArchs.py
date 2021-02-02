from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras_genomics
from keras_genomics.layers.convolutional import RevCompConv1D
import keras
import keras.layers as kl
import tensorflow as tf
import numpy as np
import seqdataloader
from seqdataloader.batchproducers import coordbased
from seqdataloader.batchproducers.coordbased import coordstovals
from seqdataloader.batchproducers.coordbased import coordbatchproducers
from seqdataloader.batchproducers.coordbased import coordbatchtransformers
from keras.layers.core import Dropout 
from keras.engine import Layer
from keras.engine.base_layer import InputSpec
from keras.callbacks import History
from keras.optimizers import Optimizer
from keras import backend as K

from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import keras_export


#Loss Function
def multinomial_nll(true_counts, logits):
    """Compute the multinomial negative log-likelihood
    Args:
      true_counts: observed count values
      logits: predicted logit values
    """
    counts_per_example = tf.reduce_sum(true_counts, axis=-1)
    dist = tf.compat.v1.distributions.Multinomial(total_count=counts_per_example,
                                         logits=logits)
    return (-tf.reduce_sum(dist.log_prob(true_counts)) / 
            tf.cast((tf.shape(true_counts)[0]), tf.float32))


#from https://github.com/kundajelab/basepair/blob/cda0875571066343cdf90aed031f7c51714d991a/basepair/losses.py#L87
class MultichannelMultinomialNLL(object):
    def __init__(self, n):
        self.__name__ = "MultichannelMultinomialNLL"
        self.n = n

    def __call__(self, true_counts, logits):
        for i in range(self.n):
            loss = multinomial_nll(true_counts[..., i], logits[..., i])
            if i == 0:
                total = loss
            else:
                total += loss
        return total

    def get_config(self):
        return {"n": self.n}

    
class AbstractProfileModel(object):
    def __init__(self, dataset, input_seq_len, c_task_weight, p_task_weight, 
                 filters, n_dil_layers, conv1_kernel_size, dil_kernel_size,
                 outconv_kernel_size, optimizer, weight_decay, lr, 
                 kernel_initializer, seed):
        self.dataset = dataset
        self.input_seq_len = input_seq_len
        self.c_task_weight = c_task_weight
        self.p_task_weight = p_task_weight
        self.filters = filters
        self.n_dil_layers = n_dil_layers
        self.conv1_kernel_size = conv1_kernel_size
        self.dil_kernel_size = dil_kernel_size
        self.outconv_kernel_size = outconv_kernel_size
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.lr = lr
        self.learning_rate = lr
        self.kernel_initializer = kernel_initializer
        self.seed = seed
    
    def get_embedding_len(self):
        embedding_len = self.input_seq_len
        embedding_len -= (self.conv1_kernel_size-1)     
        for i in range(1, self.n_dil_layers+1):
            dilation_rate = (2**i)
            embedding_len -= dilation_rate*(self.dil_kernel_size-1)
        return embedding_len
    
    def get_output_profile_len(self):
        embedding_len = self.get_embedding_len()
        out_profile_len = embedding_len - (self.outconv_kernel_size - 1)
        return out_profile_len
    
    def trim_flanks_of_conv_layer(self, conv_layer, output_len, width_to_trim, filters):
        layer = keras.layers.Lambda(
            lambda x: x[:,
              int(0.5*(width_to_trim)):-(width_to_trim-int(0.5*(width_to_trim)))],
            output_shape=(output_len, filters))(conv_layer)
        return layer 
    
    
    def get_inputs(self):
        out_pred_len = self.get_output_profile_len()
        
        inp = kl.Input(shape=(self.input_seq_len, 4), name='sequence')
        bias_counts_input = kl.Input(shape=(1,), name="patchcap.logcount")
        #if working with raw counts, go from logcount->count
        bias_profile_input = kl.Input(shape=(out_pred_len, 2),
                                    name="patchcap.profile")
        return inp, bias_counts_input, bias_profile_input
    
    def get_names(self): 
        countouttaskname = "CHIPNexus.%s.logcount" % self.dataset
        profileouttaskname = "CHIPNexus.%s.profile" % self.dataset
        return countouttaskname, profileouttaskname
            
    def get_keras_model(self): 
        raise NotImplementedError()

        
class RCBPNetArch(AbstractProfileModel):   
    def __init__(self, is_add, **kwargs):
        super().__init__(**kwargs)
        self.is_add = is_add
        
    def get_keras_model(self):
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)
        
        inp, bias_counts_input, bias_profile_input = self.get_inputs()
        countouttaskname, profileouttaskname = self.get_names()
        
        out_pred_len = self.get_output_profile_len()
        curr_layer_size = self.input_seq_len - (self.conv1_kernel_size-1)
        
        first_conv = RevCompConv1D(filters=self.filters,
                                   kernel_size=self.conv1_kernel_size,
                                   kernel_initializer = self.kernel_initializer,
                                   padding='valid',
                                   activation='relu')(inp)

        prev_layers = first_conv
        for i in range(1, self.n_dil_layers + 1):
            dilation_rate = 2**i
    
            conv_output = RevCompConv1D(filters=self.filters,
                                        kernel_size=self.dil_kernel_size,
                                        kernel_initializer = self.kernel_initializer,
                                        padding='valid',
                                        activation='relu',
                                        dilation_rate=dilation_rate)(prev_layers)   

            width_to_trim = dilation_rate*(self.dil_kernel_size-1)

            curr_layer_size = (curr_layer_size - width_to_trim)
            

            prev_layers = self.trim_flanks_of_conv_layer(
                conv_layer = prev_layers, output_len = curr_layer_size, 
                width_to_trim = width_to_trim, filters = 2 * self.filters)
            
            if(self.is_add): 
                prev_layers = kl.add([prev_layers, conv_output])
            else:
                prev_layers = kl.average([prev_layers, conv_output])

        combined_conv = prev_layers

        #Counts prediction
        gap_combined_conv = kl.GlobalAvgPool1D()(combined_conv)
        count_out = kl.Reshape((-1,), name=countouttaskname)(
            RevCompConv1D(filters=1, kernel_size=1, kernel_initializer = self.kernel_initializer)(
              kl.Reshape((1,-1))(kl.concatenate([
                  #concatenation of the bias layer both before and after
                  # is needed for rc symmetry
                  kl.Lambda(lambda x: x[:, ::-1])(bias_counts_input),
                  gap_combined_conv,
                  bias_counts_input], axis=-1))))

        #Profile prediction
        profile_out_prebias = RevCompConv1D(
            filters=1,kernel_size=self.outconv_kernel_size,
            kernel_initializer = self.kernel_initializer, padding='valid')(combined_conv)
        
        profile_out = RevCompConv1D(
            filters=1, kernel_size=1, name=profileouttaskname, kernel_initializer = self.kernel_initializer)(
                    kl.concatenate([
                        #concatenation of the bias layer both before and after
                        # is needed for rc symmetry
                        kl.Lambda(lambda x: x[:, :, ::-1])(bias_profile_input),
                        profile_out_prebias,
                        bias_profile_input], axis=-1))
            
        model = keras.models.Model(
          inputs=[inp, bias_counts_input, bias_profile_input],
          outputs=[count_out, profile_out])
        
        model.compile(keras.optimizers.Adam(lr=self.lr),
                      loss=['mse', MultichannelMultinomialNLL(2)],
                      loss_weights=[self.c_task_weight, self.p_task_weight]) 
          
        return model
    
    
class CJTrainedBPNetArch(AbstractProfileModel):
    def __init__(self, is_add, **kwargs):
        super().__init__(**kwargs)
        self.is_add = is_add
    
    def trim_flanks_of_conv_layer_revcomp(self, conv_layer, output_len, width_to_trim, filters):
        layer = keras.layers.Lambda(
            lambda x: x[:,
              (width_to_trim-int(0.5*(width_to_trim))):-int(0.5*(width_to_trim))],
            output_shape=(output_len, filters))(conv_layer)
        return layer 
        
    def get_keras_model(self):
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)
        
        inp, bias_counts_input, bias_profile_input = self.get_inputs()
        rev_inp = kl.Lambda(lambda x: x[:,::-1,::-1])(inp)
        
        countouttaskname, profileouttaskname = self.get_names()
        
        first_conv = kl.Conv1D(self.filters,
                               kernel_size=self.conv1_kernel_size,
                               kernel_initializer = self.kernel_initializer,
                               padding='valid',
                               activation='relu')
        first_conv_fwd = first_conv(inp)
        first_conv_rev = first_conv(rev_inp)

        curr_layer_size = self.input_seq_len - (self.conv1_kernel_size-1)
        
        prev_layers_fwd = first_conv_fwd
        prev_layers_rev = first_conv_rev

        for i in range(1, self.n_dil_layers + 1):
            dilation_rate = 2**i
            conv_output = kl.Conv1D(self.filters, kernel_size=self.dil_kernel_size, 
                                    padding='valid',
                                    kernel_initializer = self.kernel_initializer,
                                    activation='relu', 
                                    dilation_rate=dilation_rate)

            conv_output_fwd = conv_output(prev_layers_fwd)
            conv_output_rev = conv_output(prev_layers_rev)
            
            width_to_trim = dilation_rate * (self.dil_kernel_size - 1)
            
            curr_layer_size = (curr_layer_size - width_to_trim)
            
            prev_layers_fwd = self.trim_flanks_of_conv_layer(
                conv_layer = prev_layers_fwd, output_len = curr_layer_size, 
                width_to_trim = width_to_trim, filters = self.filters)

            prev_layers_rev = self.trim_flanks_of_conv_layer_revcomp(
                conv_layer = prev_layers_rev, output_len = curr_layer_size, 
                width_to_trim = width_to_trim, filters = self.filters)
            
            if(self.is_add):
                prev_layers_fwd = kl.add([prev_layers_fwd, conv_output_fwd])
                prev_layers_rev = kl.add([prev_layers_rev, conv_output_rev])
            else: 
                prev_layers_fwd = kl.average([prev_layers_fwd, conv_output_fwd])
                prev_layers_rev = kl.average([prev_layers_rev, conv_output_rev])
                
            combined_conv_fwd = prev_layers_fwd
            combined_conv_rev = prev_layers_rev

        #Counts Prediction
        counts_dense_layer = kl.Dense(2,kernel_initializer = self.kernel_initializer,)
        gap_combined_conv_fwd = kl.GlobalAvgPool1D()(combined_conv_fwd)
        gap_combined_conv_rev = kl.GlobalAvgPool1D()(combined_conv_rev)
        
        main_count_out_fwd = counts_dense_layer(
            kl.concatenate([gap_combined_conv_fwd, bias_counts_input], axis=-1))
        
        main_count_out_rev = counts_dense_layer(
            kl.concatenate([bias_counts_input, gap_combined_conv_rev], axis=-1))
        rc_rev_count_out = kl.Lambda(lambda x: x[:,::-1])(main_count_out_rev)
        
        avg_count_out = kl.Average(name = countouttaskname)(
            [main_count_out_fwd, rc_rev_count_out])

        #Profile Prediction
        profile_penultimate_conv = kl.Conv1D(filters = 2, 
                                             kernel_size = self.outconv_kernel_size,
                                             kernel_initializer = self.kernel_initializer,
                                             padding = 'valid')
        profile_final_conv = kl.Conv1D(2, kernel_size=1, kernel_initializer = self.kernel_initializer,)
        
        profile_out_prebias_fwd = profile_penultimate_conv(combined_conv_fwd)
        main_profile_out_fwd = profile_final_conv(kl.concatenate(
            [profile_out_prebias_fwd, bias_profile_input], axis=-1))

        profile_out_prebias_rev = profile_penultimate_conv(combined_conv_rev)
        rev_bias_profile_input = kl.Lambda(lambda x: x[:,::-1,:])(bias_profile_input)
        main_profile_out_rev = profile_final_conv(kl.concatenate(
            [profile_out_prebias_rev, rev_bias_profile_input], axis=-1))
        rc_rev_profile_out = kl.Lambda(lambda x: x[:,::-1,::-1])(main_profile_out_rev)

        avg_profile_out = kl.Average(name = profileouttaskname)(
            [main_profile_out_fwd, rc_rev_profile_out])    
        
        model = keras.models.Model(
          inputs=[inp, bias_counts_input, bias_profile_input],
          outputs=[avg_count_out, avg_profile_out])
        
        model.compile(keras.optimizers.Adam(lr=self.lr),
                      loss=['mse', MultichannelMultinomialNLL(2)],
                      loss_weights=[self.c_task_weight, self.p_task_weight]) 
     
        return model
    
    
class CJRCWrapperArch(AbstractProfileModel): 
    def __init__(self,  **kwargs):
        super().__init__(**kwargs) 
        
    def trim_flanks_of_conv_layer_revcomp(self, conv_layer, output_len, width_to_trim, filters):
        layer = keras.layers.Lambda(
            lambda x: x[:,
              (width_to_trim-int(0.5*(width_to_trim))):-int(0.5*(width_to_trim))],
            output_shape=(output_len, filters))(conv_layer)
        return layer 
        
    def apply_conjoined_rcps(self, curr_submodel, input_tensor):
        RC_LAYER = keras.layers.Lambda(lambda x: x[:, ::-1, ::-1])
        
        #run on the original input tensor
        fwd_out = curr_submodel(input_tensor)
        #take revcomp of the input tensor
        rc_input_tensor = RC_LAYER(input_tensor)
        #run on the revcomp of the input tensor
        rev_out = curr_submodel(rc_input_tensor)
        #reverse the revcomp
        rc_rev_out = RC_LAYER(rev_out)
        #concatenate rc_rev_out with fwd_out along the
        # channel axis, achieving rc equivariance
        final_out = keras.layers.Concatenate(axis=2)([fwd_out, rc_rev_out])
        return final_out
    
    def get_keras_model(self):
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)
        
        inp, bias_counts_input, bias_profile_input = self.get_inputs()
        countouttaskname, profileouttaskname = self.get_names()
        #define the submodel
        submodel_inp = keras.layers.Input(shape=(self.input_seq_len, 4))
        first_conv = kl.Conv1D(self.filters,
                               kernel_size=self.conv1_kernel_size,
                               kernel_initializer = self.kernel_initializer,
                               padding='valid',
                               activation='relu')
        first_conv_fwd = first_conv(submodel_inp)
        curr_layer_size = self.input_seq_len - (self.conv1_kernel_size-1)
        prev_layers_fwd = first_conv_fwd

        for i in range(1, self.n_dil_layers + 1):
            dilation_rate = 2**i
            conv_output = kl.Conv1D(self.filters, kernel_size=self.dil_kernel_size, 
                                    padding='valid',
                                    kernel_initializer = self.kernel_initializer,
                                    activation='relu', 
                                    dilation_rate=dilation_rate)

            conv_output_fwd = conv_output(prev_layers_fwd)
            
            width_to_trim = dilation_rate * (self.dil_kernel_size - 1)
            
            curr_layer_size = (curr_layer_size - width_to_trim)
            
            prev_layers_fwd = self.trim_flanks_of_conv_layer(
                conv_layer = prev_layers_fwd, output_len = curr_layer_size, 
                width_to_trim = width_to_trim, filters = self.filters)
            
            prev_layers_fwd = kl.add([prev_layers_fwd, conv_output_fwd])
                
        combined_conv_submodel = prev_layers_fwd
        submodel = keras.models.Model(inputs = submodel_inp, outputs = combined_conv_submodel)
        combined_conv = self.apply_conjoined_rcps(curr_submodel = submodel, input_tensor = inp)
        
        #Counts prediction
        gap_combined_conv = kl.GlobalAvgPool1D()(combined_conv)
        count_out = kl.Reshape((-1,), name=countouttaskname)(
                    RevCompConv1D(filters=1, kernel_size=1, kernel_initializer = self.kernel_initializer)(
                      kl.Reshape((1,-1))(kl.concatenate([
                          #concatenation of the bias layer both before and after
                          # is needed for rc symmetry
                          kl.Lambda(lambda x: x[:, ::-1])(bias_counts_input),
                          gap_combined_conv,
                          bias_counts_input], axis=-1))))
        #Profile prediction
        profile_out_prebias = RevCompConv1D(
                    filters=1,kernel_size=self.outconv_kernel_size,
                    kernel_initializer = self.kernel_initializer, padding='valid')(combined_conv)
        profile_out = RevCompConv1D(
                    filters=1, kernel_size=1, name=profileouttaskname, kernel_initializer = self.kernel_initializer)(
                            kl.concatenate([
                                #concatenation of the bias layer both before and after
                                # is needed for rc symmetry
                                kl.Lambda(lambda x: x[:, :, ::-1])(bias_profile_input),
                                profile_out_prebias,
                                bias_profile_input], axis=-1))
        model = keras.models.Model(
                  inputs=[inp, bias_counts_input, bias_profile_input],
                  outputs=[count_out, profile_out])
        
        model.compile(keras.optimizers.Adam(lr=self.lr),
                          loss=['mse', MultichannelMultinomialNLL(2)],
                          loss_weights=[self.c_task_weight, self.p_task_weight]) 
        
        return model 
    
    
class StandardBPNetArch(AbstractProfileModel): 
    def __init__(self, is_add, **kwargs):
        super().__init__(**kwargs)
        self.is_add = is_add
        
    def get_keras_model(self):
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)
        
        inp, bias_counts_input, bias_profile_input = self.get_inputs()
        countouttaskname, profileouttaskname = self.get_names()
        
        first_conv = kl.Conv1D(self.filters,
                               kernel_size=self.conv1_kernel_size,
                               kernel_initializer = self.kernel_initializer,
                               padding='valid',
                               activation='relu')(inp)
        curr_layer_size = self.input_seq_len - (self.conv1_kernel_size-1)

        prev_layers = first_conv
        for i in range(1, self.n_dil_layers + 1):
            dilation_rate = 2**i
            conv_output = kl.Conv1D(self.filters, kernel_size=self.dil_kernel_size, 
                                    kernel_initializer = self.kernel_initializer,
                                    padding='valid',
                                    activation='relu', 
                                    dilation_rate=dilation_rate)(prev_layers)

            width_to_trim = dilation_rate * (self.dil_kernel_size - 1)


            curr_layer_size = (curr_layer_size - width_to_trim)
            prev_layers = self.trim_flanks_of_conv_layer(
              conv_layer = prev_layers, output_len = curr_layer_size, 
              width_to_trim = width_to_trim, filters = self.filters)

            if(self.is_add): 
                prev_layers = kl.add([prev_layers, conv_output])
            else:
                prev_layers = kl.average([prev_layers, conv_output])

        combined_conv = prev_layers

        #Counts Prediction
        gap_combined_conv = kl.GlobalAvgPool1D()(combined_conv)
        count_out = kl.Dense(2, kernel_initializer = self.kernel_initializer, name=countouttaskname)(
            kl.concatenate([gap_combined_conv, bias_counts_input], axis=-1))

        #Profile Prediction
        profile_out_prebias = kl.Conv1D(filters = 2, 
                                        kernel_size = self.outconv_kernel_size,
                                        kernel_initializer = self.kernel_initializer,
                                        padding = 'valid')(combined_conv)
        profile_out = kl.Conv1D(2, kernel_size=1, kernel_initializer = self.kernel_initializer, name=profileouttaskname)(
            kl.concatenate([profile_out_prebias, bias_profile_input], axis=-1))

        model = keras.models.Model(
          inputs=[inp, bias_counts_input, bias_profile_input],
          outputs=[count_out, profile_out])
        
        
        model.compile(keras.optimizers.Adam(lr=self.lr),
                      loss=['mse', MultichannelMultinomialNLL(2)],
                      loss_weights=[self.c_task_weight, self.p_task_weight]) 
       
        return model  

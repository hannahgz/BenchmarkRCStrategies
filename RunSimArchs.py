import numpy as np 
import tensorflow as tf 
import keras_genomics
import tensorflow as tf
import keras 
import keras_genomics
import numpy as np
import os
import gzip

from numpy.random import seed
from tensorflow import set_random_seed
from keras.callbacks import EarlyStopping, History, ModelCheckpoint
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from simdna.synthetic.core import read_simdata_file
from sklearn.model_selection import train_test_split


def map_embeddings(embeddings): 
    if len(embeddings) > 0:
        all_chars = ""
        for x in embeddings: 
            all_chars += str(x.what)
        if 'ELF1' in all_chars and 'GATA' in all_chars and 'RXRA' in all_chars: 
            return [1, 1, 1]
        if 'ELF1' in all_chars and 'GATA' in all_chars:
            return [1, 1, 0]
        if 'GATA' in all_chars and 'RXRA' in all_chars:
            return [0, 1, 1]
        if 'ELF1' in all_chars and 'RXRA' in all_chars:
            return [1, 0, 1]
        if 'ELF1' in all_chars: 
            return [1, 0, 0]
        if 'GATA' in all_chars:
            return [0, 1, 0]
        if 'RXRA' in all_chars:
            return [0, 0, 1]
    return [0, 0, 0]


def onehot_encode(sequence):
    ltrdict = {'a':[1,0,0,0],
               'c':[0,1,0,0],
               'g':[0,0,1,0],
               't':[0,0,0,1],
               'n':[0,0,0,0],
               'A':[1,0,0,0],
               'C':[0,1,0,0],
               'G':[0,0,1,0],
               'T':[0,0,0,1],
               'N':[0,0,0,0]}
    return np.array([ltrdict[x] for x in sequence])


def mutate(y, mutation_prob):
    np.random.seed(1234)
    mutated_y = []
    for row in y:
        new_labels_for_row = []
        for label in row:
            if np.random.uniform() < mutation_prob:
                new_labels_for_row.append(1-label)
            else:
                new_labels_for_row.append(label)
        mutated_y.append(new_labels_for_row)
    return np.array(mutated_y)


def prepare_sequences(seq_len):
    x = []
    with gzip.open("simdata_%s.sequences.gz" % str(seq_len), "r") as f: 
        for line in f: 
            curr = line[:seq_len].decode("utf-8")
            x.append(curr)
    embeddings1 = read_simdata_file("embeddings/DensityEmbedding_motifs-ELF1_known2_min-1_max-3_mean-2_zeroProb-0_seqLength-%s_numSeqs-10000.simdata" % str(seq_len)).embeddings
    embeddings2 = read_simdata_file("embeddings/DensityEmbedding_motifs-GATA_known6_min-1_max-3_mean-2_zeroProb-0_seqLength-%s_numSeqs-10000.simdata" % str(seq_len)).embeddings
    embeddings3 = read_simdata_file("embeddings/DensityEmbedding_motifs-RXRA_known1_min-1_max-3_mean-2_zeroProb-0_seqLength-%s_numSeqs-10000.simdata" % str(seq_len)).embeddings
    
    all_embeddings = [embeddings1, embeddings2, embeddings3]
    
    y = np.zeros((0, 3))
    for curr_embedding in all_embeddings: 
        for curr in curr_embedding: 
            y = np.append(y, [map_embeddings(curr)], axis = 0)
            
    x = np.array([onehot_encode(curr_seq) for curr_seq in x])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1234)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 3/7, random_state = 1234)
    
    y_train_mutate = mutate(y_train, 0.2)
    
    return x_train, x_val, x_test, y_train, y_train_mutate, y_val, y_test


class AuRocCallback(keras.callbacks.Callback):
    def __init__(self, model, valid_X, valid_Y):
        self.model = model
        self.valid_X = valid_X
        self.valid_Y = valid_Y
        self.best_auroc_sofar = 0.0
        self.best_weights = None
        self.best_epoch_number = 0
    def on_epoch_end(self, epoch, logs):
        preds = self.model.predict(self.valid_X)
        auroc = roc_auc_score(y_true=self.valid_Y, y_score=preds)
        if (auroc > self.best_auroc_sofar):
            self.best_weights = self.model.get_weights()
            self.best_epoch_number = epoch
            self.best_auroc_sofar = auroc
        
        
def train_model(model_wrapper, aug, curr_seed, batch_size, x, y, val_data):
    np.random.seed(curr_seed)
    tf.set_random_seed(curr_seed)
    
    model = model_wrapper.get_model()
    
    if aug == "rev_after_each": 
        x_train = np.asarray([val for val in x for __ in (0,1)])
        y_train = np.asarray([val for val in y for __ in (0,1)])
        for i in range(len(x_train)):
            if i % 2 == 1:
                x_train[i] = np.flip(x_train[i])
    elif aug == "rev_after_all":
        x_train = np.concatenate([x,x])
        y_train = np.concatenate([y,y])
        for i in range(len(x/2) + 1, len(x)):
            x_train[i] = np.flip(x_train[i])
    else: 
        x_train = x
        y_train = y
        
    auroc_callback = AuRocCallback(model = model,
                                   valid_X=val_data[0],
                                   valid_Y=val_data[1]) 
    
    early_stopping_callback = keras.callbacks.EarlyStopping(
                              monitor='val_loss',
                              patience = 10,
                              restore_best_weights=True)
    
    model.fit(x = x_train, y = y_train, validation_data = val_data,  
              callbacks =[early_stopping_callback, auroc_callback], 
              batch_size=batch_size, epochs=200)
    
    return model, early_stopping_callback, auroc_callback


def save_all(filepath, model_arch, curr_seed, callback, model, val_data):
    file_name = model_arch
    model.set_weights(callback.best_weights)   
    
    os.chdir("%s/%s" % (filepath, str(curr_seed)))
    results_file = open(file_name + ".txt", "w")
    y_pred = model.predict(val_data[0])
    auroc = roc_auc_score(val_data[1], y_pred) 
    auprc = average_precision_score(val_data[1], y_pred)
    print("auroc: " + str(auroc))
    print("auprc: "+ str(auprc))
    results_file.write("auroc: " + str(auroc) + "\n")
    results_file.write("auprc: " + str(auprc) + "\n")
    results_file.close()
        
    model.save(file_name + ".h5")
    os.chdir("/home/hannah/revcomp_simulated/")
    
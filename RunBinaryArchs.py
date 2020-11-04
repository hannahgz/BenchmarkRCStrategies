from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from keras.utils import CustomObjectScope
from keras.models import load_model
from scipy.special import softmax

from keras_genomics.layers.convolutional import RevCompConv1D
import tensorflow as tf

import statistics
from keras.utils import CustomObjectScope
from keras.models import load_model
from keras.callbacks import History

from scipy.stats import spearmanr
from vizsequence import viz_sequence
from matplotlib.ticker import FormatStrFormatter
from tensorflow.python.keras.utils.data_utils import Sequence

from binary_model_archs import get_rc_model
from binary_model_archs import get_reg_model
# from binary_model_archs import get_siamese_model

from seqdataloader.batchproducers.coordbased import coordbatchproducers
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from collections import namedtuple
import keras
import gzip

from seqdataloader.batchproducers.coordbased.coordbatchtransformers import get_revcomp
from seqdataloader.batchproducers.coordbased import coordbatchproducers
from seqdataloader.batchproducers.coordbased.coordbatchproducers import DownsampleNegativesCoordsBatchProducer
from seqdataloader.batchproducers.coordbased.coordbatchproducers import SimpleCoordsBatchProducer
from seqdataloader.batchproducers.coordbased import coordstovals
from seqdataloader.batchproducers import coordbased
from seqdataloader.batchproducers.coordbased import coordbatchproducers
from seqdataloader.batchproducers.coordbased import coordbatchtransformers
from seqdataloader.batchproducers.coordbased.coordstovals.bigwig import AbstractCountAndProfileTransformer 
from seqdataloader.batchproducers.coordbased.coordstovals.bigwig import LogCountsPlusOne
from seqdataloader.batchproducers.coordbased.coordstovals.bigwig import SmoothProfiles
from seqdataloader.batchproducers.coordbased.coordstovals.bigwig import BigWigReader 
from seqdataloader.batchproducers.coordbased.coordstovals.bigwig import smooth_profiles
from seqdataloader.batchproducers.coordbased.coordstovals.bigwig import rolling_window
from seqdataloader.batchproducers.coordbased.core import Coordinates, KerasBatchGenerator, apply_mask
from seqdataloader.batchproducers.coordbased.coordbatchtransformers import AbstractCoordBatchTransformer
from seqdataloader.batchproducers.coordbased.coordbatchtransformers import get_revcomp
import numpy as np

import os

Coordinates = namedtuple("Coordinates",
                         ["chrom", "start", "end", "isplusstrand"])
Coordinates.__new__.__defaults__ = (True,)


def apply_mask(tomask, mask):
    if isinstance(tomask, dict):
        return dict([(key, val[mask]) for key,val in tomask.items()])
    elif isinstance(tomask, list):
        return [x[mask] for x in mask]
    else:
        return x[mask]


class KerasBatchGenerator(keras.utils.Sequence):
  
    """
    Args:
        coordsbatch_producer (KerasSequenceApiCoordsBatchProducer)
        inputs_coordstovals (CoordsToVals)
        targets_coordstovals (CoordsToVals)
        sampleweights_coordstovals (CoordsToVals)
        coordsbatch_transformer (AbstracCoordBatchTransformer)
        qc_func (callable): function that can be used to filter
            out poor-quality sequences.
        sampleweights_coordstoval: either this argument or
            sampleweights_from_inputstargets could be used to
            specify sample weights. sampleweights_coordstoval
            takes a batch of coords as inputs.
        sampleweights_from_inputstargets: either this argument or
            sampleweights_coordstoval could be used to
            specify sample weights. sampleweights_from_inputstargets
            takes the inputs and targets values to generate the weights.
    """
    def __init__(self, coordsbatch_producer,
                       inputs_coordstovals,
                       targets_coordstovals,
                       coordsbatch_transformer=None,
                       qc_func=None,
                       sampleweights_coordstovals=None,
                       sampleweights_from_inputstargets=None):
        self.coordsbatch_producer = coordsbatch_producer
        self.inputs_coordstovals = inputs_coordstovals
        self.targets_coordstovals = targets_coordstovals
        self.coordsbatch_transformer = coordsbatch_transformer
        self.sampleweights_coordstovals = sampleweights_coordstovals
        self.sampleweights_from_inputstargets =\
            sampleweights_from_inputstargets
        if sampleweights_coordstovals is not None:
            assert sampleweights_from_inputstargets is None
        if sampleweights_from_inputstargets is not None:
            assert sampleweights_coordstovals is None
        self.qc_func = qc_func
 
    def __getitem__(self, index):
        coords_batch = self.coordsbatch_producer[index]
        if (self.coordsbatch_transformer is not None):
            coords_batch = self.coordsbatch_transformer(coords_batch)
        inputs = self.inputs_coordstovals(coords_batch)
        if (self.targets_coordstovals is not None):
            targets = self.targets_coordstovals(coords_batch)
        else:
            targets=None
        if (self.qc_func is not None):
            qc_mask = self.qc_func(inputs=inputs, targets=targets)
            inputs = apply_mask(tomask=inputs, mask=qc_mask)
            if (targets is not None):
                targets = apply_mask(tomask=targets, mask=qc_mask)
        else:
            qc_mask = None
        if (self.sampleweights_coordstovals is not None):
            sample_weights = self.sampleweights_coordstovals(coords_batch)
            return (inputs, targets, sample_weights)
        elif (self.sampleweights_from_inputstargets is not None):
            sample_weights = self.sampleweights_from_inputstargets(
                                inputs=inputs, targets=targets)
            return (inputs, targets, sample_weights)
        else:
            if (self.targets_coordstovals is not None):
                return (inputs, targets)
            else:
                return inputs
   
    def __len__(self):
        return len(self.coordsbatch_producer)
    
    def on_epoch_end(self):
        self.coordsbatch_producer.on_epoch_end()
        
        
def get_new_coors_around_center(coors, center_size_to_use):
    new_coors = []
    for coor in coors:
        coor_center = int(0.5*(coor.start + coor.end))
        left_flank = int(0.5*center_size_to_use)
        right_flank = center_size_to_use - left_flank
        new_start = coor_center-left_flank
        new_end = coor_center+right_flank
        new_coors.append(Coordinates(chrom=coor.chrom,
                                     start=new_start, end=new_end,
                                     isplusstrand=coor.isplusstrand))
    return new_coors


class CoordsToVals(object):
    
    def __call__(self, coors):
        """
        Args:
            coors (:obj:`list` of :obj:`Coordinates`):
        Returns:
            numpy ndarray OR list of ndarrays OR a dict of mode_name->ndarray.
              Returns a list of ndarrays if returning multiple modes.
              Alternatively, returns a dict where key is the mode name
              and the value is the ndarray for the mode.
        """
        raise NotImplementedError()


class CoordsToValsJoiner(CoordsToVals):

    def __init__(self, coordstovals_list):
        """
        Joins batches returned by other CoordsToVals objects
        Args:
            coorstovals_list (:obj:`list` of :obj:`CoordsToVals`): List of
                CoordsToVals whose values to combine
        """
        self.coordstovals_list = coordstovals_list
                
    def __call__(self, coors):
        batch_to_return = None        
        for idx,coordstovals_obj in enumerate(self.coordstovals_list):
            the_batch = coordstovals_obj(coors=coors)
            assert the_batch is not None
            if isinstance(the_batch, dict):
                assert ((batch_to_return is None) or
                        (isinstance(batch_to_return, dict))), (
                        "coordstovals object at idx"+str(idx)
                        +" returned a dict, but previous coordstovals"
                        +" objects had a return type incompatible with this")
                if (batch_to_return is None):
                    batch_to_return = {}
                for key in the_batch:
                    assert key not in batch_to_return, (
                      "coordstovals object at idx"+str(idx)
                      +" returned a dict with a key of "+key
                      +", which collides with a pre-existing key returned by"
                      +" another coordstovals object")
                batch_to_return.update(the_batch)
            else:
                assert ((batch_to_return is None) or
                        (isinstance(batch_to_return, list))), (
                        "coordstovals object at idx"+str(idx)
                        +" returned a type incompatible with dict, but previous"
                        +" coordstovals objects had a return type of dict")
                if (isinstance(the_batch, list)==False):
                    the_batch = [the_batch]
                if (batch_to_return is None):
                    batch_to_return = []
                batch_to_return.extend(the_batch)
        if (batch_to_return is None):
            batch_to_return = []
        return batch_to_return


class AbstractSingleNdarrayCoordsToVals(CoordsToVals):

    def __init__(self, mode_name=None):
        """
        Args:
            mode_name (:obj:`str`, optional): default None. If None, then
                the return of __call__ will be a numpy ndarray. Otherwise, it
                will be a dictionary with a key of mode_name and a value being
                the numpy ndarray.
        """
        self.mode_name = mode_name
    
    def _get_ndarray(self, coors):
        """
        Args:
            coors (:obj:`list` of :obj:`Coordinates):
            
        Returns:
            numpy ndarray
        """
        raise NotImplementedError()
    
    def __call__(self, coors):
        ndarray = self._get_ndarray(coors)
        if (self.mode_name is None):
            return ndarray
        else:
            return {self.mode_name: ndarray}
        
        
class SimpleLookup(AbstractSingleNdarrayCoordsToVals):

    def __init__(self, lookup_file,
                       transformation=None,
                       default_returnval=0.0, **kwargs):
        super(SimpleLookup, self).__init__(**kwargs)
        self.lookup_file = lookup_file
        self.transformation = transformation
        self.default_returnval = default_returnval
        self.lookup = {}
        self.num_labels = None
        for line in (gzip.open(self.lookup_file) if ".gz"
                     in self.lookup_file else open(self.lookup_file)):
            (chrom, start_str, end_str, *labels) =\
              line.decode("utf-8").rstrip().split("\t")
            coord = Coordinates(chrom=chrom,
                                start=int(start_str),
                                end=int(end_str))
            labels = [(self.transformation(float(x))
                       if self.transformation is not None else float(x))
                      for x in labels] 
            self.lookup[(coord.chrom, coord.start, coord.end)] = labels
            if (self.num_labels is None):
                self.num_labels = len(labels)
            else:
                assert len(labels)==self.num_labels,(
                  "Unequal label lengths; "+str(len(labels), self.num_labels))
    
    def _get_ndarray(self, coors):
        to_return = []
        for coor in coors:
            if (coor.chrom, coor.start, coor.end) not in self.lookup:
                to_return.append(np.ones(self.num_labels)
                                 *self.default_returnval)
            else:
                to_return.append(
                    self.lookup[(coor.chrom, coor.start, coor.end)])
        return np.array(to_return)

    
class RevcompTackedOnSimpleCoordsBatchProducer(DownsampleNegativesCoordsBatchProducer):
       def _get_coordslist(self):
            self.last_used_offset += 1
            self.last_used_offset = self.last_used_offset%self.subsample_factor
            print("Using an offset of ",self.last_used_offset," before striding")
            self.last_used_offset = self.last_used_offset%self.subsample_factor
            subsampled_neg_coords = self.neg_bedfileobj.get_strided_subsample(
                                    offset=self.last_used_offset,
                                    stride=self.subsample_factor) 
            pos_coords = self.pos_bedfileobj.coords_list
            self.subsampled_neg_coords = subsampled_neg_coords
            self.pos_coords = pos_coords
            curr_coords = pos_coords+subsampled_neg_coords
            return [x for x in curr_coords]+[get_revcomp(x) for x in curr_coords]

        
def get_aug_alt_generator(dataset, seq_len, curr_seed, curr_batch_size): 
    inputs_coordstovals = coordbased.coordstovals.fasta.PyfaidxCoordsToVals(
        genome_fasta_path="hg19.genome.fa",
        center_size_to_use= seq_len)
    
    targets_coordstovals = SimpleLookup(
        lookup_file = "%s/%s_lookup.bed.gz" % (dataset, dataset), 
        transformation = None, default_returnval = 0.0) 

    target_proportion_positives = 1/5
    
    aug_tacked_on_keras_train_batch_generator = KerasBatchGenerator(
    coordsbatch_producer=RevcompTackedOnSimpleCoordsBatchProducer(
        pos_bed_file = "%s/%s_foreground_train.bed.gz" % (dataset, dataset),
        neg_bed_file = "%s/%s_background_train.bed.gz" % (dataset, dataset), 
        target_proportion_positives = target_proportion_positives, 
        batch_size= curr_batch_size,
        shuffle_before_epoch=True, 
        seed=curr_seed),
    inputs_coordstovals=inputs_coordstovals,
    targets_coordstovals=targets_coordstovals)
    
    return aug_tacked_on_keras_train_batch_generator


def get_generators(dataset, seq_len, is_aug, curr_seed): 
    inputs_coordstovals = coordbased.coordstovals.fasta.PyfaidxCoordsToVals(
        genome_fasta_path="hg19.genome.fa",
        center_size_to_use= seq_len)
    
    targets_coordstovals = SimpleLookup(
        lookup_file = "%s/%s_lookup.bed.gz" % (dataset, dataset), 
        transformation = None, default_returnval = 0.0) 

    target_proportion_positives = 1/5
    
    if not is_aug: 
        standard_train_batch_generator = KerasBatchGenerator(
          coordsbatch_producer=coordbatchproducers.DownsampleNegativesCoordsBatchProducer(
              pos_bed_file = "%s/%s_foreground_train.bed.gz" % (dataset, dataset),
              neg_bed_file = "%s/%s_background_train.bed.gz" % (dataset, dataset), 
              target_proportion_positives = target_proportion_positives, 
              batch_size=100,
              shuffle_before_epoch=True, 
              seed=curr_seed),
          inputs_coordstovals=inputs_coordstovals,
          targets_coordstovals=targets_coordstovals)
        return standard_train_batch_generator
    else: 
        aug_train_batch_generator = KerasBatchGenerator(
          coordsbatch_producer=coordbatchproducers.DownsampleNegativesCoordsBatchProducer(
              pos_bed_file = "%s/%s_foreground_train.bed.gz" % (dataset, dataset),
              neg_bed_file = "%s/%s_background_train.bed.gz" % (dataset, dataset), 
              target_proportion_positives = target_proportion_positives, 
              batch_size=100,
              shuffle_before_epoch=True, 
              seed=curr_seed),
          inputs_coordstovals=inputs_coordstovals,
          targets_coordstovals=targets_coordstovals,
          coordsbatch_transformer=coordbatchtransformers.ReverseComplementAugmenter())
        return aug_train_batch_generator

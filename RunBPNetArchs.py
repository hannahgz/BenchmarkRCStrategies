from seqdataloader.batchproducers.coordbased.coordbatchtransformers import get_revcomp
from seqdataloader.batchproducers.coordbased import coordbatchproducers
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

def get_inputs_and_targets(dataset, seq_len, out_pred_len): 
    inputs_coordstovals = coordstovals.core.CoordsToValsJoiner(
        coordstovals_list=[
          coordbased.coordstovals.fasta.PyfaidxCoordsToVals(
            genome_fasta_path="bpnet_data/mm10_no_alt_analysis_set_ENCODE.fasta",
            mode_name="sequence",
            center_size_to_use=seq_len),
          coordstovals.bigwig.PosAndNegSeparateLogCounts(
            counts_mode_name="patchcap.logcount",
            profile_mode_name="patchcap.profile",
            pos_strand_bigwig_path="bpnet_data/patchcap/counts.pos.bw",
            neg_strand_bigwig_path="bpnet_data/patchcap/counts.neg.bw",
            center_size_to_use=out_pred_len),
        ]
    )

    targets_coordstovals = coordstovals.core.CoordsToValsJoiner(
        coordstovals_list=[
          coordstovals.bigwig.PosAndNegSeparateLogCounts(
            counts_mode_name="CHIPNexus.%s.logcount" % dataset,
            profile_mode_name="CHIPNexus.%s.profile" % dataset,
            pos_strand_bigwig_path="bpnet_data/%s/counts.pos.bw" % dataset,
            neg_strand_bigwig_path="bpnet_data/%s/counts.neg.bw" % dataset,
            center_size_to_use=out_pred_len)
        ]
    )
    return inputs_coordstovals, targets_coordstovals


class RevcompTackedOnSimpleCoordsBatchProducer(SimpleCoordsBatchProducer):
       def _get_coordslist(self):
        return [x for x in self.bed_file.coords_list]+[get_revcomp(x) for x in self.bed_file.coords_list ]
    
def get_specific_generator(PARAMETERS, inputs_coordstovals, targets_coordstovals, model_arch, curr_seed): 
    train_file = "bpnet_%s_train_1k_around_summits.bed.gz" % PARAMETERS['dataset']
    valid_file = "bpnet_%s_valid_1k_around_summits.bed.gz" % PARAMETERS['dataset']
    chromsizes_file="bpnet_data/mm10.chrom.sizes"
    
    if model_arch == "reg" or model_arch == "RevComp" or model_arch == "siamese" or model_arch == "RevComp_half": 
        train_batch_generator = KerasBatchGenerator(
            coordsbatch_producer=coordbatchproducers.SimpleCoordsBatchProducer(
                bed_file = train_file,
                batch_size=64,
                shuffle_before_epoch=True, 
                seed=curr_seed),
            inputs_coordstovals=inputs_coordstovals,
            targets_coordstovals=targets_coordstovals,
            coordsbatch_transformer=coordbatchtransformers.UniformJitter(
                maxshift=200, chromsizes_file=chromsizes_file))
    elif model_arch == "aug": 
        train_batch_generator = KerasBatchGenerator(
            coordsbatch_producer=coordbatchproducers.SimpleCoordsBatchProducer(
                bed_file=train_file,
                batch_size=64,
                shuffle_before_epoch=True, 
                seed=curr_seed),
            inputs_coordstovals=inputs_coordstovals,
            targets_coordstovals=targets_coordstovals,
            coordsbatch_transformer=coordbatchtransformers.ReverseComplementAugmenter().chain(
                coordbatchtransformers.UniformJitter(
                    maxshift=200, chromsizes_file=chromsizes_file)))
    elif model_arch == "aug_alt":
        train_batch_generator = KerasBatchGenerator(
            coordsbatch_producer=RevcompTackedOnSimpleCoordsBatchProducer(
                bed_file=train_file,
                batch_size=64,
                shuffle_before_epoch=True, 
                seed=curr_seed),
            inputs_coordstovals=inputs_coordstovals,
            targets_coordstovals=targets_coordstovals,
            coordsbatch_transformer=coordbatchtransformers.UniformJitter(
                maxshift=200, chromsizes_file=chromsizes_file))
    elif model_arch == "aug_alt_double": 
        train_batch_generator = KerasBatchGenerator(
          coordsbatch_producer=RevcompTackedOnSimpleCoordsBatchProducer(
              bed_file=train_file,
              batch_size=132,
              shuffle_before_epoch=True, 
              seed=curr_seed),
          inputs_coordstovals=inputs_coordstovals,
          targets_coordstovals=targets_coordstovals,
          coordsbatch_transformer=coordbatchtransformers.UniformJitter(
                      maxshift=200, chromsizes_file=chromsizes_file)) 
    val_batch_generator = KerasBatchGenerator(
      coordsbatch_producer=coordbatchproducers.SimpleCoordsBatchProducer(
                bed_file = valid_file,
                batch_size=64,
                shuffle_before_epoch=False, 
                seed=curr_seed),
      inputs_coordstovals=inputs_coordstovals,
      targets_coordstovals=targets_coordstovals
    )
    return train_batch_generator, val_batch_generator
        
    
def get_generators(PARAMETERS, inputs_coordstovals, targets_coordstovals, curr_seed): 
   
    train_file = "bpnet_%s_train_1k_around_summits.bed.gz" % PARAMETERS['dataset']
    valid_file = "bpnet_%s_valid_1k_around_summits.bed.gz" % PARAMETERS['dataset']
    test_file = "bpnet_%s_test_1k_around_summits.bed.gz" % PARAMETERS['dataset']
    chromsizes_file="bpnet_data/mm10.chrom.sizes"
   
    standard_train_batch_generator = KerasBatchGenerator(
      coordsbatch_producer=coordbatchproducers.SimpleCoordsBatchProducer(
          bed_file = train_file,
          batch_size=64,
          shuffle_before_epoch=True, 
          seed=curr_seed),
      inputs_coordstovals=inputs_coordstovals,
      targets_coordstovals=targets_coordstovals,
       coordsbatch_transformer=coordbatchtransformers.UniformJitter(
                  maxshift=200, chromsizes_file=chromsizes_file))    

    aug_train_batch_generator = KerasBatchGenerator(
      coordsbatch_producer=coordbatchproducers.SimpleCoordsBatchProducer(
          bed_file=train_file,
          batch_size=64,
          shuffle_before_epoch=True, 
          seed=curr_seed),
      inputs_coordstovals=inputs_coordstovals,
      targets_coordstovals=targets_coordstovals,
      coordsbatch_transformer=coordbatchtransformers.ReverseComplementAugmenter().chain(
          coordbatchtransformers.UniformJitter(
              maxshift=200, chromsizes_file=chromsizes_file)))

    aug_tacked_on_train_batch_generator = KerasBatchGenerator(
      coordsbatch_producer=RevcompTackedOnSimpleCoordsBatchProducer(
          bed_file=train_file,
          batch_size=64,
          shuffle_before_epoch=True, 
          seed=curr_seed),
      inputs_coordstovals=inputs_coordstovals,
      targets_coordstovals=targets_coordstovals,
      coordsbatch_transformer=coordbatchtransformers.UniformJitter(
                  maxshift=200, chromsizes_file=chromsizes_file))
    
    aug_tacked_on_double_train_batch_generator = KerasBatchGenerator(
      coordsbatch_producer=RevcompTackedOnSimpleCoordsBatchProducer(
          bed_file=train_file,
          batch_size=132,
          shuffle_before_epoch=True, 
          seed=curr_seed),
      inputs_coordstovals=inputs_coordstovals,
      targets_coordstovals=targets_coordstovals,
      coordsbatch_transformer=coordbatchtransformers.UniformJitter(
                  maxshift=200, chromsizes_file=chromsizes_file))

    val_batch_generator = KerasBatchGenerator(
      coordsbatch_producer=coordbatchproducers.SimpleCoordsBatchProducer(
                bed_file = valid_file,
                batch_size=64,
                shuffle_before_epoch=False, 
                seed=curr_seed),
      inputs_coordstovals=inputs_coordstovals,
      targets_coordstovals=targets_coordstovals
    )
    
    train_batch_generators = {
        'standard': standard_train_batch_generator,
        'aug': aug_train_batch_generator,
        'aug_alt': aug_tacked_on_train_batch_generator,
        'aug_alt_double': aug_tacked_on_double_train_batch_generator, 
    }
    
    return train_batch_generators, val_batch_generator


class GeneralReverseComplement(AbstractCoordBatchTransformer):
    def __call__(self, coords):
        return [get_revcomp(x) for x in coords]
    
def get_test_generator(dataset, inputs_coordstovals, targets_coordstovals, size, curr_seed): 
    test_file = "bpnet_%s_test_1k_around_summits.bed.gz" % dataset
    chromsizes_file="bpnet_data/mm10.chrom.sizes"
        
    keras_test_batch_generator = KerasBatchGenerator(
      coordsbatch_producer=coordbatchproducers.SimpleCoordsBatchProducer(
                bed_file = test_file,
                batch_size=64,
                shuffle_before_epoch=False, 
                seed=curr_seed),
      inputs_coordstovals=inputs_coordstovals,
      targets_coordstovals=targets_coordstovals)
    
    keras_rc_test_batch_generator  = KerasBatchGenerator(
      coordsbatch_producer=coordbatchproducers.SimpleCoordsBatchProducer(
          bed_file=test_file,
          batch_size=64,
          shuffle_before_epoch=False, 
          seed=curr_seed),
      inputs_coordstovals=inputs_coordstovals,
      targets_coordstovals=targets_coordstovals,
      coordsbatch_transformer=GeneralReverseComplement())
    
    return keras_test_batch_generator, keras_rc_test_batch_generator


def save_all(PARAMETERS, model, model_arch, model_history): 

    if PARAMETERS["c_task_weight"] !=0 and PARAMETERS["p_task_weight"] != 0: 
        task_weight = "both_non_zero_" 
    elif PARAMETERS["c_task_weight"] != 0: 
        task_weight = "c_non_zero_"
    else: 
        task_weight = ""
        

    os.chdir("bpnet_data/%s/new_results/%s" % (PARAMETERS['dataset'], str(PARAMETERS['seed'])))        
        
    txt_file_name = ("%s_TrainProfileModel%s_%s%s_loss_add_profile_only.txt" % 
                     (str(PARAMETERS['seed']), PARAMETERS['dataset'], 
                      task_weight, model_arch))
        
    loss_file = open(txt_file_name, "w")
    loss_file.write("model parameters" + "\n")
    for x in PARAMETERS: 
        loss_file.write(str(x) + ": " + str(PARAMETERS[x]) + "\n")
    loss_file.write("val_loss\n")
    for row in model_history.history["val_loss"]: 
        loss_file.write(str(row) + "\n")
    loss_file.write("min val loss: " + str(np.min(model_history.history["val_loss"])))  

    loss_file.close()
    
    model_save_name = ("%s_TrainProfileModel%s_%s%s_add_profile_only.h5" % 
                     (str(PARAMETERS['seed']), PARAMETERS['dataset'], 
                      task_weight, model_arch))        
        
    model.save(model_save_name)
    os.chdir("/home/hannah")
    

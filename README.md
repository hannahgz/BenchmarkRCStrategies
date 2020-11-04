# Benchmarking Reverse-Complement Strategies for Deep Learning Models in Genomics 

## Code snippets for converting trained models to post-training conjoined models
### Simulated Models 

```python
model = load_model('model_name')
binary_model_getlogits = keras.models.Model(inputs=model.inputs,
                                            outputs=model.layers[-2].output)

fwd_sequence_input = keras.layers.Input(shape=('seq_len', 4))
rev_sequence_input = keras.layers.Lambda(function=lambda x: x[:,::-1,::-1])(fwd_sequence_input)
fwd_logit_output = binary_model_getlogits(fwd_sequence_input)
rev_logit_output = binary_model_getlogits(rev_sequence_input)
average_logits = keras.layers.Average()([fwd_logit_output, rev_logit_output])
sigmoid_out = keras.layers.Activation("sigmoid")(average_logits)

siamese_model = keras.models.Model(inputs=[fwd_sequence_input],
                                           outputs=[sigmoid_out])
```

### Binary Prediction Models

```python
model = load_model('model_name')
binary_model_get_out = keras.models.Model(inputs=model.inputs,
                                        outputs=model.layers[-1].output)

fwd_sequence_input = keras.layers.Input(shape=('seq_len', 4))
rev_sequence_input = keras.layers.Lambda(function=lambda x: x[:,::-1,::-1])(fwd_sequence_input)
fwd_output = binary_model_get_out(fwd_sequence_input)
rev_output = binary_model_get_out(rev_sequence_input)
average_out = keras.layers.Average()([fwd_output, rev_output])

siamese_model = keras.models.Model(inputs=[fwd_sequence_input],
                                   outputs=[average_out])
```


### Profile Prediction Models

```python
loaded_model = load_model('model_name)

#Let's create the model
#Define the inputs
fwd_sequence_input = keras.models.Input(shape=(1346,4))
fwd_patchcap_logcount = keras.models.Input(shape=(2,))
fwd_patchcap_profile = keras.models.Input(shape=(1000,2))

#RevComp input
rev_sequence_input = keras.layers.Lambda(lambda x: x[:,::-1,::-1])(fwd_sequence_input)
rev_patchcap_logcount = keras.layers.Lambda(lambda x: x[:,::-1])(fwd_patchcap_logcount)
#note that last axis is NOT fwd vs reverse strand, but different smoothing levels
#that's why we only flip the middle axis
rev_patchcap_profile = keras.layers.Lambda(lambda x: x[:,::-1])(fwd_patchcap_profile)

#Run the model on the original fwd inputs
fwd_logcount, fwd_profile = loaded_model(
    [fwd_sequence_input, fwd_patchcap_logcount, fwd_patchcap_profile])

#Run the original model on the reverse inputs
rev_logcount, rev_profile = loaded_model(
    [rev_sequence_input, rev_patchcap_logcount, rev_patchcap_profile])

#Reverse complement rev_logcount and rev_profile to be compatible with fwd
revcompd_rev_logcount = keras.layers.Lambda(lambda x: x[:,::-1])(rev_logcount)
revcompd_rev_profile = keras.layers.Lambda(lambda x: x[:,::-1,::-1])(rev_profile)

#Average the two
avg_logcount = keras.layers.Average()([fwd_logcount, revcompd_rev_logcount])
avg_profile = keras.layers.Average()([fwd_profile, revcompd_rev_profile])

#Create a model that goes from the inputs to the averaged output
siamese_model = keras.models.Model(inputs=[fwd_sequence_input,
                                           fwd_patchcap_logcount,
                                           fwd_patchcap_profile],
                                   outputs=[avg_logcount, avg_profile])
```

## Important Notes 
* Typical BPNet architectures are multitasked, but here we benchmakred on only the single-tasked version. 
* Keras 2.2.4 was used to train all models. Keras 2.3.0 and other newer versions do not sum the loss values for all batches. More information can be found here: 

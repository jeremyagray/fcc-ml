#!/usr/bin/env python

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

ti = pd.read_csv("https://storage.googleapis.com/"
                 "tf-datasets/titanic/train.csv")

tif = ti.copy()
til = tif.pop('survived')

inputs = {}

for name, column in tif.items():
    dtype = column.dtype
    if dtype == object:
        dtype = tf.string
    else:
        dtype = tf.float32

    inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

numeric_inputs = {name: input for name, input in inputs.items()
                  if input.dtype == tf.float32}

x = layers.Concatenate()(list(numeric_inputs.values()))
norm = preprocessing.Normalization()
norm.adapt(np.array(ti[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)

ppi = [all_numeric_inputs]

for name, input in inputs.items():
    if input.dtype == tf.float32:
        continue

    lookup = preprocessing.StringLookup(
        vocabulary=np.unique(tif[name]))
    one_hot = preprocessing.CategoryEncoding(max_tokens=lookup.vocab_size())

    x = lookup(input)
    x = one_hot(x)
    ppi.append(x)

ppic = layers.Concatenate()(ppi)
tip = tf.keras.Model(inputs, ppic)
# tf.keras.utils.plot_model(model=tip, rankdir='LR', dpi=72, show_shapes=True)

tifd = {name: np.array(value)
        for name, value in tif.items()}

# fd = {name: values[:1] for name, values in tifd.items()}
# print(tip(fd))


def ti_model(preprocessing_head, inputs):
    body = tf.keras.Sequential([
        layers.Dense(64),
        layers.Dense(1)
    ])

    preprocessed_inputs = preprocessing_head(inputs)
    result = body(preprocessed_inputs)
    model = tf.keras.Model(inputs, result)

    model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.optimizers.Adam(),
                  metrics=['accuracy'])

    return model


tim = ti_model(tip, inputs)

# tim.fit(x=tifd, y=til, epochs=10)

tids = tf.data.Dataset.from_tensor_slices((tifd, til))
tib = tids.shuffle(len(til)).batch(32)
tim.fit(tib, epochs=50)

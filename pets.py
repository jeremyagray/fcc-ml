#!/usr/bin/env python

from tensorflow.keras.layers.experimental import preprocessing
import numpy as np
import os
import pandas as pd
import shutil
import tensorflow as tf
import urllib
import zipfile


def load_data():
    file = 'datasets/petfinder-mini/petfinder-mini.csv'

    # Temporary directory paths.
    tmp = '/home/gray/tmp'
    directory = 'fcc-ml'
    tmpdir = os.path.join(tmp, directory)
    zipdir = os.path.join(tmpdir, 'petfinder-mini')
    zipname = 'petfinder-mini.zip'
    zfile = os.path.join(tmpdir, zipname)
    petname = 'petfinder-mini.csv'
    petfile = os.path.join(zipdir, petname)

    # Test for the data files before downloading them.
    if not os.path.isfile(petfile):
        # Create the temporary directory if it does not exist.
        if os.path.exists(tmp):
            if not os.path.exists(tmpdir):
                os.makedirs(tmpdir)

        # Download and prepare data set.
        url = 'http://storage.googleapis.com/'\
            'download.tensorflow.org/data/petfinder-mini.zip'

        # Make a good fake request for the CDN.
        req = urllib.request.Request(url,
                                     headers={'User-Agent': "Magic Browser"})

        # Download the file and save it.
        with urllib.request.urlopen(req) \
             as response, open(zfile, 'wb') as file:
            shutil.copyfileobj(response, file)

        with zipfile.ZipFile(zfile, 'r') as archive:
            archive.extractall(tmpdir)

    # Load CSV data into pandas.
    pets = pd.read_csv(petfile)

    return pets


# Pandas dataframe to tensorflow.data dataset.
# Requires whole dataframe, with features to use and target.
# https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers
def df_to_ds(df, target, shuffle=True, batch=32):
    features = df.copy()
    labels = features.pop(target)

    ds = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(features))

    ds = ds.batch(batch)
    ds = ds.prefetch(batch)

    return ds


def get_normalization_layer(dataset, feature):
    # Create a Normalization layer for our feature.
    normalizer = preprocessing.Normalization()

    # Prepare a Dataset that only yields our feature.
    feature_ds = dataset.map(lambda x, y: x[feature])

    # Learn the statistics of the data.
    normalizer.adapt(feature_ds)

    return normalizer


def get_category_encoding_layer(dataset, feature, dtype, max_tokens=None):
    # Create a StringLookup layer which will turn strings into integer
    # indices.
    if dtype == 'string':
        index = preprocessing.StringLookup(max_tokens=max_tokens)
    else:
        index = preprocessing.IntegerLookup(max_values=max_tokens)

    # Prepare a dataset that only yields our feature.
    feature_ds = dataset.map(lambda x, y: x[feature])

    # Learn the set of possible values and assign them a fixed integer index.
    index.adapt(feature_ds)

    # Create a discretization for our integer indices.
    encoder = preprocessing.CategoryEncoding(max_tokens=index.vocab_size())

    # Prepare a dataset that only yields our feature.
    feature_ds = feature_ds.map(index)

    # Learn the space of possible indices.
    encoder.adapt(feature_ds)

    # Apply one-hot encoding to our indices. The lambda function
    # captures the layer so we can use them, or include them in the
    # functional model later.
    return lambda x: encoder(index(x))


if __name__ == '__main__':
    pets_d = load_data()
    pets_d['target'] = np.where(pets_d['AdoptionSpeed'] == 4, 0, 1)
    pets_d = pets_d.drop(columns=['AdoptionSpeed', 'Description'])

    # Split train/val/test 60/20/20 dataframes.
    train_d, val_d, test_d = \
        np.split(pets_d.sample(frac=1, random_state=42),
                 [int(0.60 * len(pets_d)), int(0.80 * len(pets_d))])

    # Features.
    train_f = train_d.copy()
    val_f = val_d.copy()
    test_f = test_d.copy()

    # Labels (targets).
    train_l = train_f.pop('target')
    val_l = val_f.pop('target')
    test_l = test_f.pop('target')

    # Datasets.
    train_ds = df_to_ds(train_d, 'target', shuffle=True, batch=256)
    val_ds = df_to_ds(val_d, 'target', shuffle=False, batch=256)
    test_ds = df_to_ds(test_d, 'target', shuffle=False, batch=256)

    # Build model.
    inputs = []
    encodeds = []

    # Numeric features.
    for feature in ['PhotoAmt', 'Fee']:
        input = tf.keras.Input(shape=(1,), name=feature)
        normalization = get_normalization_layer(train_ds, feature)
        encoded = normalization(input)
        inputs.append(input)
        encodeds.append(encoded)

    # Categorical features encoded as integers.
    for feature in ['Age']:
        input = tf.keras.Input(shape=(1,), name=feature, dtype='int64')
        encoding = get_category_encoding_layer(train_ds,
                                               feature,
                                               dtype='int64',
                                               max_tokens=5)
        encoded = encoding(input)
        inputs.append(input)
        encodeds.append(encoded)

    # Categorical features encoded as string.
    categoricals = ['Type', 'Color1', 'Color2', 'Gender',
                    'MaturitySize', 'FurLength', 'Vaccinated',
                    'Sterilized', 'Health', 'Breed1']

    for feature in categoricals:
        input = tf.keras.Input(shape=(1,), name=feature, dtype='string')
        encoding = get_category_encoding_layer(train_ds,
                                               feature,
                                               dtype='string',
                                               max_tokens=5)
        encoded = encoding(input)
        inputs.append(input)
        encodeds.append(encoded)

    encoded_layers = tf.keras.layers.concatenate(encodeds)
    x = tf.keras.layers.Dense(64, activation='relu')(encoded_layers)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, output)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['mse', 'accuracy'])

    tf.keras.utils.plot_model(model=model,
                              rankdir="LR",
                              dpi=72,
                              show_shapes=True)

    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=100)

    print(model.evaluate(test_ds))

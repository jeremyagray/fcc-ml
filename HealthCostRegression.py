#!/usr/bin/env python

from tensorflow.keras.layers.experimental import preprocessing
# import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil
import tensorflow as tf
import urllib


# Download and load data.
def load_data():
    # Temporary directory paths.
    tmp = '/home/gray/tmp'
    directory = 'fcc-ml'
    tmpdir = os.path.join(tmp, directory)

    # Test for the data files before downloading them.
    file = os.path.join(tmpdir, 'insurance.csv')

    if not (os.path.isfile(file)):
        # Create the temporary directory if it does not exist.
        if os.path.exists(tmp):
            if not os.path.exists(tmpdir):
                os.makedirs(tmpdir)

        # Data set URL.
        url = 'https://cdn.freecodecamp.org/project-data/'\
            'health-costs/insurance.csv'

        # Make a good fake request for the CDN.
        req = urllib.request.Request(url,
                                     headers={'User-Agent': "Magic Browser"})

        # Download the file and save it.
        data = os.path.join(tmpdir, 'insurance.csv')
        with urllib.request.urlopen(req) as response, open(data, 'wb') as file:
            shutil.copyfileobj(response, file)

    costs = pd.read_csv(file, sep=",")

    return costs


# Test model by checking how well the model generalizes using the test set.
def test_predictions(model, dataset):
    loss, mae, mse = model.evaluate(dataset, verbose=2)

    print("Testing set Mean Abs Error: {:9,.2f} expenses".format(mae))

    if mae < 3500:
        print("You passed the challenge.  Great job!")
    else:
        print("The mean absolute error must be less than $3,500.  "
              "Keep trying.")


# Plot predictions.
def plot_predictions(model, data, labels):
    test_predictions = model.predict(data).flatten()

    plt.axes(aspect='equal')
    # a = plt.axes(aspect='equal')
    plt.scatter(labels, test_predictions)
    plt.xlabel('True values (expenses)')
    plt.ylabel('Predictions (expenses)')
    lims = [0, 50000]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)


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


# Create and save training and validation loss versus epoch plot.
def plot_losses(epochs, train, val, file='loss.png'):
    fig = plt.figure(figsize=(8, 8))

    ax = plt.subplot(1, 2, 1)
    ax.plot(range(epochs), train, label='Training Loss')
    ax.plot(range(epochs), val, label='Validation Loss')
    ax.legend(loc='upper right')
    ax.set_title('Training and Validation Loss')
    fig.savefig(file)

    return


if __name__ == '__main__':
    costs = load_data()

    # Process costs.
    # Columns:
    # 'age':  integer (categorical?)
    # 'sex':  categorical
    # 'bmi':  float
    # 'children':  integer
    # 'smoker':  categorical
    # 'region':  categorical
    # 'expenses':  float; training target

    # Split 70%/20%/10% training/validation/testing.
    # pandas.DataFrame.sample():  Set random_state to reproduce partitioning.
    # https://stackoverflow.com/a/38251213/12968623

    # Data frames.
    train_d, val_d, test_d = \
        np.split(costs.sample(frac=1, random_state=42),
                 [int(0.70 * len(costs)), int(0.90 * len(costs))])

    # Features.
    train_f = train_d.copy()
    val_f = val_d.copy()
    test_f = test_d.copy()

    # Labels (targets).
    train_l = train_f.pop('expenses')
    val_l = val_f.pop('expenses')
    test_l = test_f.pop('expenses')

    train_ds = df_to_ds(train_d, 'expenses', shuffle=True, batch=256)
    val_ds = df_to_ds(val_d, 'expenses', shuffle=False, batch=256)
    test_ds = df_to_ds(test_d, 'expenses', shuffle=False, batch=256)

    # Build model.

    # Numeric features:  age, bmi, children.
    numerics = ['age', 'bmi', 'children']
    # numerics = ['bmi', 'children']
    # numerics = ['bmi']
    # Categorical features:  sex, smoker, region.
    categoricals = ['sex', 'smoker', 'region']
    # categoricals = ['sex', 'smoker']
    # categoricals = ['smoker']
    # Categorical features as integers:  age.  Maybe?
    # categorical_integers = ['age']
    categorical_integers = []

    inputs = []
    encodeds = []

    # Numeric features.
    for feature in numerics:
        input = tf.keras.Input(shape=(1,), name=feature)
        normalization = get_normalization_layer(train_ds, feature)
        encoded = normalization(input)
        inputs.append(input)
        encodeds.append(encoded)

    # Categorical features encoded as integers.
    for feature in categorical_integers:
        input = tf.keras.Input(shape=(1,), name=feature, dtype='int64')
        encoding = get_category_encoding_layer(train_ds,
                                               feature,
                                               dtype='int64',
                                               max_tokens=5)
        encoded = encoding(input)
        inputs.append(input)
        encodeds.append(encoded)

    # Categorical features encoded as strings.
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
    x = tf.keras.layers.Dense(4096, activation='relu')(encoded_layers)
    x = tf.keras.layers.Dropout(0.6)(x)
    x = tf.keras.layers.Dense(2048, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    # x = tf.keras.layers.Dropout(0.4)(x)
    output = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, output)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanAbsoluteError(),
                  metrics=['mae', 'mse'])

    tf.keras.utils.plot_model(model=model,
                              rankdir="LR",
                              dpi=72,
                              show_shapes=True)

    epochs = 30
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=epochs)

    # Plot the losses.
    plot_losses(epochs, history.history['loss'], history.history['val_loss'])

    test_predictions(model, test_ds)

    # print(model.evaluate(test_ds, return_dict=True))
    # print(model.metrics_names)
    # loss, mae, mse = model.evaluate(test_ds)
    # print('loss:  {}\nmae:  {}\nmse:  {}'.format(loss, mae, mse))

    # print(model.predict(test_ds))
    # actual = ''
    # for x, y in test_ds:
    #     actual = y.numpy().tolist()

    # predicted = model.predict(test_ds)
    # actual = [y for x, y in test_ds]
    # print(actual)

    # error = 0
    # num = 0

    # for i, item in enumerate(actual):
    #     print('actual:  ${:9,.2f} '\
    #           'predicted:  ${:9,.2f}'.format(item, predicted[i][0]))
    #     error += math.fabs(item - predicted[i][0])
    #     num = i

    # print('error:  {}'.format(error / num))

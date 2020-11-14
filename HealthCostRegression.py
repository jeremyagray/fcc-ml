#!/usr/bin/env python

# from tensorflow import keras
# import tensorflow_docs as tfdocs
# import tensorflow_docs.modeling
# import tensorflow_docs.plots
from tensorflow.keras.layers.experimental import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil
import tensorflow as tf
import urllib


# Download and load data.
def load_data():
    # Old way.  Fails.
    # url = 'https://cdn.freecodecamp.org/'
    # 'project-data/health-costs/insurance.csv'
    # file = 'insurance.csv'
    # dataset = keras.utils.get_file(file, url)

    # Improved way.
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
        url = 'https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv'

        # Make a good fake request for the CDN.
        req = urllib.request.Request(url,
                                     headers={'User-Agent': "Magic Browser"})

        # Download the file and save it.
        data = os.path.join(tmpdir, 'insurance.csv')
        with urllib.request.urlopen(req) as response, open(data, 'wb') as file:
            shutil.copyfileobj(response, file)

    costs = pd.read_csv(file, sep=",")

    return costs


# Create the regression model.
def create_model():
    pass


# Test model by checking how well the model generalizes using the test set.
def test_predictions(model, data, labels):
    loss, mae, mse = model.evaluate(data, labels, verbose=2)

    print("Testing set Mean Abs Error: {:5.2f} expenses".format(mae))

    if mae < 3500:
        print("You passed the challenge.  Great job!")
    else:
        print("The mean absolute error must be less than $3,500.  Keep trying.")


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


# Normalize the numeric inputs.
def get_normalization_layer(frame, features):
    inputs = {}

    for feature in features:
        inputs[feature] = tf.keras.Input(shape=(1,),
                                        name=feature,
                                        dtype=tf.float32)

    # Concatenate the numeric inputs.
    x = tf.keras.layers.Concatenate()(list(inputs.values()))

    # Initialize and adapt the normalization layer.
    normalizer = preprocessing.Normalization()
    normalizer.adapt(np.array(frame[inputs.keys()]))

    return normalizer(x)


# Encode the categorical inputs.
def get_categorical_layer(frame, features):
    inputs = []

    for feature in features:
        input = tf.keras.Input(shape=(1,),
                               name=feature,
                               dtype=tf.string)

        lookup = preprocessing.StringLookup(
            vocabulary=np.unique(frame[feature]))
        one_hot = preprocessing.CategoryEncoding(
            max_tokens=lookup.vocab_size())

        x = lookup(input)
        x = one_hot(x)
        inputs.append(x)

    # return tf.keras.layers.Concatenate()(inputs)
    return inputs


def get_catint_layer():
    pass


def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
    # Create a StringLookup layer which will turn strings into integer indices
    if dtype == 'string':
        index = preprocessing.StringLookup(max_tokens=max_tokens)
    else:
        index = preprocessing.IntegerLookup(max_values=max_tokens)

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])

    # Learn the set of possible values and assign them a fixed integer index.
    index.adapt(feature_ds)

    # Create a Discretization for our integer indices.
    encoder = preprocessing.CategoryEncoding(max_tokens=index.vocab_size())

    # Prepare a Dataset that only yields our feature.
    feature_ds = feature_ds.map(index)

    # Learn the space of possible indices.
    encoder.adapt(feature_ds)

    # Apply one-hot encoding to our indices. The lambda function captures the
    # layer so we can use them, or include them in the functional model later.
    return lambda feature: encoder(index(feature))


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
    # print(train_ds)
    # name = 'bmi'
    # feature_ds = train_ds.map(lambda x, y: x[name])
    # print(feature_ds)
    # exit()

    # Instantiate tensors for each input with tf.keras.Input().
    # Numeric features:  age, bmi, children.
    # numerics = ['age', 'bmi', 'children']
    numerics = ['bmi', 'children']
    # Categorical features:  sex, smoker, region.
    categoricals = ['sex', 'smoker', 'region']
    # Categorical features as integers:  age.  Maybe?
    cat_integers = ['age']

    inputs = {}

    # print(list(train_f.items()))

    for name, column in train_f.items():
        dtype = ''
        if name in numerics:
            dtype = tf.float32
        elif name in categoricals:
            dtype = tf.string
        elif name in cat_integers:
            dtype = tf.int64
        else:
            raise ValueError('{} is an unrecognized feature name.'.format(name))

        inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

    print(inputs)

    # Normalize the numeric inputs.

    # Dictionary of numeric inputs.
    numeric_inputs = {name: input for name, input in inputs.items()
                      if input.dtype == tf.float32}

    # Concatenate the numeric inputs.
    x = tf.keras.layers.Concatenate()(list(numeric_inputs.values()))

    # Initialize and adapt the normalization layer.
    normalizer = preprocessing.Normalization()
    normalizer.adapt(np.array(train_f[numeric_inputs.keys()]))

    # Normalize the numeric inputs.
    all_numeric_inputs = normalizer(x)

    print(all_numeric_inputs)

    processed_inputs = [all_numeric_inputs]

    # Categorical features encoded as integers.
    age_col = tf.keras.Input(shape=(1,), name='age', dtype='int64')
    encoding_layer = get_category_encoding_layer('age',
                                                 train_ds,
                                                 dtype='int64',
                                                 max_tokens=5)
    encoded_age_col = encoding_layer(age_col)
    # processed_inputs.append(age_col)
    processed_inputs.append(encoded_age_col)

    # Encode the categorical inputs.

    for name, input in inputs.items():
        if ((input.dtype == tf.float32) or (input.dtype == tf.int64)):
            continue

        lookup = preprocessing.StringLookup(
            vocabulary=np.unique(train_f[name]))
        one_hot = preprocessing.CategoryEncoding(
            max_tokens=lookup.vocab_size())

        x = lookup(input)
        x = one_hot(x)
        processed_inputs.append(x)

    processed_input_layers = tf.keras.layers.Concatenate()(processed_inputs)

    # processing_model = tf.keras.Model(inputs, processed_input_layers)

    # tf.keras.utils.plot_model(model=processing_model,
    #                           rankdir="LR",
    #                           dpi=72,
    #                           show_shapes=True)

    output = tf.keras.layers\
                     .Dense(64, activation='relu')(processed_input_layers)
    output = tf.keras.layers\
                     .Dense(64, activation='relu')(output)
    output = tf.keras.layers.Dense(1)(output)
    model = tf.keras.Model(inputs, output)

    model.compile(optimizer=tf.optimizers.Adam(),
                  loss=tf.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['mae', 'acc'])

    model.fit(train_ds,
              validation_data=val_ds,
              epochs=25)

    loss, mae, mse = model.evaluate(test_ds)
    print(loss, mae, mse)

    tf.keras.utils.plot_model(model=model,
                              rankdir="LR",
                              dpi=72,
                              show_shapes=True)

    # train_fd = {name: np.array(value)
    #             for name, value in train_feat.items()}

    # def train_model(preprocessing_head, inputs):
    #     body = tf.keras.Sequential([
    #         tf.keras.layers.Dense(64),
    #         tf.keras.layers.Dense(1)
    #     ])

    #     preprocessed_inputs = preprocessing_head(inputs)
    #     result = body(preprocessed_inputs)
    #     model = tf.keras.Model(inputs, result)

    #     model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True),
    #                   optimizer=tf.optimizers.Adam())

    #     return model

    # train_model = train_model(train_pp, inputs)

    # train_model.fit(train_fd,
    #                 train_labels,
    #                 validation_split=0.2,
    #                 epochs=10)

    # test_results = {}
    # test_results['dnn_model'] = train_model.evaluate(
    #     test_data, test_labels, verbose=0)
    # print(test_results)

    # test_feat = test_data.copy()
    # test_fd = {name: np.array(value)
    #             for name, value in test_feat.items()}

    # print(train_model.evaluate(test_fd, test_labels, verbose=2))
    # print(train_model.predict(test_fd))

    # Create model.
    # model = create_model()
    # Train model.
    # model.train()
    # Test model.
    # test_predictions(model, test_data, test_labels)
    # Plot predictions of model.
    # plot_predictions(model, test_data, test_labels)

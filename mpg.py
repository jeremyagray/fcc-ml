#!/usr/bin/env python

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/' \
    'auto-mpg/auto-mpg.data'

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower',
                'Weight', 'Acceleration', 'Model Year', 'Origin']

raw = pd.read_csv(url, names=column_names,
                  na_values='?', comment='\t',
                  sep=' ', skipinitialspace=True)

# Copy the original to features (for future label removal).
rawf = raw.copy()

# Drop na.
rawf = rawf.dropna()

# Pandas "one-hot."
rawf['Origin'] = rawf['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
rawf = pd.get_dummies(rawf, prefix='', prefix_sep='')

# print(rawf.tail())

# Split data.
traind = rawf.sample(frac=0.8, random_state=0)
testd = rawf.drop(traind.index)

# Copy to features and remove labels.
trainf = traind.copy()
testf = testd.copy()
trainl = trainf.pop('MPG')
testl = testf.pop('MPG')

# Inspect the data.
# graph = sns.pairplot(trainf[['MPG', 'Cylinders', 'Displacement', 'Weight']],
#                      diag_kind='kde')
# fig = graph.fig
# fig.savefig('mpg.png')
# print(trainf.describe().transpose())

# print(traind.describe().transpose()[['mean', 'std']])

# Normalizer layer.
norm = preprocessing.Normalization()
norm.adapt(np.array(trainf))
# print(norm.mean.numpy())
# first = np.array(trainf[:1])

# with np.printoptions(precision=2, suppress=True):
#     print('First example:', first)
#     print('Normalized:', norm(first).numpy())
# print(norm(trainf))

# Linear regression.

# hp = np.array(trainf['Horsepower'])

# hpnorm = preprocessing.Normalization(input_shape=[1, ])
# hpnorm.adapt(hp)
# hpm = tf.keras.Sequential([
#     hpnorm,
#     layers.Dense(units=1)
# ])

# hpm.summary()
# hpm.compile(
#     optimizer=tf.optimizers.Adam(learning_rate=0.1),
#     loss='mean_absolute_error')

# history = hpm.fit(
#     trainf['Horsepower'],
#     trainl,
#     epochs=100,
#     verbose=0,
#     # Calculate validation results on 20% of the training data
#     validation_split=0.2)

# hist = pd.DataFrame(history.history)
# hist['epoch'] = history.epoch

# print(hist.tail())


# def plot_loss(history):
#     plt.plot(history.history['loss'], label='loss')
#     plt.plot(history.history['val_loss'], label='val_loss')
#     plt.ylim([0, 10])
#     plt.xlabel('Epoch')
#     plt.ylabel('Error [MPG]')
#     plt.legend()
#     plt.grid(True)


# plot_loss(history)

test_results = {}

# test_results['horsepower_model'] = hpm.evaluate(
#     testf['Horsepower'],
#     testl,
#     verbose=0)

# x = tf.linspace(0.0, 250, 251)
# y = hpm.predict(x)


# def plot_horsepower(x, y):
#     plt.scatter(trainf['Horsepower'], trainl, label='Data')
#     plt.plot(x, y, color='k', label='Predictions')
#     plt.xlabel('Horsepower')
#     plt.ylabel('MPG')
#     plt.legend()
#     # fig.savefig('hp.png')

# plot_horsepower(x,y)

# # Multiple linear.
# mm = tf.keras.Sequential([
#     norm,
#     layers.Dense(units=1)
# ])

# mm.summary()
# mm.compile(
#     optimizer=tf.optimizers.Adam(learning_rate=0.1),
#     loss='mean_absolute_error')

# history = mm.fit(
#     trainf,
#     trainl,
#     epochs=100,
#     verbose=0,
#     validation_split=0.2)

# test_results['multiple_linear'] = mm.evaluate(
#     testf, testl, verbose=0)

def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))

    return model

# dhpm = build_and_compile_model(hpnorm)
# dhpm.summary()

# history = dhpm.fit(
#     trainf['Horsepower'],
#     trainl,
#     validation_split=0.2,
#     verbose=0,
#     epochs=100)

# test_results['dnn_horsepower_model'] = dhpm.evaluate(
#     testf['Horsepower'],
#     testl,
#     verbose=0)

dmm = build_and_compile_model(norm)
dmm.summary()

history = dmm.fit(
    trainf,
    trainl,
    validation_split=0.2,
    verbose=0,
    epochs=100)

test_results['dmm'] = dmm.evaluate(
    testf,
    testl,
    verbose=0)

print(pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T)

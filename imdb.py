#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def plot_graphs(history, metric):
    fig = plt.figure(figsize=(8, 8))

    ax = plt.subplot(1, 2, 1)
    ax.plot(history.history[metric])
    ax.plot(history.history['val_' + metric], '')
    ax.xlabel("Epochs")
    ax.ylabel(metric)
    ax.legend([metric, 'val_' + metric])

    fig.savefig('imdb-loss.png')

    return


dataset, info = tfds.load('imdb_reviews', with_info=True,
                          as_supervised=True)
train_ds, test_ds = dataset['train'], dataset['test']

# print(train_ds.element_spec)

BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_ds = train_ds\
    .shuffle(BUFFER_SIZE)\
    .batch(BATCH_SIZE)\
    .prefetch(tf.data.experimental.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

print(type(train_ds))
print(type(dataset))
print(train_ds)
print(dataset)
for example, label in train_ds.take(1):
    print('text: ', example.numpy())
    print('label: ', label.numpy())
exit()
# for example, label in train_ds.take(1):
#     print('texts: ', example.numpy()[:3])
#     print('labels: ', label.numpy()[:3])

VOCAB_SIZE = 1000
encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=VOCAB_SIZE)
encoder.adapt(train_ds.map(lambda text, label: text))

vocab = np.array(encoder.get_vocabulary())
vocab[:20]

# encoded_example = encoder(example)[:3].numpy()
# print(encoded_example)

# for n in range(3):
#     print("Original: ", example[n].numpy())
#     print("Round-trip: ", " ".join(vocab[encoded_example[n]]))

model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# print([layer.supports_masking for layer in model.layers])

# predict on a sample text without padding.

sample_text = ('The movie was cool. The animation and the graphics '
               'were out of this world. I would recommend this movie.')
predictions = model.predict(np.array([sample_text]))
# print(predictions[0])

# predict on a sample text with padding

padding = "the " * 2000
predictions = model.predict(np.array([sample_text, padding]))
# print(predictions[0])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(train_ds, epochs=10,
                    validation_data=test_ds,
                    validation_steps=30)

test_loss, test_acc = model.evaluate(test_ds)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))

# plt.figure(figsize=(16,8))
# plt.subplot(1,2,1)
# plot_graphs(history, 'accuracy')
# plt.ylim(None,1)
# plt.subplot(1,2,2)
# plot_graphs(history, 'loss')
# plt.ylim(0,None)

sample_text = ('The movie was cool. The animation and the graphics '
               'were out of this world. I would recommend this movie.')
predictions = model.predict(np.array([sample_text]))

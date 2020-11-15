#!/usr/bin/env python

import numpy as np
import os
import pandas as pd
import shutil
import tensorflow as tf
import urllib.request


# Download and load data.
def load_data():
    # Temporary directory paths.
    tmp = '/home/gray/tmp'
    directory = 'fcc-ml'
    tmpdir = os.path.join(tmp, directory)

    # Test for the data files before downloading them.
    train_name = 'train-data.tsv'
    test_name = 'valid-data.tsv'
    train_path = os.path.join(tmpdir, train_name)
    test_path = os.path.join(tmpdir, test_name)

    if not (os.path.isfile(train_path) and os.path.isfile(test_path)):
        # Create the temporary directory if it does not exist.
        if os.path.exists(tmp):
            if not os.path.exists(tmpdir):
                os.makedirs(tmpdir)

        # Data URLs.
        train_url = "https://raw.githubusercontent.com/beaucarnes/"\
            "fcc_python_curriculum/master/sms/train-data.tsv"
        test_url = "https://raw.githubusercontent.com/beaucarnes/"\
            "fcc_python_curriculum/master/sms/valid-data.tsv"

        # Make a good fake request for the CDN.
        req = urllib.request.Request(train_url,
                                     headers={'User-Agent': "Magic Browser"})

        # Download the file and save it.
        with urllib.request.urlopen(req)\
             as response, open(train_path, 'wb') as file:
            shutil.copyfileobj(response, file)

        # Make a good fake request for the CDN.
        req = urllib.request.Request(test_url,
                                     headers={'User-Agent': "Magic Browser"})

        # Download the file and save it.
        with urllib.request.urlopen(req)\
             as response, open(test_path, 'wb') as file:
            shutil.copyfileobj(response, file)

    train = pd.read_csv(train_path, sep='\t', names=['label', 'message'])
    test = pd.read_csv(test_path, sep='\t', names=['label', 'message'])

    return (train, test)


def test_predictions(model):
    test_messages = [
        "how are you doing today",
        "sale today! to stop texts call 98912460324",
        "i dont want to go. can we try it a different day? available sat",
        "our new mobile video service is live. "
        "just install on your phone to start watching.",
        "you have won £1000 cash! call to claim your prize.",
        "i'll bring it tomorrow. don't forget the milk.",
        "wow, is your arm alright. that happened to me one time too"
    ]

    test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
    passed = True

    for msg, ans in zip(test_messages, test_answers):
        prediction = predict_message(model, msg)
        if prediction[1] != ans:
            passed = False

    if passed:
        print("You passed the challenge.  Great job!")
    else:
        print("You haven't passed yet.  Keep trying.")


# Predict the message type.  Returns list containing probability and label.
# prediction = predict_message(msg)
# print(prediction)
# >>> [0.008318834938108921, 'ham']
def predict_message(model, msg):
    prediction = model.predict(np.array([msg])).flatten()[0]
    # print('msg:  {} type:  {}'.format(msg, prediction))

    if prediction > 0:
        return [prediction, 'ham']
    else:
        return [prediction, 'spam']


# Pandas dataframe to tensorflow.data dataset.
# Requires whole dataframe, with features to use and target.
# https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers
def df_to_ds(df, target, shuffle=True, batch=32):
    features = df.copy()
    labels = features.pop(target)

    ds = tf.data.Dataset.from_tensor_slices(
        (features['message'].values, labels.values))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(features))

    ds = ds.batch(batch)
    ds = ds.prefetch(batch)

    return ds


if __name__ == '__main__':
    HAM = 1
    SPAM = 0

    (train_d, val_d) = load_data()

    train_d['target'] = np.where(train_d['label'] == 'ham', HAM, SPAM)
    train_d = train_d.drop(columns=['label'])
    val_d['target'] = np.where(val_d['label'] == 'ham', HAM, SPAM)
    val_d = val_d.drop(columns=['label'])

    TRAIN_MSGS = train_d.shape[0]
    VAL_MSGS = val_d.shape[0]

    # Features.
    train_f = train_d.copy()
    val_f = val_d.copy()

    # Labels (targets).
    train_l = train_f.pop('target')
    val_l = val_f.pop('target')

    BUFFER_SIZE = 10000
    BATCH_SIZE = 64

    train_ds = df_to_ds(train_d, 'target', shuffle=True, batch=BATCH_SIZE)
    val_ds = df_to_ds(val_d, 'target', shuffle=False, batch=BATCH_SIZE)

    VOCAB_SIZE = 1000
    encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
        max_tokens=VOCAB_SIZE)
    encoder.adapt(train_ds.map(lambda text, label: text))

    vocab = np.array(encoder.get_vocabulary())

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

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(1e-4),
        metrics=['accuracy'],
    )

    history = model.fit(
        train_ds,
        epochs=10,
        validation_data=val_ds,
    )

    val_loss, val_acc = model.evaluate(val_ds)

    print('Validation Loss: {}'.format(val_loss))
    print('Validation Accuracy: {}'.format(val_acc))

    test_predictions(model)

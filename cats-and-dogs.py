#!/usr/bin/env python

# import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
# import shutil
import tensorflow.keras as keras
# import urllib
# import zipfile


def plotImages(images_arr, probabilities=False):
    fig, axes = plt.subplots(len(images_arr),
                             1,
                             figsize=(5, len(images_arr) * 3))

    if probabilities is False:
        for img, ax in zip(images_arr, axes):
            ax.imshow(img)
            ax.axis('off')
    else:
        for img, probability, ax in zip(images_arr, probabilities, axes):
            ax.imshow(img)
            ax.axis('off')
            if probability > 0.5:
                ax.set_title("%.2f" % (probability * 100) + "% dog")
            else:
                ax.set_title("%.2f" % ((1 - probability) * 100) + "% cat")

    plt.show()


# URL = 'https://cdn.freecodecamp.org/'
# 'project-data/cats-and-dogs/cats_and_dogs.zip'

# The cdn requires a user agent
# req = urllib.request.Request(URL, headers={'User-Agent': "Magic Browser"})

# Download the file from `url` and save it.
data = 'cats_and_dogs.zip'
# with urllib.request.urlopen(req) as response, open(data, 'wb') as out_file:
#     shutil.copyfileobj(response, out_file)

images = os.path.join(os.path.dirname(data), 'cats_and_dogs')

# with zipfile.ZipFile(data, 'r') as archive:
#     archive.extractall()

train_dir = os.path.join(images, 'train')
validation_dir = os.path.join(images, 'validation')
test_dir = os.path.join(images, 'test')

# Get number of files in each directory. The train and validation
# directories each have the subdirecories "dogs" and "cats."
total_train = sum([len(files) for r, d, files in os.walk(train_dir)])
total_val = sum([len(files) for r, d, files in os.walk(validation_dir)])
total_test = len(os.listdir(test_dir))

# Variables for pre-processing and training.
batch = 128
epochs = 15
height = 150
width = 150
dim = (height, width)
scale = 1.0 / 255.0

# training_images = ImageDataGenerator(rescale=scale)
training_images = ImageDataGenerator(rescale=(1.0 / 255.0),
                                     rotation_range=90,
                                     width_shift_range=0.3,
                                     height_shift_range=0.3,
                                     brightness_range=(0.5, 1.5),
                                     shear_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True,
                                     vertical_flip=True)

validation_images = ImageDataGenerator(rescale=scale)
testing_images = ImageDataGenerator(rescale=scale)

# training_data = \
#   training_images.flow_from_directory(batch_size=batch,
#                                       target_size=dim,
#                                       directory=train_dir)
training_data = \
    training_images.flow_from_directory(batch_size=batch,
                                        directory=train_dir,
                                        target_size=dim,
                                        class_mode='binary')
validation_data = \
    validation_images.flow_from_directory(batch_size=batch,
                                          target_size=(height, width),
                                          directory=validation_dir)
testing_data = \
    testing_images.flow_from_directory(batch_size=batch,
                                       target_size=dim,
                                       directory=images,
                                       classes=['test'],
                                       shuffle=False)

# Show some sample images.
# training_samples, _ = next(training_data)
# plotImages(training_samples[:5])

# Show some images to check random transformations.
# augmented_images = [training_data[0][0][0] for i in range(5)]
# plotImages(augmented_images)

# The model.
model = Sequential()
model.add(keras.Input(shape=(height, width, 3)))

# 76% correct.
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(metrics=['accuracy'],
              optimizer='rmsprop',
              loss='binary_crossentropy')

# 66% correct.
# model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))
# model.summary()

# model.compile(metrics=['accuracy'],
#               optimizer='rmsprop',
#               loss='binary_crossentropy')

# 68% correct.
# model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# model.compile(metrics=['accuracy'],
#               optimizer='adam',
#               loss='binary_crossentropy')

model.summary()

# Train the model.
history = model.fit(x=training_data,
                    epochs=epochs,
                    steps_per_epoch=2000 // batch,
                    validation_data=validation_data,
                    validation_steps=800 // batch)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

# Loss/validation plots.
# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()

# test_images, _ = next(testing_data)
probabilities = (model.predict(testing_data) > 0.5).astype("int32")
# plotImages(test_images, probabilities=probabilities)

answers = [
    1, 0, 0, 1, 0, 0, 0, 0, 1, 1,
    0, 1, 0, 1, 0, 1, 1, 0, 1, 1,
    0, 0, 1, 1, 1, 1, 1, 0, 0, 0,
    0, 0, 1, 1, 0, 1, 1, 1, 1, 0,
    1, 0, 1, 1, 0, 0, 0, 0, 0, 0
]

correct = 0

for probability, answer in zip(probabilities, answers):
    print(probability)
    if np.round(probability, decimals=0) == answer:
        correct += 1

percentage_identified = (correct / len(answers))
passed_challenge = percentage_identified > 0.63

print("correct:  {} total:  {}  percent:  {}"
      .format(correct, len(answers), percentage_identified))
print("Your model correctly identified {}% "
      "of the images of cats and dogs."
      .format(int(round(percentage_identified, 2) * 100)))

if passed_challenge:
    print("You passed the challenge!")
else:
    print("You haven't passed yet.  "
          "Your model should identify at least 63% of the images.  "
          "Keep trying.  You will get it!")

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

model_filename = "fashion_mnist.h5"
overwrite_model = True
restore = False
epochs = 20

img_height = 28
img_width = 28
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]
num_classes = len(class_names)

data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# train_images = train_images / 255.0
# test_images = test_images / 255.0

val_images, val_labels = train_images[50000:], train_labels[50000:]
train_images, train_labels = train_images[:50000], train_labels[:50000]
print(f"TTS -\nTrain:{len(train_images)}\nVal:{len(val_images)}\nTest:{len(test_images)}")


def train_model(
    train_images, train_labels, val_images, val_labels, test_images, test_labels,
    model_filename="model.h5", epochs=5
    ):
    if restore:
        print(f"Loading model from {model_filename}...")
        model = keras.models.load_model(model_filename)
    else:
        data_augmentation = keras.Sequential([
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.05),
            layers.experimental.preprocessing.RandomZoom(0.05),
            layers.experimental.preprocessing.RandomContrast(0.05),
        ])

        model = Sequential([
            layers.experimental.preprocessing.Rescaling(
                1./255, input_shape=(img_height, img_width, 1)),
            data_augmentation,
            layers.Conv2D(8, 3, padding='same', activation='swish'),
            layers.MaxPooling2D(),
            layers.Conv2D(16, 3, padding='same', activation='swish'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='swish'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='swish'),
            layers.MaxPooling2D(),
            layers.Dropout(0.15),
            layers.Flatten(),
            layers.Dense(128, activation='swish'),
            layers.Dense(num_classes)
            ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    model.summary()

    history = model.fit(
        train_images, train_labels, validation_data=(val_images, val_labels),
        verbose=1, epochs=epochs
    )

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f'Test loss: {test_loss}, Test accuracy: {test_acc}')

    visualize(history)

    model.save(model_filename)
    return model, history


def visualize(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


if not os.path.exists(model_filename) or overwrite_model:
    print("Training model...")
    model, history = train_model(
        train_images, train_labels, val_images, val_labels, test_images, test_labels,
        model_filename, epochs=epochs)
else:
    try:
        print(f"Loading model from {model_filename}...")
        model = keras.models.load_model(model_filename)
    except Exception as e:
        print(f"Failed to load model from {model_filename}. Exception:\n{e}")

prediction = model.predict(test_images)

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(f"Actual: {class_names[test_labels[i]]}")
    plt.title(f"Prediction {class_names[np.argmax(prediction[i])]}")
    plt.show()

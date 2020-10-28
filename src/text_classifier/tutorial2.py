import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

vocabulary = 88000
model_filename = "model.h5"
overwrite_model = False

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(
    num_words=vocabulary)

word_index = data.get_word_index()
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(v, k) for (k, v) in word_index.items()])

train_data = keras.preprocessing.sequence.pad_sequences(
    train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(
    test_data, value=word_index["<PAD>"], padding="post", maxlen=250)


def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

def train_model(train_data, train_labels, test_data, test_labels, model_filename):
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocabulary, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation="relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.summary()

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    x_train = train_data[10000:]
    x_val = train_data[:10000]

    y_train = train_labels[10000:]
    y_val = train_labels[:10000]

    fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

    results = model.evaluate(test_data, test_labels)
    print(results)

    model.save("model.h5")
    return model


if not os.path.exists(model_filename) or overwrite_model:
    print("Training model...")
    model = train_model(train_data, train_labels, test_data, test_labels, model_filename)
else:
    try:
        print(f"Loading model from {model_filename}...")
        model = keras.models.load_model(model_filename)
    except Exception as e:
        print(f"Failed to load model from {model_filename}. Exception:\n{e}")


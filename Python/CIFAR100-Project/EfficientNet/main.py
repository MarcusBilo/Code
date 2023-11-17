import os
import warnings
import tensorflow as tf
from time import time
from keras import layers, regularizers
from keras.applications import EfficientNetV2B1
from keras.applications.efficientnet_v2 import preprocess_input
from keras.layers import Flatten, Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical


def build_model(n):
    resize_layer = layers.Lambda(lambda x: tf.image.resize(x, (224, 224)), input_shape=(32, 32, 3))
    base_model = EfficientNetV2B1(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    model = Sequential([
        resize_layer,
        base_model, Dropout(0.5),
        Flatten(),
        Dense(n,
              activation='softmax',
              kernel_regularizer=regularizers.l2(l=0.001))
    ])

    for layer in model.layers[1].layers:
        layer.trainable = False

    model.compile(
        loss="categorical_crossentropy",
        optimizer="Nadam",
        metrics=["accuracy"]
    )

    return model


def train_model(model, train_data, train_labels, val_data, val_labels):
    model.fit(train_data, train_labels,
              batch_size=1000,
              epochs=1,
              verbose=True,
              validation_data=(val_data, val_labels))


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings("ignore", module='tensorflow')

    start_time = time()

    NUMBER_OF_CLASSES = 100
    (train_data, train_labels), (val_data, val_labels) = tf.keras.datasets.cifar100.load_data()
    train_data = preprocess_input(train_data)
    val_data = preprocess_input(val_data)
    train_labels = to_categorical(train_labels, NUMBER_OF_CLASSES)
    val_labels = to_categorical(val_labels, NUMBER_OF_CLASSES)
    model = build_model(NUMBER_OF_CLASSES)
    model.summary()
    train_model(model, train_data, train_labels, val_data, val_labels)

    end_time = time()
    elapsed_time_min = ((end_time - start_time) / 60)
    print(f"\nProgram Runtime: {elapsed_time_min:.1f}min")


if __name__ == "__main__":
    main()

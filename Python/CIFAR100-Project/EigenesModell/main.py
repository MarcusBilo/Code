import datetime
import os
import warnings
from time import time
import numpy as np
import psutil
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, InputLayer
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm


def augment_data(train_data, train_labels, batch_size):
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        brightness_range=(0.875, 1.125),
        channel_shift_range=32
    )
    datagen.fit(train_data)
    data_augmented = train_data.copy()
    labels_augmented = train_labels.copy()
    pbar = tqdm(total=50000)
    for data_batch, labels_batch in datagen.flow(
            train_data,
            train_labels,
            batch_size=batch_size
    ):
        data_augmented = np.concatenate((data_augmented, data_batch))
        labels_augmented = np.concatenate((labels_augmented, labels_batch))
        if data_augmented.shape[0] >= train_data.shape[0] + 50000:
            print()
            break
        pbar.update(data_batch.shape[0])
    pbar.close()
    return data_augmented, labels_augmented


def build_model(input_shape, num_classes):
    model = Sequential([
        InputLayer(input_shape),
        Conv2D(128, (2, 2), activation="relu", padding="same"),
        Conv2D(128, (2, 2), activation="relu", padding="same"),
        MaxPooling2D(2, 2),
        BatchNormalization(),
        Conv2D(256, (2, 2), activation="relu", padding="same"),
        Conv2D(256, (2, 2), activation="relu", padding="same"),
        MaxPooling2D(2, 2),
        BatchNormalization(),
        Conv2D(512, (2, 2), activation="relu", padding="same"),
        Conv2D(512, (2, 2), activation="relu", padding="same"),
        MaxPooling2D(2, 2),
        BatchNormalization(),
        Conv2D(1024, (2, 2), activation="relu", padding="same"),
        Conv2D(1024, (2, 2), activation="relu", padding="same"),
        MaxPooling2D(2, 2),
        BatchNormalization(),
        Flatten(),
        Dense(4096, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
    return model


def train_model(model, train_data, train_labels, val_data, val_labels):
    model.compile(
        loss="kl_divergence",
        optimizer="Nadam",
        metrics=["accuracy"]
    )
    model.summary()

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(
        train_data,
        train_labels,
        batch_size=200,
        epochs=1,
        verbose=True,
        validation_data=(val_data, val_labels),
        callbacks=[tensorboard_callback]
    )


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings("ignore", module='tensorflow')
    # p = psutil.Process(os.getpid())
    # p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)

    start_time = time()

    N_CLASSES = 100
    (train_data, train_labels), (val_data, val_labels) = tf.keras.datasets.cifar100.load_data()
    data_augmented, labels_augmented = augment_data(train_data, train_labels, N_CLASSES)
    labels_augmented = tf.keras.utils.to_categorical(labels_augmented, num_classes=N_CLASSES)
    val_labels = tf.keras.utils.to_categorical(val_labels, num_classes=N_CLASSES)
    model = build_model(train_data.shape[1:], N_CLASSES)
    train_model(model, data_augmented, labels_augmented, val_data, val_labels)

    end_time = time()
    elapsed_time_min = ((end_time - start_time) / 60)
    print(f"\nProgram Runtime: {elapsed_time_min:.1f}min")


if __name__ == '__main__':
    main()

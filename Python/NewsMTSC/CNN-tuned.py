import json
import os
import silence_tensorflow.auto  # pip install silence-tensorflow
import spacy
import numpy as np
from tabulate import tabulate
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from keras.models import Sequential
from keras.layers import Dense, Flatten, Masking, Dropout, Conv1D
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.losses import categorical_crossentropy
from keras.metrics import CategoricalAccuracy
import tensorflow as tf
import psutil
from keras.models import model_from_json


tf.random.set_seed(2024)
# spacy.cli.download("en_core_web_lg")
nlp = spacy.load("en_core_web_lg")
p = psutil.Process(os.getpid())
p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)


def preprocess_tensorflow(data):
    processed_data = []
    for text in data:
        lemmatized_text = ' '.join([token.lemma_ for token in nlp(text) if not token.is_stop])
        doc_lemmatized = nlp(lemmatized_text)
        doc_vector = doc_lemmatized.vector
        processed_data.append(doc_vector)
    processed_data = np.array(processed_data)
    processed_data = processed_data.reshape((processed_data.shape[0], processed_data.shape[1], 1))
    return processed_data


def undersample_classes(data, labels):
    positive_indices = [i for i, label in enumerate(labels) if label == "positive"]
    negative_indices = [i for i, label in enumerate(labels) if label == "negative"]
    neutral_indices = [i for i, label in enumerate(labels) if label == "neutral"]

    negative_resampled = resample(negative_indices, n_samples=len(positive_indices), random_state=2024)
    neutral_resampled = resample(neutral_indices, n_samples=len(positive_indices), random_state=2024)

    np.random.seed(2024)
    balanced_indices = positive_indices + list(negative_resampled) + list(neutral_resampled)
    np.random.shuffle(balanced_indices)

    balanced_data = [data[i] for i in balanced_indices]
    balanced_labels = [labels[i] for i in balanced_indices]

    return balanced_data, balanced_labels


def load_data(x):
    train_data = []
    with open("train.jsonl", "r", encoding="utf-8") as train_file:
        for line in train_file:
            train_data.append(json.loads(line))

    train_data_normalized = [item["sentence_normalized"] for item in train_data]
    y_train = [item["targets"][0]["polarity"] for item in train_data]

    test_data = []
    with open("test.jsonl", "r", encoding="utf-8") as test_file:
        for line in test_file:
            test_data.append(json.loads(line))

    test_data_normalized = [item["sentence_normalized"] for item in test_data]
    y_test = [item["targets"][0]["polarity"] for item in test_data]

    sentiment_map = {2.0: "negative", 4.0: "neutral", 6.0: "positive"}
    train_labels_map = [sentiment_map[p] for p in y_train]
    test_labels_map = [sentiment_map[p] for p in y_test]

    if x == "undersampled":
        train_data, train_labels = undersample_classes(train_data_normalized, train_labels_map)
        test_data, test_labels = undersample_classes(test_data_normalized, test_labels_map)
        return train_data, test_data, train_labels, test_labels
    else:
        return train_data_normalized, test_data_normalized, train_labels_map, test_labels_map


def preprocess_labels(label_encoder, train_labels, test_labels, num_classes=3):
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    train_labels_categorical = to_categorical(y=train_labels_encoded, num_classes=num_classes)
    test_labels_encoded = label_encoder.fit_transform(test_labels)
    test_labels_categorical = to_categorical(y=test_labels_encoded, num_classes=num_classes)
    return train_labels_categorical, test_labels_categorical


def cnn_11_best():
    model = Sequential()
    model.add(Masking(mask_value=0))
    model.add(Conv1D(filters=192, kernel_size=8, strides=2))
    model.add(Dropout(0.1))
    model.add(Dense(units=544, activation="hard_sigmoid"))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(3, activation="softmax"))
    adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.002)
    model.compile(optimizer=adam, loss=categorical_crossentropy, metrics=CategoricalAccuracy())
    model._name = "CNN_11_best"
    return model


def cnn_11_sec_best():
    model = Sequential()
    model.add(Masking(mask_value=0))
    model.add(Conv1D(filters=992, kernel_size=18, strides=5))
    model.add(Dropout(0.1))
    model.add(Dense(units=736, activation="hard_sigmoid"))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(3, activation="softmax"))
    adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.0024)
    model.compile(optimizer=adam, loss=categorical_crossentropy, metrics=CategoricalAccuracy())
    model._name = "CNN_11_sec_best"
    return model


def cnn_12_best():
    model = Sequential()
    model.add(Masking(mask_value=0))
    model.add(Conv1D(filters=640, kernel_size=12, strides=6))
    model.add(Dropout(0.5))
    model.add(Dense(units=832, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units=544, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(3, activation="softmax"))
    adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.002474)
    model.compile(optimizer=adam, loss=categorical_crossentropy, metrics=CategoricalAccuracy())
    model._name = "CNN_12_best"
    return model


def cnn_12_sec_best():
    model = Sequential()
    model.add(Masking(mask_value=0))
    model.add(Conv1D(filters=416, kernel_size=14, strides=7))
    model.add(Dropout(0.5))
    model.add(Dense(units=544, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(units=512, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(3, activation="softmax"))
    adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.0006)
    model.compile(optimizer=adam, loss=categorical_crossentropy, metrics=CategoricalAccuracy())
    model._name = "CNN_12_sec_best"
    return model


def cnn_21_best():
    model = Sequential()
    model.add(Masking(mask_value=0))
    model.add(Conv1D(filters=960, kernel_size=4, strides=5))
    model.add(Dropout(0.4))
    model.add(Conv1D(filters=1024, kernel_size=6, strides=2))
    model.add(Dropout(0.2))
    model.add(Dense(units=512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(3, activation="softmax"))
    adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001)
    model.compile(optimizer=adam, loss=categorical_crossentropy, metrics=CategoricalAccuracy())
    model._name = "CNN_21_best"
    return model


def cnn_21_sec_best():
    model = Sequential()
    model.add(Masking(mask_value=0))
    model.add(Conv1D(filters=480, kernel_size=2, strides=4))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=192, kernel_size=10, strides=9))
    model.add(Dropout(0.4))
    model.add(Dense(units=864, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(3, activation="softmax"))
    adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.0002)
    model.compile(optimizer=adam, loss=categorical_crossentropy, metrics=CategoricalAccuracy())
    model._name = "CNN_21_sec_best"
    return model


def cnn_22_best():
    model = Sequential()
    model.add(Masking(mask_value=0))
    model.add(Conv1D(filters=32, kernel_size=16, strides=8))
    model.add(Dropout(0.1))
    model.add(Conv1D(filters=160, kernel_size=8, strides=5))
    model.add(Dropout(0.2))
    model.add(Dense(units=832, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units=128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(3, activation="softmax"))
    adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.0018)
    model.compile(optimizer=adam, loss=categorical_crossentropy, metrics=CategoricalAccuracy())
    model._name = "CNN_22_best"
    return model


def cnn_22_sec_best():
    model = Sequential()
    model.add(Masking(mask_value=0))
    model.add(Conv1D(filters=704, kernel_size=6, strides=7))
    model.add(Dropout(0.4))
    model.add(Conv1D(filters=832, kernel_size=14, strides=6))
    model.add(Dropout(0.2))
    model.add(Dense(units=320, activation="hard_sigmoid"))
    model.add(Dropout(0.1))
    model.add(Dense(units=640, activation="hard_sigmoid"))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(3, activation="softmax"))
    adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.0016)
    model.compile(optimizer=adam, loss=categorical_crossentropy, metrics=CategoricalAccuracy())
    model._name = "CNN_22_sec_best"
    return model


def main():

    """
    def save_model_architecture(model, filename):
        model_json = model.to_json()
        with open(filename, 'w') as json_file:
            json_file.write(model_json)

    model = cnn_21_best()
    save_model_architecture(model, 'cnn_21_best_architecture.json')
    """

    """
    label_encoder = LabelEncoder()

    train_data, test_data, train_labels, test_labels = load_data("undersampled")
    train_data_tf, test_data_tf = preprocess_tensorflow(train_data), preprocess_tensorflow(test_data)
    train_labels_one_hot, test_labels_one_hot = preprocess_labels(label_encoder, train_labels, test_labels, num_classes=3)

    def load_model_architecture(filename):
        with open(filename, 'r') as json_file:
            model_json = json_file.read()
        return model_from_json(model_json)

    loaded_model = load_model_architecture('cnn_12_sec_best_architecture.json')
    loaded_model.build((None, 300, 1))
    adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.0006)
    loaded_model.load_weights('10_epochs_CNN_0.6136_weights.h5')
    loaded_model.compile(optimizer=adam, loss=categorical_crossentropy, metrics=CategoricalAccuracy())

    train_predictions = loaded_model.predict(train_data_tf, verbose=0)
    train_accuracy = accuracy_score(train_labels_one_hot.argmax(axis=1), np.argmax(train_predictions, axis=1))
    test_predictions = loaded_model.predict(test_data_tf, verbose=0)
    test_accuracy = accuracy_score(test_labels_one_hot.argmax(axis=1), np.argmax(test_predictions, axis=1))

    print(train_accuracy)
    print(test_accuracy)
    """

    classifiers = [
        cnn_11_best(),
        cnn_11_sec_best(),
        cnn_12_best(),
        cnn_12_sec_best(),
        cnn_21_best(),
        cnn_21_sec_best(),
        cnn_22_best(),
        cnn_22_sec_best()
    ]

    results = []
    train_accuracy, test_accuracy = 0.0, 0.0
    label_encoder = LabelEncoder()

    for _ in tqdm(range(1), desc=f"Preprocessing Data"):
        train_data, test_data, train_labels, test_labels = load_data("undersampled")
        train_data_tf, test_data_tf = preprocess_tensorflow(train_data), preprocess_tensorflow(test_data)
        train_labels_one_hot, test_labels_one_hot = preprocess_labels(label_encoder, train_labels, test_labels, num_classes=3)

    for clf in classifiers:
        iteration_losses = []
        epoch = 0
        if clf._name == "CNN_11_best" or clf._name == "CNN_21_best" or clf._name == "CNN_21_sec_best":
            highest_test_accuracy = 0.0
            for _ in tqdm(range(25), desc=f"Processing {getattr(clf, 'name', clf.__class__.__name__)}", unit="Epoch"):
                epoch += 1
                loss = clf.fit(train_data_tf, train_labels_one_hot, verbose=0, epochs=1, batch_size=71).history['loss'][0]
                iteration_losses.append(round(loss, 4))
                train_predictions = clf.predict(train_data_tf, verbose=0)
                train_accuracy = accuracy_score(train_labels_one_hot.argmax(axis=1), np.argmax(train_predictions, axis=1))
                test_predictions = clf.predict(test_data_tf, verbose=0)
                test_accuracy = accuracy_score(test_labels_one_hot.argmax(axis=1), np.argmax(test_predictions, axis=1))
                if test_accuracy > highest_test_accuracy and test_accuracy > 0.6:
                    highest_test_accuracy = test_accuracy
                    rounded_accuracy = round(highest_test_accuracy, 4)
                    filename = f"{clf._name}_{rounded_accuracy}_e{epoch}_weights.h5"
                    clf.save_weights(filename)
        elif clf._name == "CNN_11_sec_best" or clf._name == "CNN_12_best" or clf._name == "CNN_12_sec_best" or clf._name == "CNN_22_best" or clf._name == "CNN_22_sec_best":
            highest_test_accuracy = 0.0
            for _ in tqdm(range(25), desc=f"Processing {getattr(clf, 'name', clf.__class__.__name__)}", unit="Epoch"):
                epoch += 1
                loss = clf.fit(train_data_tf, train_labels_one_hot, verbose=0, epochs=1, batch_size=142).history['loss'][0]
                iteration_losses.append(round(loss, 4))
                train_predictions = clf.predict(train_data_tf, verbose=0)
                train_accuracy = accuracy_score(train_labels_one_hot.argmax(axis=1), np.argmax(train_predictions, axis=1))
                test_predictions = clf.predict(test_data_tf, verbose=0)
                test_accuracy = accuracy_score(test_labels_one_hot.argmax(axis=1), np.argmax(test_predictions, axis=1))
                if test_accuracy > highest_test_accuracy and test_accuracy > 0.6:
                    highest_test_accuracy = test_accuracy
                    rounded_accuracy = round(highest_test_accuracy, 4)
                    filename = f"{clf._name}_{rounded_accuracy}_e{epoch}_weights.h5"
                    clf.save_weights(filename)
        else:
            raise Exception("no defined training for ", getattr(clf, 'name', clf.__class__.__name__))

        results.append([
            getattr(clf, 'name', clf.__class__.__name__),
            round(train_accuracy, 4),
            round(test_accuracy, 4),
            iteration_losses
        ])

    headers = ["Classifier", "Train Acc", "Test Acc", "Loss"]
    print("\n", tabulate(results, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    main()

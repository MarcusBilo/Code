import json
import os
import silence_tensorflow.auto  # pip install silence-tensorflow
import spacy
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from keras.models import Sequential
from keras.layers import Dense, Flatten, Masking, Dropout, LSTM, Bidirectional, Conv1D, GRU
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.losses import categorical_crossentropy
from keras.metrics import CategoricalAccuracy
import tensorflow as tf
from keras.models import model_from_json
from sklearn.metrics import confusion_matrix


tf.random.set_seed(2024)
# spacy.cli.download("en_core_web_lg")
nlp = spacy.load("en_core_web_lg")


def preprocess_tensorflow(data):
    """
    This function takes a list of text data and performs preprocessing using spaCy for lemmatization and
    stop-word removal. It then converts the processed text into numerical vectors using spaCy's word vectors.
    After that it converts it into a TensorFlow-compatible format.

    Parameters:
    - data (list): List of strings representing the input text data.

    Returns:
    - processed_data (numpy.ndarray): Processed data in TensorFlow-compatible format.
    """
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
    """
    This function performs undersampling of the majority classes ('negative' and 'neutral') to balance the dataset.
    It resamples the 'negative' and 'neutral' classes to match the number of instances in the 'positive' class.

    Parameters:
    - data (list): A list containing the input data.
    - labels (list): A list containing class labels corresponding to the input data.

    Returns:
    - balanced_data (list): A list of input data after undersampling.
    - balanced_labels (list): A list of corresponding class labels after undersampling.
    """
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
    """
    This function loads and preprocesses sentiment analysis data from JSONL files ('train.jsonl' and 'test.jsonl').
    It extracts normalized sentences and corresponding sentiment labels for training and testing sets.

    Parameters:
    - x (str): A string indicating whether to return the original or undersampled data.

    Returns:
    - train_data (list): A list of normalized sentences from the training set.
    - test_data (list): A list of normalized sentences from the testing set.
    - train_labels (list): A list of sentiment labels corresponding to the training set.
    - test_labels (list): A list of sentiment labels corresponding to the testing set.
    """
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
    """
    This function preprocesses the categorical labels by encoding them using a provided label encoder
    and converting them into one-hot encoded categorical format.

    Parameters:
    - label_encoder (LabelEncoder): A scikit-learn LabelEncoder instance for encoding labels.
    - train_labels (list): A list of training set labels (original categorical labels).
    - test_labels (list): A list of testing set labels (original categorical labels).
    - num_classes (int): The total number of classes. Default is 3.

    Returns:
    - train_labels_categorical (numpy.ndarray): One-hot encoded labels for the training set.
    - test_labels_categorical (numpy.ndarray): One-hot encoded labels for the testing set.
    """
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    train_labels_categorical = to_categorical(y=train_labels_encoded, num_classes=num_classes)
    test_labels_encoded = label_encoder.fit_transform(test_labels)
    test_labels_categorical = to_categorical(y=test_labels_encoded, num_classes=num_classes)
    return train_labels_categorical, test_labels_categorical


def cnn_22_sec_best():
    """
    This function defines a CNN classification model with specific configurations.
    The model is setup for a sentiment analysis task with three output classes ('positive', 'neutral', 'negative').

    Returns:
    - sequential (tf.keras.Sequential): A CNN-based sentiment analysis model with multiple Conv1D and Dense Layers.
    """
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


def gru_12_sec_best():
    """
    This function defines a GRU classification model with specific configurations.
    The model is setup for a sentiment analysis task with three output classes ('positive', 'neutral', 'negative').

    Returns:
    - sequential (tf.keras.Sequential): A GRU-based sentiment analysis model with one GRU and multiple Dense Layers.
    """
    model = Sequential()
    model.add(Masking(mask_value=0))
    model.add(GRU(units=512))
    model.add(Dropout(0.1))
    model.add(Dense(units=960, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(units=224, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(3, activation="softmax"))
    adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
    model.compile(optimizer=adam, loss=categorical_crossentropy, metrics=CategoricalAccuracy())
    model._name = "GRU_12_sec_best"
    return model


def bigru_22_sec_best():
    """
    This function defines a Bi-GRU classification model with specific configurations.
    The model is setup for a sentiment analysis task with three output classes ('positive', 'neutral', 'negative').

    Returns:
    - sequential (tf.keras.Sequential): A Bi-GRU-based sentiment analysis model with multiple Bi-GRU and Dense Layers.
    """
    model = Sequential()
    model.add(Masking(mask_value=0))
    model.add(Bidirectional(GRU(units=256, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(GRU(units=416)))
    model.add(Dropout(0.1))
    model.add(Dense(units=864, activation="hard_sigmoid"))
    model.add(Dropout(0.4))
    model.add(Dense(units=960, activation="hard_sigmoid"))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(3, activation="softmax"))
    adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.0003)
    model.compile(optimizer=adam, loss=categorical_crossentropy, metrics=CategoricalAccuracy())
    model._name = "BiGRU_22_sec_best"
    return model


def lstm_21_sec_best():
    """
    This function defines a LSTM classification model with specific configurations.
    The model is setup for a sentiment analysis task with three output classes ('positive', 'neutral', 'negative').

    Returns:
    - sequential (tf.keras.Sequential): A LSTM-based sentiment analysis model with multiple LSTM and a Dense Layers.
    """
    model = Sequential()
    model.add(Masking(mask_value=0))
    model.add(LSTM(units=768, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(units=160))
    model.add(Dropout(0.5))
    model.add(Dense(units=128, activation="linear"))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(3, activation="softmax"))
    adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.0015)
    model.compile(optimizer=adam, loss=categorical_crossentropy, metrics=CategoricalAccuracy())
    model._name = "LSTM_21_sec_best"
    return model


def bilstm_22_sec_best():
    """
    This function defines a Bi-LSTM classification model with specific configurations.
    The model is setup for a sentiment analysis task with three output classes ('positive', 'neutral', 'negative').

    Returns:
    - sequential (tf.keras.Sequential): A Bi-LSTM-based sentiment analysis model with multiple Bi-LSTM and Dense Layers.
    """
    model = Sequential()
    model.add(Masking(mask_value=0))
    model.add(Bidirectional(LSTM(units=768, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(units=128)))
    model.add(Dropout(0.3))
    model.add(Dense(units=352, activation="linear"))
    model.add(Dropout(0.1))
    model.add(Dense(units=576, activation="linear"))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(3, activation="softmax"))
    adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.0003)
    model.compile(optimizer=adam, loss=categorical_crossentropy, metrics=CategoricalAccuracy())
    model._name = "BiLSTM_22_sec_best"
    return model


def main():
    """
    This is the main function that demonstrates the usage of different classifiers (CNN, GRU, Bi-GRU, LSTM, Bi-LSTM)
    on a sentiment analysis task. It loads models, weights, data. Preprocesses data, applies various classifiers,
    and prints confusion matrices for each classifier on both the training and testing sets.
    """
    def save_model_architecture(model, filename):
        model_json = model.to_json()
        with open(filename, 'w') as json_file:
            json_file.write(model_json)

    model = cnn_22_sec_best()
    save_model_architecture(model, 'cnn_22_sec_best_architecture.json')
    model = gru_12_sec_best()
    save_model_architecture(model, 'gru_12_sec_best_architecture.json')
    model = bigru_22_sec_best()
    save_model_architecture(model, 'bigru_22_sec_best_architecture.json')
    model = lstm_21_sec_best()
    save_model_architecture(model, 'lstm_21_sec_best_architecture.json')
    model = bilstm_22_sec_best()
    save_model_architecture(model, 'bilstm_22_sec_best_architecture.json')

    label_encoder = LabelEncoder()
    train_data, test_data, train_labels, test_labels = load_data("undersampled")
    train_data_tf, test_data_tf = preprocess_tensorflow(train_data), preprocess_tensorflow(test_data)
    train_labels_one_hot, test_labels_one_hot = preprocess_labels(label_encoder, train_labels, test_labels, num_classes=3)

    def load_model_architecture(filename):
        with open(filename, 'r') as json_file:
            model_json = json_file.read()
        return model_from_json(model_json)

    loaded_model = load_model_architecture('cnn_22_sec_best_architecture.json')
    loaded_model.build((None, 300, 1))
    adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.0016)
    loaded_model.load_weights('CNN_22_sec_best_0.615_e23_weights.h5')
    loaded_model.compile(optimizer=adam, loss=categorical_crossentropy, metrics=CategoricalAccuracy())
    train_predictions = loaded_model.predict(train_data_tf, verbose=0)
    train_accuracy = accuracy_score(train_labels_one_hot.argmax(axis=1), np.argmax(train_predictions, axis=1))
    test_predictions = loaded_model.predict(test_data_tf, verbose=0)
    test_accuracy = accuracy_score(test_labels_one_hot.argmax(axis=1), np.argmax(test_predictions, axis=1))
    train_conf_matrix = confusion_matrix(train_labels_one_hot.argmax(axis=1), np.argmax(train_predictions, axis=1))
    test_conf_matrix = confusion_matrix(test_labels_one_hot.argmax(axis=1), np.argmax(test_predictions, axis=1))
    print("cnn", round(train_accuracy, 4), "\n", train_conf_matrix)
    print("cnn", round(test_accuracy, 4), "\n", test_conf_matrix, "\n")

    loaded_model = load_model_architecture('gru_12_sec_best_architecture.json')
    loaded_model.build((None, 300, 1))
    adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
    loaded_model.load_weights('GRU_12_sec_best_0.5238_e22_weights.h5')
    loaded_model.compile(optimizer=adam, loss=categorical_crossentropy, metrics=CategoricalAccuracy())
    train_predictions = loaded_model.predict(train_data_tf, verbose=0)
    train_accuracy = accuracy_score(train_labels_one_hot.argmax(axis=1), np.argmax(train_predictions, axis=1))
    test_predictions = loaded_model.predict(test_data_tf, verbose=0)
    test_accuracy = accuracy_score(test_labels_one_hot.argmax(axis=1), np.argmax(test_predictions, axis=1))
    train_conf_matrix = confusion_matrix(train_labels_one_hot.argmax(axis=1), np.argmax(train_predictions, axis=1))
    test_conf_matrix = confusion_matrix(test_labels_one_hot.argmax(axis=1), np.argmax(test_predictions, axis=1))
    print("gru", round(train_accuracy, 4), "\n", train_conf_matrix)
    print("gru", round(test_accuracy, 4), "\n", test_conf_matrix, "\n")

    loaded_model = load_model_architecture('bigru_22_sec_best_architecture.json')
    loaded_model.build((None, 300, 1))
    adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.0003)
    loaded_model.load_weights('BiGRU_22_sec_best_0.5469_e15_weights.h5')
    loaded_model.compile(optimizer=adam, loss=categorical_crossentropy, metrics=CategoricalAccuracy())
    train_predictions = loaded_model.predict(train_data_tf, verbose=0)
    train_accuracy = accuracy_score(train_labels_one_hot.argmax(axis=1), np.argmax(train_predictions, axis=1))
    test_predictions = loaded_model.predict(test_data_tf, verbose=0)
    test_accuracy = accuracy_score(test_labels_one_hot.argmax(axis=1), np.argmax(test_predictions, axis=1))
    train_conf_matrix = confusion_matrix(train_labels_one_hot.argmax(axis=1), np.argmax(train_predictions, axis=1))
    test_conf_matrix = confusion_matrix(test_labels_one_hot.argmax(axis=1), np.argmax(test_predictions, axis=1))
    print("bigru", round(train_accuracy, 4), "\n", train_conf_matrix)
    print("bigru", round(test_accuracy, 4), "\n", test_conf_matrix, "\n")

    loaded_model = load_model_architecture('lstm_21_sec_best_architecture.json')
    loaded_model.build((None, 300, 1))
    adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.0015)
    loaded_model.load_weights('LSTM_21_sec_best_0.5469_e20_weights.h5')
    loaded_model.compile(optimizer=adam, loss=categorical_crossentropy, metrics=CategoricalAccuracy())
    train_predictions = loaded_model.predict(train_data_tf, verbose=0)
    train_accuracy = accuracy_score(train_labels_one_hot.argmax(axis=1), np.argmax(train_predictions, axis=1))
    test_predictions = loaded_model.predict(test_data_tf, verbose=0)
    test_accuracy = accuracy_score(test_labels_one_hot.argmax(axis=1), np.argmax(test_predictions, axis=1))
    train_conf_matrix = confusion_matrix(train_labels_one_hot.argmax(axis=1), np.argmax(train_predictions, axis=1))
    test_conf_matrix = confusion_matrix(test_labels_one_hot.argmax(axis=1), np.argmax(test_predictions, axis=1))
    print("lstm", round(train_accuracy, 4), "\n", train_conf_matrix)
    print("lstm", round(test_accuracy, 4), "\n", test_conf_matrix, "\n")

    loaded_model = load_model_architecture('bilstm_22_sec_best_architecture.json')
    loaded_model.build((None, 300, 1))
    adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.0003)
    loaded_model.load_weights('BiLSTM_22_sec_best_0.5524_e24_weights.h5')
    loaded_model.compile(optimizer=adam, loss=categorical_crossentropy, metrics=CategoricalAccuracy())
    train_predictions = loaded_model.predict(train_data_tf, verbose=0)
    train_accuracy = accuracy_score(train_labels_one_hot.argmax(axis=1), np.argmax(train_predictions, axis=1))
    test_predictions = loaded_model.predict(test_data_tf, verbose=0)
    test_accuracy = accuracy_score(test_labels_one_hot.argmax(axis=1), np.argmax(test_predictions, axis=1))
    train_conf_matrix = confusion_matrix(train_labels_one_hot.argmax(axis=1), np.argmax(train_predictions, axis=1))
    test_conf_matrix = confusion_matrix(test_labels_one_hot.argmax(axis=1), np.argmax(test_predictions, axis=1))
    print("bilstm", round(train_accuracy, 4), "\n", train_conf_matrix)
    print("bilstm", round(test_accuracy, 4), "\n", test_conf_matrix, "\n")


if __name__ == "__main__":
    main()

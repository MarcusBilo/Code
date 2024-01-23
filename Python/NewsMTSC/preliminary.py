# python -m spacy download en_core_web_md

import json
import os
import spacy
import warnings
import numpy as np
from tabulate import tabulate
from tqdm import tqdm
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn.svm import SVC
from keras.models import Sequential, Model
from keras.layers import Conv1D, Dense, LeakyReLU, Flatten, SimpleRNN, LSTM, Bidirectional, Input
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from keras.losses import categorical_crossentropy
from transformers import BertTokenizer, TFBertForSequenceClassification


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", module='tensorflow')
warnings.filterwarnings("ignore", category=ConvergenceWarning)
nlp = spacy.load("en_core_web_md")


def print_10_longest_sentences(data, labels):
    def count_words(sentence):
        return len(sentence.split())

    print(f"\n10 Longest Sentences:")
    sorted_indices = sorted(range(len(data)), key=lambda i: count_words(data[i]), reverse=True)
    for i in range(min(10, len(data))):
        index = sorted_indices[i]
        print(f"(Word Count: {count_words(data[index])}, Label: {labels[index]})")


def preprocess_text_sklearn(data, max_length=120):
    processed_data = []
    for text in data:
        doc = nlp(text)
        doc_vector = doc.vector
        if len(doc_vector) > max_length:
            doc_vector = doc_vector[:max_length]
        else:
            padding_size = max_length - len(doc_vector)
            doc_vector = np.concatenate([doc_vector, np.zeros((padding_size,))])
        processed_data.append(doc_vector)
    return np.array(processed_data)


def preprocess_text_tensorflow(data, max_length=120):
    processed_data = []
    for text in data:
        doc = nlp(text)
        doc_vector = doc.vector
        if len(doc_vector) > max_length:
            doc_vector = doc_vector[:max_length]
        else:
            padding_size = max_length - len(doc_vector)
            doc_vector = np.concatenate([doc_vector, np.zeros((padding_size,))])
        processed_data.append(doc_vector)
    processed_data = pad_sequences(processed_data, maxlen=max_length, padding='post', truncating='post')
    processed_data = processed_data.reshape(processed_data.shape[0], processed_data.shape[1], 1)
    processed_data = processed_data.astype('float32')
    return np.array(processed_data)


def preprocess_text_bert(data, max_length=60):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    input_ids = []
    attention_masks = []
    for text in data:
        tokens = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='np'
        )
        input_ids.append(tokens['input_ids'])
        attention_masks.append(tokens['attention_mask'])
    return np.array(input_ids), np.array(attention_masks)


def undersample_classes(data, labels):
    positive_indices = [i for i, label in enumerate(labels) if label == "positive"]
    negative_indices = [i for i, label in enumerate(labels) if label == "negative"]
    neutral_indices = [i for i, label in enumerate(labels) if label == "neutral"]

    negative_resampled = resample(negative_indices, n_samples=len(positive_indices))
    neutral_resampled = resample(neutral_indices, n_samples=len(positive_indices))

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


def cnn_model():
    model = Sequential()
    model.add(Conv1D(128, 5, activation="relu"))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(3, activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model._name = "CNN"
    return model


def rnn_model():
    model = Sequential()
    model.add(SimpleRNN(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(3, activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model._name = "RNN"
    return model


def lstm_model():
    model = Sequential()
    model.add(LSTM(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(3, activation="softmax"))
    model.compile(optimizer="adam", loss=categorical_crossentropy, metrics=["accuracy"])
    model._name = "LSTM"
    return model


def bi_lstm_model():
    model = Sequential()
    model.add(Bidirectional(LSTM(128, activation="relu")))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(3, activation="softmax"))
    model.compile(optimizer="adam", loss=categorical_crossentropy, metrics=["accuracy"])
    model._name = "Bi-LSTM"
    return model


def bert_model():
    bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    for layer in bert_model.layers:
        layer.trainable = False
    input_ids = Input(shape=(60,), dtype="int32", name="input_ids")
    attention_mask = Input(shape=(60,), dtype="int32", name="attention_mask")
    outputs = bert_model(input_ids, attention_mask=attention_mask)[0]
    outputs = Dense(10, activation='relu')(outputs)
    outputs = Dense(3, activation='softmax')(outputs)
    model = Model(inputs=[input_ids, attention_mask], outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model._name = 'BERT_Model'
    return model


def main():

    classifiers = [
        SVC(),
        cnn_model(),
        rnn_model(),
        lstm_model(),
        bi_lstm_model(),
        bert_model(),
    ]

    results = []
    label_encoder = LabelEncoder()

    for clf in classifiers:
        train_accuracies = []
        test_accuracies = []

        for _ in tqdm(range(2), desc=f"Processing {getattr(clf, 'name', clf.__class__.__name__)}", unit="iteration"):
            train_data, test_data, train_labels, test_labels = load_data("undersampled")

            if isinstance(clf, Sequential):
                train_data = preprocess_text_tensorflow(train_data)
                test_data = preprocess_text_tensorflow(test_data)
                train_labels = label_encoder.fit_transform(train_labels)
                train_labels = to_categorical(y=train_labels, num_classes=3)
                test_labels = label_encoder.fit_transform(test_labels)
                test_labels = to_categorical(y=test_labels, num_classes=3)
                clf.fit(train_data, train_labels, verbose=0)
                train_predictions = clf.predict(train_data, verbose=0)
                train_accuracy = accuracy_score(train_labels.argmax(axis=1), np.argmax(train_predictions, axis=1))
                train_accuracies.append(train_accuracy)
                test_predictions = clf.predict(test_data, verbose=0)
                test_accuracy = accuracy_score(test_labels.argmax(axis=1), np.argmax(test_predictions, axis=1))
                test_accuracies.append(test_accuracy)
            elif isinstance(clf, Model):
                train_input_ids, train_attention_mask = preprocess_text_bert(train_data)
                test_input_ids, test_attention_mask = preprocess_text_bert(test_data)
                train_labels = label_encoder.fit_transform(train_labels)
                train_labels = to_categorical(y=train_labels, num_classes=3)
                test_labels = label_encoder.fit_transform(test_labels)
                test_labels = to_categorical(y=test_labels, num_classes=3)
                train_input_ids = np.squeeze(train_input_ids, axis=1)
                train_attention_mask = np.squeeze(train_attention_mask, axis=1)
                test_input_ids = np.squeeze(test_input_ids, axis=1)
                test_attention_mask = np.squeeze(test_attention_mask, axis=1)
                clf.fit([train_input_ids, train_attention_mask], train_labels, verbose=0)
                train_predictions = clf.predict([train_input_ids, train_attention_mask], verbose=0)
                train_accuracy = accuracy_score(train_labels.argmax(axis=1), np.argmax(train_predictions, axis=1))
                train_accuracies.append(train_accuracy)
                test_predictions = clf.predict([test_input_ids, test_attention_mask], verbose=0)
                test_accuracy = accuracy_score(test_labels.argmax(axis=1), np.argmax(test_predictions, axis=1))
                test_accuracies.append(test_accuracy)
            else:
                train_data = preprocess_text_sklearn(train_data)
                test_data = preprocess_text_sklearn(test_data)
                clf.fit(train_data, train_labels)
                train_predictions = clf.predict(train_data)
                train_accuracy = accuracy_score(train_labels, train_predictions)
                train_accuracies.append(train_accuracy)
                test_predictions = clf.predict(test_data)
                test_accuracy = accuracy_score(test_labels, test_predictions)
                test_accuracies.append(test_accuracy)

        train_min, train_max = np.min(train_accuracies), np.max(train_accuracies)
        test_min, test_max = np.min(test_accuracies), np.max(test_accuracies)
        train_range, test_range = (train_max - train_min) / 2, (test_max - test_min) / 2

        results.append([
            getattr(clf, 'name', clf.__class__.__name__),
            np.mean(train_accuracies).round(4),
            round(train_range, 4),
            np.mean(test_accuracies).round(4),
            round(test_range, 4)
        ])

    headers = ["Classifier", "Train Acc x̄", "Train Acc ±", "Test Acc x̄", "Test Acc ±"]
    print("\n", tabulate(results, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    main()

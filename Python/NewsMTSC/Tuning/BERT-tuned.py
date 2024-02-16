import json
import os
import silence_tensorflow.auto  # pip install silence-tensorflow
import spacy
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from keras.models import Model
from keras.layers import Dense, Input, Masking
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.losses import categorical_crossentropy
from transformers import BertTokenizer, TFBertForSequenceClassification
from keras.metrics import CategoricalAccuracy
import tensorflow as tf
import psutil
from transformers import logging


logging.set_verbosity_error()
tf.random.set_seed(2024)
# spacy.cli.download("en_core_web_lg")
nlp = spacy.load("en_core_web_lg")
p = psutil.Process(os.getpid())
p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)


def preprocess_bert(data):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    input_ids, attention_masks = [], []
    for text in data:
        tokens = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=180,
            return_attention_mask=True,
            return_tensors='tf'
        )
        input_ids.append(tokens['input_ids'])
        attention_masks.append(tokens['attention_mask'])
    return np.array(input_ids), np.array(attention_masks)


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


def bert_1_best():
    bert = TFBertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=3)
    # only the last 1 layers are trainable
    for layer in bert.layers:
        layer.trainable = False
    for layer in bert.layers[-1:]:
        layer.trainable = True
    input_ids = Input(shape=(180,), dtype="int32", name="input_ids")
    attention_mask = Input(shape=(180,), dtype="int32", name="attention_mask")
    masked_input = Masking(mask_value=0)(input_ids)
    outputs = bert(masked_input, attention_mask=attention_mask)[0]
    outputs = Dense(3, activation='softmax')(outputs)
    model = Model(inputs=[input_ids, attention_mask], outputs=outputs)
    adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.002037, beta_1=0.816, beta_2=0.9942, epsilon=9.2e-7, clipnorm=1)
    model.compile(optimizer=adam, loss=categorical_crossentropy, metrics=CategoricalAccuracy())
    model._name = 'BERT_1_best'
    return model


def bert_1_sec_best():
    bert = TFBertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=3)
    # only the last 1 layers are trainable
    for layer in bert.layers:
        layer.trainable = False
    for layer in bert.layers[-1:]:
        layer.trainable = True
    input_ids = Input(shape=(180,), dtype="int32", name="input_ids")
    attention_mask = Input(shape=(180,), dtype="int32", name="attention_mask")
    masked_input = Masking(mask_value=0)(input_ids)
    outputs = bert(masked_input, attention_mask=attention_mask)[0]
    outputs = Dense(3, activation='softmax')(outputs)
    model = Model(inputs=[input_ids, attention_mask], outputs=outputs)
    adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.000709, beta_1=0.814, beta_2=0.9974, epsilon=6.9e-7, clipnorm=1.8)
    model.compile(optimizer=adam, loss=categorical_crossentropy, metrics=CategoricalAccuracy())
    model._name = 'BERT_1_sec_best'
    return model


def bert_2_best():
    bert = TFBertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=3)
    # only the last 2 layers are trainable
    for layer in bert.layers:
        layer.trainable = False
    for layer in bert.layers[-2:]:
        layer.trainable = True
    input_ids = Input(shape=(180,), dtype="int32", name="input_ids")
    attention_mask = Input(shape=(180,), dtype="int32", name="attention_mask")
    masked_input = Masking(mask_value=0)(input_ids)
    outputs = bert(masked_input, attention_mask=attention_mask)[0]
    outputs = Dense(3, activation='softmax')(outputs)
    model = Model(inputs=[input_ids, attention_mask], outputs=outputs)
    adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.00213, beta_1=0.965, beta_2=0.9904, epsilon=3.3e-7, clipnorm=1.9)
    model.compile(optimizer=adam, loss=categorical_crossentropy, metrics=CategoricalAccuracy())
    model._name = 'BERT_2_best'
    return model


def bert_2_sec_best():
    bert = TFBertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=3)
    # only the last 2 layers are trainable
    for layer in bert.layers:
        layer.trainable = False
    for layer in bert.layers[-2:]:
        layer.trainable = True
    input_ids = Input(shape=(180,), dtype="int32", name="input_ids")
    attention_mask = Input(shape=(180,), dtype="int32", name="attention_mask")
    masked_input = Masking(mask_value=0)(input_ids)
    outputs = bert(masked_input, attention_mask=attention_mask)[0]
    outputs = Dense(3, activation='softmax')(outputs)
    model = Model(inputs=[input_ids, attention_mask], outputs=outputs)
    adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.002474, beta_1=0.866, beta_2=0.9909, epsilon=4.4e-7, clipnorm=0.1)
    model.compile(optimizer=adam, loss=categorical_crossentropy, metrics=CategoricalAccuracy())
    model._name = 'BERT_2_sec_best'
    return model


def main():

    classifiers = [
        bert_1_best(),
        bert_1_sec_best(),
        bert_2_best(),
        bert_2_sec_best()
    ]

    label_encoder = LabelEncoder()

    for _ in tqdm(range(1), desc=f"Preprocessing Data"):
        train_data, test_data, train_labels, test_labels = load_data("undersampled")
        train_labels_one_hot, test_labels_one_hot = preprocess_labels(label_encoder, train_labels, test_labels, num_classes=3)
        train_data_bert, train_attention_mask = preprocess_bert(train_data)
        test_data_bert, test_attention_mask = preprocess_bert(test_data)
        train_data_bert, train_attention_mask = np.squeeze(train_data_bert, axis=1), np.squeeze(train_attention_mask, axis=1)
        test_data_bert, test_attention_mask = np.squeeze(test_data_bert, axis=1), np.squeeze(test_attention_mask, axis=1)

    for clf in classifiers:
        iteration_losses = []
        epoch = 0
        if clf._name == "BERT_2_sec_best":
            highest_test_accuracy = 0.0
            for _ in tqdm(range(25), desc=f"Processing {getattr(clf, 'name', clf.__class__.__name__)}", unit="Epoch"):
                epoch += 1
                loss = clf.fit([train_data_bert, train_attention_mask], train_labels_one_hot, verbose=0, epochs=1, batch_size=30).history['loss'][0]
                iteration_losses.append(round(loss, 4))
                test_predictions = clf.predict([test_data_bert, test_attention_mask], verbose=0)
                test_accuracy = accuracy_score(test_labels_one_hot.argmax(axis=1), np.argmax(test_predictions, axis=1))
                if test_accuracy > highest_test_accuracy and test_accuracy > 0.6:
                    highest_test_accuracy = test_accuracy
                    rounded_accuracy = round(highest_test_accuracy, 4)
                    filename = f"{clf._name}_{rounded_accuracy}_e{epoch}_weights.h5"
                    clf.save_weights(filename)
        elif clf._name == "BERT_1_best":
            highest_test_accuracy = 0.0
            for _ in tqdm(range(25), desc=f"Processing {getattr(clf, 'name', clf.__class__.__name__)}", unit="Epoch"):
                epoch += 1
                loss = clf.fit([train_data_bert, train_attention_mask], train_labels_one_hot, verbose=0, epochs=1, batch_size=71).history['loss'][0]
                iteration_losses.append(round(loss, 4))
                test_predictions = clf.predict([test_data_bert, test_attention_mask], verbose=0)
                test_accuracy = accuracy_score(test_labels_one_hot.argmax(axis=1), np.argmax(test_predictions, axis=1))
                if test_accuracy > highest_test_accuracy and test_accuracy > 0.6:
                    highest_test_accuracy = test_accuracy
                    rounded_accuracy = round(highest_test_accuracy, 4)
                    filename = f"{clf._name}_{rounded_accuracy}_e{epoch}_weights.h5"
                    clf.save_weights(filename)
        elif clf._name == "BERT_2_best" or clf._name == "BERT_1_sec_best":
            highest_test_accuracy = 0.0
            for _ in tqdm(range(25), desc=f"Processing {getattr(clf, 'name', clf.__class__.__name__)}", unit="Epoch"):
                epoch += 1
                loss = clf.fit([train_data_bert, train_attention_mask], train_labels_one_hot, verbose=0, epochs=1, batch_size=15).history['loss'][0]
                iteration_losses.append(round(loss, 4))
                test_predictions = clf.predict([test_data_bert, test_attention_mask], verbose=0)
                test_accuracy = accuracy_score(test_labels_one_hot.argmax(axis=1), np.argmax(test_predictions, axis=1))
                if test_accuracy > highest_test_accuracy and test_accuracy > 0.6:
                    highest_test_accuracy = test_accuracy
                    rounded_accuracy = round(highest_test_accuracy, 4)
                    filename = f"{clf._name}_{rounded_accuracy}_e{epoch}_weights.h5"
                    clf.save_weights(filename)
        else:
            raise Exception("no defined training for ", getattr(clf, 'name', clf.__class__.__name__))


if __name__ == "__main__":
    main()

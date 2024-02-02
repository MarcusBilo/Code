import json
import os
import keras_tuner
import silence_tensorflow.auto  # pip install silence-tensorflow
import spacy
import numpy as np
from keras.callbacks import EarlyStopping
from sklearn.utils import resample
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import psutil
from keras_tuner.tuners import RandomSearch  # pip install keras-tuner
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Masking, Input
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import logging
from tensorflow_addons.optimizers import AdamW  # pip install tensorflow-addons==0.18.0


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
            max_length=100,
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


class HyperModel(keras_tuner.HyperModel):

    def build(self, hp):
        bert = TFBertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=3)

        n_layers = (hp.Int('trainable_layers', min_value=1, max_value=4, step=1))
        for layer in bert.layers:
            layer.trainable = False
        for layer in bert.layers[-n_layers:]:
            layer.trainable = True

        input_ids = Input(shape=(100,), dtype="int32", name="input_ids")
        attention_mask = Input(shape=(100,), dtype="int32", name="attention_mask")
        masked_input = Masking(mask_value=0)(input_ids)
        outputs = bert(masked_input, attention_mask=attention_mask)[0]
        outputs = Dropout(rate=hp.Float('dropout_0', min_value=0.0, max_value=0.5, step=0.05))(outputs)

        # dense_activation = hp.Choice('dense_activation', values=['linear', 'relu', 'hard_sigmoid'])
        # outputs = Dense(hp.Int('dense_units_1', min_value=32, max_value=1024, step=32), activation=dense_activation)(outputs)
        # outputs = Dropout(rate=hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1))(outputs)
        # outputs = Dense(hp.Int('dense_units_2', min_value=32, max_value=1024, step=32), activation=dense_activation)(outputs)
        # outputs = Dropout(rate=hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1))(outputs)

        outputs = Dense(3, activation='softmax')(outputs)
        model = Model(inputs=[input_ids, attention_mask], outputs=outputs)
        optimizer = AdamW(
            learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, step=1e-5),
            weight_decay=hp.Float('weight_decay ', min_value=1e-5, max_value=1e-3, step=1e-5)
        )
        model.compile(optimizer=optimizer, loss=CategoricalCrossentropy(), metrics=[CategoricalAccuracy()])
        model._name = 'BERT'
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", [30, 71, 142]),
            **kwargs,
        )


def main():
    label_encoder = LabelEncoder()
    train_data, _, train_labels, _ = load_data("undersampled")
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_data, train_labels, test_size=2130, random_state=2024, stratify=train_labels
    )
    train_data_bert, train_attention_mask = preprocess_bert(train_data)
    val_data_bert, val_attention_mask = preprocess_bert(val_data)
    train_data_bert, train_attention_mask = np.squeeze(train_data_bert, axis=1), np.squeeze(train_attention_mask, axis=1)
    val_data_bert, val_attention_mask = np.squeeze(val_data_bert, axis=1), np.squeeze(val_attention_mask, axis=1)
    train_labels_one_hot, val_labels_one_hot = preprocess_labels(label_encoder, train_labels, val_labels, num_classes=3)
    tuner = RandomSearch(
        HyperModel(),
        objective='val_categorical_accuracy',
        max_trials=10,
        directory=r'D:\Ablage\PycharmProjects\bert_tuning_dir',
        project_name='bert_tuning'
    )
    epochs = 25
    early_stopping = EarlyStopping(monitor='val_categorical_accuracy', patience=5)
    tuner.search(
        [train_data_bert, train_attention_mask], train_labels_one_hot, epochs=epochs,
        validation_data=([val_data_bert, val_attention_mask], val_labels_one_hot), verbose=1, callbacks=[early_stopping]
    )
    best_trials = tuner.oracle.get_best_trials(num_trials=2)
    combined_results = {'Training Accuracy': [], 'Validation Accuracy': []}
    for trial in best_trials:
        hyperparameters_dict = {
            key: round(value, 7) if isinstance(value, (int, float)) else value for key, value in trial.hyperparameters.values.items()
        }
        training_accuracy = round(trial.metrics.get_last_value('categorical_accuracy'), 5)
        validation_accuracy = round(trial.metrics.get_last_value('val_categorical_accuracy'), 5)
        for key, value in hyperparameters_dict.items():
            combined_results.setdefault(key, []).append(value)
        combined_results['Training Accuracy'].append(training_accuracy)
        combined_results['Validation Accuracy'].append(validation_accuracy)
    table_data = [[key, values[0], values[1]] for key, values in combined_results.items()]
    headers = ["Parameters", "Best", "2nd-Best"]
    print("\n", epochs, "Epochs", "\n", tabulate(table_data, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    main()

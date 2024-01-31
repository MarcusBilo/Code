import json
import os
import silence_tensorflow.auto  # pip install silence-tensorflow
import spacy
import numpy as np
from tabulate import tabulate
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn.svm import SVC
from keras.models import Sequential, Model
from keras.layers import Conv1D, Dense, Flatten, LSTM, Bidirectional, Input, Masking, Dropout, GRU
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.losses import categorical_crossentropy
from transformers import BertTokenizer, TFBertForSequenceClassification
from keras.metrics import CategoricalAccuracy
from keras.callbacks import EarlyStopping
import tensorflow as tf
import psutil
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # pip install vaderSentiment


adam = tf.keras.optimizers.legacy.Adam(clipvalue=1.0)
spacy.cli.download("en_core_web_lg")
nlp = spacy.load("en_core_web_lg")
# p = psutil.Process(os.getpid())
# p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)


def preprocess_sklearn(data):
    processed_data = []
    for text in data:
        lemmatized_text = ' '.join([token.lemma_ for token in nlp(text) if not token.is_stop])
        doc_lemmatized = nlp(lemmatized_text)
        doc_vector = doc_lemmatized.vector
        processed_data.append(doc_vector)
    processed_data = np.array(processed_data)
    return processed_data


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


def preprocess_bert(data):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    input_ids, attention_masks = [], []
    for text in data:
        lemmatized_text = ' '.join([token.lemma_ for token in nlp(text) if not token.is_stop])
        tokens = tokenizer.encode_plus(
            lemmatized_text,
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


class VADER:

    def predict(self, x):
        predictions = [self.analyze_sentiment(sentence) for sentence in x]
        return predictions

    @staticmethod
    def analyze_sentiment(sentence):
        analyzer = SentimentIntensityAnalyzer()
        vs = analyzer.polarity_scores(sentence)
        return vs['compound']

    # https://github.com/cjhutto/vaderSentiment?tab=readme-ov-file#about-the-scoring
    @staticmethod
    def classify_sentiment(compound_score):
        if compound_score > 0.05:
            return 'positive'
        elif compound_score < -0.05:
            return 'negative'
        else:
            return 'neutral'


def cnn_model():
    model = Sequential()
    model.add(Masking(mask_value=0))
    model.add(Conv1D(768, 5))
    model.add(Conv1D(768, 5))
    model.add(Dropout(0.5))
    model.add(Dense(768, activation="linear"))
    # model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(3, activation="softmax"))
    model.compile(optimizer=adam, loss=categorical_crossentropy, metrics=CategoricalAccuracy())
    model._name = "CNN"
    return model


def gru_model():
    model = Sequential()
    model.add(Masking(mask_value=0))
    model.add(GRU(units=384, return_sequences=True))
    model.add(GRU(units=384))
    model.add(Dropout(0.5))
    model.add(Dense(384, activation="linear"))
    # model.add(Dropout(0.5))
    model.add(Dense(3, activation="softmax"))
    model.compile(optimizer=adam, loss=categorical_crossentropy, metrics=CategoricalAccuracy())
    model._name = "GRU"
    return model


def bi_gru_model():
    model = Sequential()
    model.add(Masking(mask_value=0))
    model.add(Bidirectional(GRU(units=192, return_sequences=True)))
    model.add(Bidirectional(GRU(units=192)))
    model.add(Dropout(0.5))
    model.add(Dense(192, activation="linear"))
    # model.add(Dropout(0.5))
    model.add(Dense(3, activation="softmax"))
    model.compile(optimizer=adam, loss=categorical_crossentropy, metrics=CategoricalAccuracy())
    model._name = "Bi-GRU"
    return model


def lstm_model():
    model = Sequential()
    model.add(Masking(mask_value=0))
    model.add(LSTM(units=384, return_sequences=True))
    model.add(LSTM(units=384))
    model.add(Dropout(0.5))
    model.add(Dense(384, activation="linear"))
    # model.add(Dropout(0.5))
    model.add(Dense(3, activation="softmax"))
    model.compile(optimizer=adam, loss=categorical_crossentropy, metrics=CategoricalAccuracy())
    model._name = "LSTM"
    return model


def bi_lstm_model():
    model = Sequential()
    model.add(Masking(mask_value=0))
    model.add(Bidirectional(LSTM(units=192, return_sequences=True)))
    model.add(Bidirectional(LSTM(units=192)))
    model.add(Dropout(0.5))
    model.add(Dense(192, activation="linear"))
    # model.add(Dropout(0.5))
    model.add(Dense(3, activation="softmax"))
    model.compile(optimizer=adam, loss=categorical_crossentropy, metrics=CategoricalAccuracy())
    model._name = "Bi-LSTM"
    return model


def bert_model_1_128():
    bert = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    # only the last 1 layers are trainable
    for layer in bert.layers:
        layer.trainable = False
    for layer in bert.layers[-1:]:
        layer.trainable = True
    input_ids = Input(shape=(100,), dtype="int32", name="input_ids")
    attention_mask = Input(shape=(100,), dtype="int32", name="attention_mask")
    masked_input = Masking(mask_value=0)(input_ids)
    outputs = bert(masked_input, attention_mask=attention_mask)[0]
    outputs = Dense(128, activation="linear")(outputs)
    outputs = Dropout(0.5)(outputs)
    outputs = Dense(3, activation='softmax')(outputs)
    model = Model(inputs=[input_ids, attention_mask], outputs=outputs)
    model.compile(optimizer=adam, loss=categorical_crossentropy, metrics=CategoricalAccuracy())
    model._name = 'BERT_1_128'
    return model


def bert_model_2_256():
    bert = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    # only the last 2 layers are trainable
    for layer in bert.layers:
        layer.trainable = False
    for layer in bert.layers[-2:]:
        layer.trainable = True
    input_ids = Input(shape=(100,), dtype="int32", name="input_ids")
    attention_mask = Input(shape=(100,), dtype="int32", name="attention_mask")
    masked_input = Masking(mask_value=0)(input_ids)
    outputs = bert(masked_input, attention_mask=attention_mask)[0]
    outputs = Dense(256, activation="linear")(outputs)
    outputs = Dropout(0.5)(outputs)
    outputs = Dense(3, activation='softmax')(outputs)
    model = Model(inputs=[input_ids, attention_mask], outputs=outputs)
    model.compile(optimizer=adam, loss=categorical_crossentropy, metrics=CategoricalAccuracy())
    model._name = 'BERT_2_256'
    return model


def main():

    classifiers = [
        VADER(),
        SVC(),
        cnn_model(),
        gru_model(),
        bi_gru_model(),
        lstm_model(),
        bi_lstm_model(),
        bert_model_1_128(),
        bert_model_2_256(),
    ]

    results = []
    train_accuracy, test_accuracy = 0.0, 0.0
    label_encoder = LabelEncoder()
    callback = EarlyStopping(monitor='val_loss', patience=3, verbose=0, restore_best_weights=True)

    for clf in classifiers:
        iteration_losses = []
        train_data, test_data, train_labels, test_labels = load_data("undersampled")

        if isinstance(clf, Sequential):
            train_data, test_data = preprocess_tensorflow(train_data), preprocess_tensorflow(test_data)
            train_labels, test_labels = preprocess_labels(label_encoder, train_labels, test_labels, num_classes=3)
            for _ in tqdm(range(10), desc=f"Processing {getattr(clf, 'name', clf.__class__.__name__)}", unit="Epoch"):
                loss = clf.fit(train_data, train_labels, verbose=0, batch_size=71, callbacks=[callback]).history['loss'][0]
                iteration_losses.append(round(loss, 4))
                train_predictions = clf.predict(train_data, verbose=0)
                train_accuracy = accuracy_score(train_labels.argmax(axis=1), np.argmax(train_predictions, axis=1))
                test_predictions = clf.predict(test_data, verbose=0)
                test_accuracy = accuracy_score(test_labels.argmax(axis=1), np.argmax(test_predictions, axis=1))
        elif isinstance(clf, Model):
            train_input_ids, train_attention_mask = preprocess_bert(train_data)
            test_input_ids, test_attention_mask = preprocess_bert(test_data)
            train_labels, test_labels = preprocess_labels(label_encoder, train_labels, test_labels, num_classes=3)
            train_input_ids, train_attention_mask = np.squeeze(train_input_ids, axis=1), np.squeeze(train_attention_mask, axis=1)
            test_input_ids, test_attention_mask = np.squeeze(test_input_ids, axis=1), np.squeeze(test_attention_mask, axis=1)
            for _ in tqdm(range(10), desc=f"Processing {getattr(clf, 'name', clf.__class__.__name__)}", unit="Epoch"):
                loss = clf.fit([train_input_ids, train_attention_mask], train_labels, verbose=0, batch_size=10, callbacks=[callback]).history['loss'][0]
                iteration_losses.append(round(loss, 4))
                train_predictions = clf.predict([train_input_ids, train_attention_mask], verbose=0)
                train_accuracy = accuracy_score(train_labels.argmax(axis=1), np.argmax(train_predictions, axis=1))
                test_predictions = clf.predict([test_input_ids, test_attention_mask], verbose=0)
                test_accuracy = accuracy_score(test_labels.argmax(axis=1), np.argmax(test_predictions, axis=1))
        elif isinstance(clf, VADER):
            for _ in tqdm(range(1), desc=f"Processing {getattr(clf, 'name', clf.__class__.__name__)}", unit="Epoch"):
                train_predictions = clf.predict(train_data)
                train_sentiment_labels = [VADER.classify_sentiment(pred) for pred in train_predictions]
                train_accuracy = accuracy_score(train_labels, train_sentiment_labels)
                test_predictions = clf.predict(test_data)
                test_sentiment_labels = [VADER.classify_sentiment(pred) for pred in test_predictions]
                test_accuracy = accuracy_score(test_labels, test_sentiment_labels)
                iteration_losses.append("---")
        elif isinstance(clf, SVC):
            train_data, test_data = preprocess_sklearn(train_data), preprocess_sklearn(test_data)
            for _ in tqdm(range(1), desc=f"Processing {getattr(clf, 'name', clf.__class__.__name__)}", unit="Epoch"):
                clf.fit(train_data, train_labels)
                train_predictions = clf.predict(train_data)
                train_accuracy = accuracy_score(train_labels, train_predictions)
                test_predictions = clf.predict(test_data)
                test_accuracy = accuracy_score(test_labels, test_predictions)
                iteration_losses.append("---")

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

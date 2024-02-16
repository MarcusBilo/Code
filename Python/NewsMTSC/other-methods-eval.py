import json
import os
import silence_tensorflow.auto  # pip install silence-tensorflow
import spacy
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import resample
from sklearn.svm import SVC
from keras.models import Model
from keras.layers import Dense, Input, Masking
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.losses import categorical_crossentropy
from transformers import BertTokenizer, TFBertForSequenceClassification
from keras.metrics import CategoricalAccuracy
import tensorflow as tf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # pip install vaderSentiment
from transformers import logging


logging.set_verbosity_error()
tf.random.set_seed(2024)
# spacy.cli.download("en_core_web_lg")
nlp = spacy.load("en_core_web_lg")


def preprocess_sklearn(data):
    """
    This function takes a list of text data and performs preprocessing using spaCy for lemmatization and
    stop-word removal. It then converts the processed text into numerical vectors using spaCy's word vectors.

    Parameters:
    - data (list): A list containing textual data to be preprocessed.

    Returns:
    - processed_data (numpy.ndarray): An array of numerical vectors representing the preprocessed text data.
    """
    processed_data = []
    for text in data:
        lemmatized_text = ' '.join([token.lemma_ for token in nlp(text) if not token.is_stop])
        doc_lemmatized = nlp(lemmatized_text)
        doc_vector = doc_lemmatized.vector
        processed_data.append(doc_vector)
    processed_data = np.array(processed_data)
    return processed_data


def preprocess_bert(data):
    """
    This function utilizes the BERT (Bidirectional Encoder Representations from Transformers) tokenizer to
    preprocess a list of textual data by converting it into input IDs and attention masks suitable for BERT-based models.

    Parameters:
    - data (list): A list containing textual data to be preprocessed.

    Returns:
    - input_ids (numpy.ndarray): An array of input IDs representing the preprocessed textual data.
    - attention_masks (numpy.ndarray): An array of attention masks corresponding to the input IDs.
    """
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


class VADER:
    """
    This class provides a wrapper for sentiment analysis using the VADER (Valence Aware Dictionary and sEntiment Reasoner)
    sentiment analysis tool.

    Attributes:
    - None

    Methods:
    - predict(x): Predicts sentiment labels for a list of sentences using VADER.
    - analyze_sentiment(sentence): Analyzes the sentiment of a single sentence using VADER.
    - classify_sentiment(compound_score): Classifies sentiment based on the compound score.

    Reference:
    - VADER Sentiment Analysis: https://github.com/cjhutto/vaderSentiment
    """
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


def bert_2():
    """
    This function defines a BERT-based sequence classification model with specific configurations.
    The model is setup for a sentiment analysis task with three output classes ('positive', 'neutral', 'negative').

    Returns:
    - model (tf.keras.Model): A BERT-based sentiment analysis model with trainable last two layers.
    """
    bert = TFBertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=3)
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
    """
    This is the main function that demonstrates the usage of different classifiers (VADER, SVC, BERT)
    on a sentiment analysis task. It loads and preprocesses data, applies various classifiers,
    and prints confusion matrices for each classifier on both the training and testing sets.
    """
    classifiers = [
        VADER(),
        SVC(C=0.96),
        bert_2()
    ]

    label_encoder = LabelEncoder()
    train_data, test_data, train_labels, test_labels = load_data("undersampled")
    train_data_sklearn, test_data_sklearn = preprocess_sklearn(train_data), preprocess_sklearn(test_data)
    train_labels_one_hot, test_labels_one_hot = preprocess_labels(label_encoder, train_labels, test_labels, num_classes=3)
    train_data_bert, train_attention_mask = preprocess_bert(train_data)
    test_data_bert, test_attention_mask = preprocess_bert(test_data)
    train_data_bert, train_attention_mask = np.squeeze(train_data_bert, axis=1), np.squeeze(train_attention_mask, axis=1)
    test_data_bert, test_attention_mask = np.squeeze(test_data_bert, axis=1), np.squeeze(test_attention_mask, axis=1)

    for clf in classifiers:
        if isinstance(clf, VADER):
            train_predictions = clf.predict(train_data)
            train_sentiment_labels = [VADER.classify_sentiment(pred) for pred in train_predictions]
            train_accuracy = accuracy_score(train_labels, train_sentiment_labels)
            test_predictions = clf.predict(test_data)
            test_sentiment_labels = [VADER.classify_sentiment(pred) for pred in test_predictions]
            test_accuracy = accuracy_score(test_labels, test_sentiment_labels)
            train_conf_matrix = confusion_matrix(train_labels, train_sentiment_labels)
            test_conf_matrix = confusion_matrix(test_labels, test_sentiment_labels)
            print("VADER", round(train_accuracy, 4), "\n", train_conf_matrix)
            print("VADER", round(test_accuracy, 4), "\n", test_conf_matrix, "\n")
        elif isinstance(clf, SVC):
            clf.fit(train_data_sklearn, train_labels)
            train_predictions = clf.predict(train_data_sklearn)
            train_accuracy = accuracy_score(train_labels, train_predictions)
            test_predictions = clf.predict(test_data_sklearn)
            test_accuracy = accuracy_score(test_labels, test_predictions)
            train_conf_matrix = confusion_matrix(train_labels, train_predictions)
            test_conf_matrix = confusion_matrix(test_labels, test_predictions)
            print("SVC", round(train_accuracy, 4), "\n", train_conf_matrix)
            print("SVC", round(test_accuracy, 4), "\n", test_conf_matrix, "\n")
        elif isinstance(clf, Model):
            clf.build(180)
            clf.load_weights("BERT_2_sec_best_0.6299_e20_weights.h5")
            train_predictions = clf.predict([train_data_bert, train_attention_mask], verbose=0)
            train_accuracy = accuracy_score(train_labels_one_hot.argmax(axis=1), np.argmax(train_predictions, axis=1))
            test_predictions = clf.predict([test_data_bert, test_attention_mask], verbose=0)
            test_accuracy = accuracy_score(test_labels_one_hot.argmax(axis=1), np.argmax(test_predictions, axis=1))
            train_conf_matrix = confusion_matrix(train_labels_one_hot.argmax(axis=1), np.argmax(train_predictions, axis=1))
            test_conf_matrix = confusion_matrix(test_labels_one_hot.argmax(axis=1), np.argmax(test_predictions, axis=1))
            print("BERT", round(train_accuracy, 4), "\n", train_conf_matrix)
            print("BERT", round(test_accuracy, 4), "\n", test_conf_matrix)


if __name__ == "__main__":
    main()

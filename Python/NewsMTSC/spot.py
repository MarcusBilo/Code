# python -m spacy download en_core_web_md

import json
import spacy
import warnings
import numpy as np
from tabulate import tabulate
from tqdm import tqdm
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore", category=ConvergenceWarning)
nlp = spacy.load("en_core_web_md")


def preprocess_text(data):
    processed_data = []
    for text in data:
        doc = nlp(text)
        doc_vector = doc.vector
        processed_data.append(doc_vector)
    return processed_data


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


def main():

    train_data, test_data, train_labels, test_labels = load_data("undersampled")
    train_data = preprocess_text(train_data)

    classifiers = [
        RandomForestClassifier(),
        RandomForestClassifier(),
        SVC(),
        HistGradientBoostingClassifier(),
        HistGradientBoostingClassifier(),
        MLPClassifier(),
        AdaBoostClassifier(),
        AdaBoostClassifier(),
    ]

    param_grid = [
        # For RandomForestClassifier
        {
            'n_estimators': [20, 40, 60, 80, 100],
        },
        # For RandomForestClassifier
        {
            'max_depth': [2, 4, 6, 8, 10],
        },
        # For SVC
        {
            'kernel': ["linear", "poly", "rbf", "sigmoid"],
        },
        # For HistGradientBoostingClassifier
        {
            'max_iter': [20, 40, 60, 80, 100],
        },
        # For HistGradientBoostingClassifier
        {
            'max_depth': [2, 4, 6, 8, 10],
        },
        # For MLPClassifier
        {
            'activation': ["identity", "logistic", "tanh", "relu"],
        },
        # For AdaBoostClassifier
        {
            'n_estimators': [10, 20, 30, 40, 50],
        },
        # For AdaBoostClassifier
        {
            'learning_rate': [0.5, 1.0, 1.5, 2.0, 2.5]
        },
    ]

    best_parameters = []

    for clf, param in tqdm(zip(classifiers, param_grid), total=len(classifiers), desc="Hyperparameter optimization"):
        search = GridSearchCV(clf, param, scoring="balanced_accuracy", cv=5)
        search.fit(train_data, train_labels)
        best_params = search.best_params_
        best_parameters.append((type(clf).__name__, best_params))

    for clf_name, params in best_parameters:
        print(f"Best parameters for {clf_name}: {params}")


if __name__ == "__main__":
    main()

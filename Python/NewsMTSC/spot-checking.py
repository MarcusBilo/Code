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
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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
    classifiers = [
        RandomForestClassifier(),
        SVC(),
        SGDClassifier(),
        KNeighborsClassifier(),
        DecisionTreeClassifier(),
        HistGradientBoostingClassifier(),
        MLPClassifier(),
        AdaBoostClassifier(),
        GaussianNB()
    ]

    results = []

    for clf in classifiers:
        train_accuracies = []
        test_accuracies = []

        for _ in tqdm(range(3), desc=f"Processing {clf.__class__.__name__}", unit="iteration"):
            train_data, test_data, train_labels, test_labels = load_data("undersampled")

            train_data = preprocess_text(train_data)
            test_data = preprocess_text(test_data)

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
            clf.__class__.__name__,
            np.mean(train_accuracies).round(4),
            round(train_range, 4),
            np.mean(test_accuracies).round(4),
            round(test_range, 4)
        ])

    headers = ["Classifier", "Train Acc x̄", "Train Acc ±", "Test Acc x̄", "Test Acc ±"]
    print("\n", tabulate(results, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    main()

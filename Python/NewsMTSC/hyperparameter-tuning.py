# !python -m spacy download en_core_web_md

import json
from itertools import product
import spacy
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
from sklearn.utils import resample
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

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
    test_data = preprocess_text(test_data)

    classifiers = [
        RandomForestClassifier(),
        SVC(),
        SGDClassifier(),
        HistGradientBoostingClassifier(),
        MLPClassifier(),
        AdaBoostClassifier(),
    ]

    param_grid = [
        {   # For RandomForestClassifier
            'n_estimators': [80, 90, 100],
            'max_depth': [8, 9, 10, None],
            'criterion': ["gini", "entropy", "log_loss"],
            'class_weight': ["balanced", None],
            'n_jobs': [4],
        },
        {   # For SVC
            'C': [0.8, 0.9, 1.0, 1.1],
            'degree': [2, 3, 4],
            'coef0': [0.0, 0.05, 0.1],
            'kernel': ["poly"],
            'gamma': ["auto"],
            'class_weight': ["balanced", None]
        },
        {   # For SGD
            'loss': ["hinge", "log_loss", "modified_huber", "squared_hinge",
                     "perceptron", "squared_error", "huber", "epsilon_insensitive",
                     "squared_epsilon_insensitive"],
            'penalty': ["l2", "l1", "elasticnet", None],
            'class_weight': ["balanced", None],
            'n_jobs': [4],
        },
        {   # For HistGradientBoostingClassifier
            'max_iter': [80, 90, 100],
            'max_depth': [8, 9, 10, None],
            'min_samples_leaf': [10, 15, 20],
            'class_weight': ["balanced", None]
        },
        {   # For MLPClassifier
            'activation': ["logistic"],
            'solver': ["adam"],
            'hidden_layer_sizes': [(50,), (75,), (100,)],
            'max_iter': [50, 100, 150, 200],
            'learning_rate_init': [0.001, 0.01]
        },
        {   # For AdaBoostClassifier
            'n_estimators': [70, 80, 90, 100],
            'learning_rate': [0.6, 0.7, 0.8, 0.9, 1.0]
        },
    ]

    best_parameters = []

    for clf, param in zip(classifiers, param_grid):
        clf_name = type(clf).__name__
        param_combinations = product(*param.values())

        best_accuracy = float("-inf")
        best_params = None

        for params in tqdm(param_combinations, desc=f"{clf_name}"):
            param_dict = dict(zip(param.keys(), params))
            clf.set_params(**param_dict)
            clf.fit(train_data, train_labels)
            predicted_labels = clf.predict(test_data)
            accuracy = balanced_accuracy_score(test_labels, predicted_labels)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = param_dict.copy()

        best_parameters.append((clf_name, best_params))

    for clf_name, params in best_parameters:
        print(f"Best parameters for {clf_name}: {params}")


if __name__ == "__main__":
    main()

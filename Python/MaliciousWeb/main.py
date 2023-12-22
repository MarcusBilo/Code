# pip install spacy scikit-learn joblib
# python -m spacy download en_core_web_md

import os
import spacy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import joblib
import pandas as pd
from sklearn.utils import resample
from os.path import exists

nlp = spacy.load("en_core_web_md")


def prepare_data(path):
    columns_to_keep = ["url", "content", "label"]

    if not exists(path + "\Good_Compact_Webpages_Classification_test_data.csv"):
        original_data_test = pd.read_csv(path + "\Webpages_Classification_test_data.csv")
        selected_columns_test = original_data_test[columns_to_keep]
        good_data_test = selected_columns_test[selected_columns_test["label"] == "good"]
        bad_data_test = selected_columns_test[selected_columns_test["label"] == "bad"]
        good_data_test.to_csv(path + "\Good_Compact_Webpages_Classification_test_data.csv", index=False)
        bad_data_test.to_csv(path + "\Bad_Compact_Webpages_Classification_test_data.csv", index=False)
        del original_data_test, selected_columns_test, good_data_test, bad_data_test

    if not exists(path + "\Good_Compact_Webpages_Classification_train_data.csv"):
        original_data_train = pd.read_csv(path + "\Webpages_Classification_train_data.csv")
        selected_columns_train = original_data_train[columns_to_keep]
        good_data_train = selected_columns_train[selected_columns_train["label"] == "good"]
        bad_data_train = selected_columns_train[selected_columns_train["label"] == "bad"]
        good_data_train.to_csv(path + "\Good_Compact_Webpages_Classification_train_data.csv", index=False)
        bad_data_train.to_csv(path + "\Bad_Compact_Webpages_Classification_train_data.csv", index=False)
        del original_data_train, selected_columns_train, good_data_train, bad_data_train

    good_train = pd.read_csv(path + "\Good_Compact_Webpages_Classification_train_data.csv")
    good_test = pd.read_csv(path + "\Good_Compact_Webpages_Classification_test_data.csv")
    bad_train = pd.read_csv(path + "\Bad_Compact_Webpages_Classification_train_data.csv")
    bad_test = pd.read_csv(path + "\Bad_Compact_Webpages_Classification_test_data.csv")

    undersampled_good_train = resample(good_train, replace=False, n_samples=len(bad_train))
    undersampled_good_test = resample(good_test, replace=False, n_samples=len(bad_test))
    del good_train, good_test

    train_data = pd.concat([undersampled_good_train, bad_train])
    test_data = pd.concat([undersampled_good_test, bad_test])
    del undersampled_good_train, undersampled_good_test, bad_train, bad_test

    # to prevent the model from potentially learning order-specific patterns
    train_data_shuffled = train_data.sample(frac=1)
    test_data_shuffled = test_data.sample(frac=1)
    del train_data, test_data

    return train_data_shuffled, test_data_shuffled


def preprocess_text(data):
    processed_data = []
    for text in tqdm(data, desc="Preprocessing"):
        doc = nlp(text)
        doc_vector = doc.vector
        processed_data.append(doc_vector)
    return processed_data


def load_prepped_train_data(filename="prepped_train_data.joblib"):
    if os.path.exists(filename):
        return joblib.load(filename)
    else:
        return None


def load_prepped_test_data(filename="prepped_test_data.joblib"):
    if os.path.exists(filename):
        return joblib.load(filename)
    else:
        return None


def load_data(data):
    texts = data["content"].tolist()
    labels = data["label"].tolist()
    return texts, labels


def main():
    path = "D:\Ablage\Dataset of Malicious and Benign Webpages"
    train_data, test_data = prepare_data(path)

    content_train, labels_train = load_data(train_data)
    content_test, labels_test = load_data(test_data)

    X_train = load_prepped_train_data()
    if X_train is None:
        X_train = preprocess_text(content_train)

    X_test = load_prepped_test_data()
    if X_test is None:
        X_test = preprocess_text(content_test)

    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(X_train, labels_train)

    predictions = clf.predict(X_test)
    accuracy = accuracy_score(labels_test, predictions)
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    main()

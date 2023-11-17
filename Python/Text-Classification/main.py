# pip install spacy scikit-learn joblib
# python -m spacy download en_core_web_md

import spacy
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import joblib
import os

nlp = spacy.load("en_core_web_md")


def preprocess_text(data):
    processed_data = []
    for text in tqdm(data, desc="Preprocessing"):
        doc = nlp(text)
        doc_vector = doc.vector
        processed_data.append(doc_vector)
    return processed_data


def load_preprocessed_data(filename="preprocessed_data.joblib"):
    if os.path.exists(filename):
        return joblib.load(filename)
    else:
        return None


def main():
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    X = load_preprocessed_data()
    if X is None:
        X = preprocess_text(newsgroups.data)
        joblib.dump(X, "preprocessed_data.joblib")
    y = newsgroups.target
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True)
    accuracy_scores = []

    for train_idx, test_idx in tqdm(stratified_kfold.split(X, y), desc="KFold", total=stratified_kfold.get_n_splits()):
        X_train, X_test = [X[i] for i in train_idx], [X[i] for i in test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        rf_classifier = RandomForestClassifier()
        rf_classifier.fit(X_train, y_train)
        predictions = rf_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        accuracy_scores.append(accuracy)

    avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
    print("\nAverage Accuracy: ", avg_accuracy, "\n")


if __name__ == "__main__":
    main()

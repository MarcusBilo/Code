# pip install spacy scikit-learn joblib
# python -m spacy download en_core_web_md

import spacy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import json

nlp = spacy.load("en_core_web_md")


def preprocess_text(data):
    processed_data = []
    for text in tqdm(data, desc="Preprocessing"):
        doc = nlp(text)
        doc_vector = doc.vector
        processed_data.append(doc_vector)
    return processed_data


def main():

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

    train_labels = [sentiment_map[p] for p in y_train]
    test_labels = [sentiment_map[p] for p in y_test]

    train_data = preprocess_text(train_data_normalized)
    test_data = preprocess_text(test_data_normalized)

    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(train_data, train_labels)

    predictions = clf.predict(test_data)
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    main()

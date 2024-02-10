import json
import silence_tensorflow.auto  # pip install silence-tensorflow
import numpy as np
from sklearn.utils import resample
import textstat  # pip install textstat


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


def main():
    train_data, test_data, train_labels, test_labels = load_data("original")

    flesch_scores = []
    for sentence in train_data:
        score = textstat.textstat.flesch_reading_ease(sentence)
        flesch_scores.append(score)
    threshold = np.percentile(flesch_scores, 15)
    top_10_percent_scores = [score for score in flesch_scores if score < threshold]
    print("Top 10% Flesch Reading Ease Scores:", min(top_10_percent_scores), max(top_10_percent_scores))

    flesch_scores = []
    for sentence in test_data:
        score = textstat.textstat.flesch_reading_ease(sentence)
        flesch_scores.append(score)
    threshold = np.percentile(flesch_scores, 15)
    top_10_percent_scores = [score for score in flesch_scores if score < threshold]
    print("Top 10% Flesch Reading Ease Scores:", min(top_10_percent_scores), max(top_10_percent_scores))


if __name__ == "__main__":
    main()

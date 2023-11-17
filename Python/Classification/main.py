from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from tabulate import tabulate


def main():
    iris = datasets.load_iris()
    features = iris.data
    labels = iris.target

    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True)
    predictions, true_labels, accuracy_scores = [], [], []

    for train_idx, test_idx in stratified_kfold.split(features, labels):
        clf = SVC(kernel="poly", C=1).fit(features[train_idx], labels[train_idx])
        predicted_labels = clf.predict(features[test_idx])
        predictions.extend(predicted_labels)
        true_labels.extend(labels[test_idx])
        accuracy_scores.append(accuracy_score(predicted_labels, labels[test_idx]))

    avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
    print("\nAverage Accuracy: ", avg_accuracy, "\n")

    conf_matrix = confusion_matrix(true_labels, predictions, labels=[0, 1, 2])
    class_names = iris.target_names

    table_data = []
    headers = [""] + [f"Pred {class_name}" for class_name in class_names]

    for i in range(len(class_names)):
        row = [f"True {class_names[i]}"] + [conf_matrix[i][j] for j in range(len(class_names))]
        table_data.append(row)

    table = tabulate(table_data, headers, tablefmt="grid")
    print(table)


if __name__ == "__main__":
    main()

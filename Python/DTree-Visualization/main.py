from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import dtreeviz  # https://github.com/parrt/dtreeviz


def main():
    iris = load_iris()
    X = iris.data
    y = iris.target

    clf = DecisionTreeClassifier(max_depth=4)
    clf.fit(X, y)

    viz_model = dtreeviz.model(clf,
                               X_train=X, y_train=y,
                               feature_names=iris.feature_names,
                               target_name='iris',
                               class_names=iris.target_names)

    v = viz_model.view()
    v.show()  # pop up window
    v.save("iris.svg")


if __name__ == "__main__":
    main()

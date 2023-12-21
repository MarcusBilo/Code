import pandas as pd
from sklearn.utils import resample
from os.path import exists


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

    return train_data, test_data


def main():
    path = "D:\Ablage\Dataset of Malicious and Benign Webpages"
    train_data, test_data = prepare_data(path)

    print(train_data.head())
    print(train_data.tail())
    print("\n", "_" * 75, "\n")
    print(test_data.head())
    print(test_data.tail())


if __name__ == "__main__":
    main()

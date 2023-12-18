import numpy as np
import pandas as pd
from os.path import exists

path = 'D:\Ablage\Dataset of Malicious and Benign Webpages'

if not exists(path+'\Good_Compact_Webpages_Classification_test_data.csv'):

    original_data_test = pd.read_csv(path+'\Webpages_Classification_test_data.csv')
    original_data_train = pd.read_csv(path+'\Webpages_Classification_train_data.csv')

    columns_to_keep = ['url', 'content', 'label']

    selected_columns_test = original_data_test[columns_to_keep]
    selected_columns_train = original_data_test[columns_to_keep]

    good_data_test = selected_columns_test[selected_columns_test['label'] == 'good'].copy()
    bad_data_test = selected_columns_test[selected_columns_test['label'] == 'bad'].copy()
    good_data_train = selected_columns_train[selected_columns_train['label'] == 'good'].copy()
    bad_data_train = selected_columns_train[selected_columns_train['label'] == 'bad'].copy()

    good_data_test.drop(columns='label', inplace=True)
    bad_data_test.drop(columns='label', inplace=True)
    good_data_train.drop(columns='label', inplace=True)
    bad_data_train.drop(columns='label', inplace=True)

    good_data_test.to_csv(path+'\Good_Compact_Webpages_Classification_test_data.csv', index=False)
    bad_data_test.to_csv(path+'\Bad_Compact_Webpages_Classification_test_data.csv', index=False)
    good_data_train.to_csv(path+'\Good_Compact_Webpages_Classification_train_data.csv', index=False)
    bad_data_train.to_csv(path+'\Bad_Compact_Webpages_Classification_train_data.csv', index=False)

og_good_train = pd.read_csv(path+"\Good_Compact_Webpages_Classification_train_data.csv")
og_good_test = pd.read_csv(path+"\Good_Compact_Webpages_Classification_test_data.csv")
bad_train = pd.read_csv(path+"\Bad_Compact_Webpages_Classification_train_data.csv")
bad_test = pd.read_csv(path+"\Bad_Compact_Webpages_Classification_test_data.csv")

print("OG Good train\n", og_good_train.shape)
print("OG Good test\n", og_good_test.shape)
print("Bad train\n", bad_train.shape)
print("Bad test\n", bad_test.shape)

if not exists(path+'\Good_Compact_Webpages_Classification_test_data_split_0.csv'):
    good_test_split = pd.read_csv(path+"\Good_Compact_Webpages_Classification_test_data.csv")
    groups = good_test_split.groupby(np.arange(len(good_test_split.index)) // 22117)  # 353872 mod 16 = 0
    for (frameno, frame) in groups:
        frame.to_csv(path+f"\Good_Compact_Webpages_Classification_test_data_split_%s.csv" % frameno, index=False)

if not exists(path+'\Good_Compact_Webpages_Classification_train_data_split_0.csv'):
    good_train_split = pd.read_csv(path+"\Good_Compact_Webpages_Classification_train_data.csv")
    good_train_split = good_train_split.iloc[:-11]  # Drop the last 11 rows
    groups = good_train_split.groupby(np.arange(len(good_train_split.index)) // 73296)  # 1172747-11 mod 16 = 0
    for (frameno, frame) in groups:
        frame.to_csv(path+f"\Good_Compact_Webpages_Classification_train_data_split_%s.csv" % frameno, index=False)

edit_good_train = pd.read_csv(path+"\Good_Compact_Webpages_Classification_train_data_split_0.csv")
edit_good_test = pd.read_csv(path+"\Good_Compact_Webpages_Classification_test_data_split_0.csv")
print("Edit Good train\n", edit_good_train.shape)
print("Edit Good test\n", edit_good_test.shape)

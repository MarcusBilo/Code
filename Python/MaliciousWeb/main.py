import pandas as pd

original_data = pd.read_csv('D:\Ablage\Dataset of Malicious and Benign Webpages\Webpages_Classification_test_data.csv')

columns_to_keep = ['url', 'content', 'label']

selected_columns_data = original_data[columns_to_keep]

good_data = selected_columns_data[selected_columns_data['label'] == 'good'].copy()
bad_data = selected_columns_data[selected_columns_data['label'] == 'bad'].copy()

good_data.drop(columns='label', inplace=True)
bad_data.drop(columns='label', inplace=True)

good_data.to_csv('D:\Ablage\Dataset of Malicious and Benign Webpages\Good_Compact_Webpages_Classification_test_data.csv', index=False)
bad_data.to_csv('D:\Ablage\Dataset of Malicious and Benign Webpages\Bad_Compact_Webpages_Classification_test_data.csv', index=False)

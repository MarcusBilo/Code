import os
import pandas as pd

data = pd.read_csv('when2heat.csv', delimiter=';')

country_abbreviations = ['AT', 'BE', 'BG', 'CZ', 'DE', 'FR', 'GB', 'HR', 'HU', 'IE', 'LU',
                         'NL', 'PL', 'RO', 'SI', 'SK']

timestamp_columns = ['utc_timestamp']

for abbreviation in country_abbreviations:
    matching_columns = [col for col in data.columns if col.startswith(f"{abbreviation}_")]
    if matching_columns:
        selected_columns = timestamp_columns + matching_columns
        country_data = data[selected_columns]
        directory = f"country_split"
        if not os.path.exists(directory):
            os.makedirs(directory)
        country_data.to_csv(f"{directory}/{abbreviation}_data.csv", index=False)
    else:
        print(f"No columns found for abbreviation '{abbreviation}'")

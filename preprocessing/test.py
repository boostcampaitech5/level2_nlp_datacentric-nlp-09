from data_cleaning import g2p_cleaning
import pandas as pd

data = pd.read_csv('../data/train_split_base.csv').iloc[:100]

cleaned_data = g2p_cleaning(data, find_g2p=True)
print(data[data.ID == "ynat-v1_train_07365"].text)
print(cleaned_data[cleaned_data.ID == "ynat-v1_train_07365"].text)
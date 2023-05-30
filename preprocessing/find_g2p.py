import pandas as pd
import os
from g2pk import G2p
from tqdm import tqdm
from sklearn.model_selection import train_test_split


BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '../data')

total_data = pd.read_csv(os.path.join(DATA_DIR, 'train_with_labels.csv'))
train_data, valid_data = train_test_split(total_data, test_size=0.3, random_state=456)

# train_data.to_csv(os.path.join(DATA_DIR, 'train_split_base.csv'), index=False)
# valid_data.to_csv(os.path.join(DATA_DIR, 'valid_split_base.csv'), index=False)

def find_g2p_suspicious(df):
    G2P_suspicious = pd.DataFrame()
    g2p = G2p()
    print("[Finding G2P Suspicious]")
    for idx, td in tqdm(total_data.iterrows(), total=len(total_data)):
        if g2p(td.text, descriptive=False) == td.text or g2p(td.text, descriptive=True) == td.text:
            td_df = pd.DataFrame(td).transpose()
            G2P_suspicious = pd.concat([G2P_suspicious, td_df], ignore_index=False)

    print("# of G2P Suspicious: {}".format(len(G2P_suspicious)))

    return G2P_suspicious

G2P_suspicious = find_g2p_suspicious(total_data)
G2P_suspicious.to_csv(os.path.join(DATA_DIR, 'g2p_suspicious.csv'), index=True)

G2P_ids = G2P_suspicious.ID

train_without_g2ps = train_data[~train_data.ID.isin(G2P_ids)]
train_without_g2ps.to_csv(os.path.join(DATA_DIR, 'train_without_g2p.csv'), index=False)
train_without_g2ps.text.to_csv(os.path.join(DATA_DIR, 'train_without_g2p.txt'), index=False, header=False)

valid_without_g2ps = valid_data[~valid_data.ID.isin(G2P_ids)]
valid_without_g2ps.to_csv(os.path.join(DATA_DIR, 'valid_without_g2p.csv'), index=False)
valid_without_g2ps.text.to_csv(os.path.join(DATA_DIR, 'valid_without_g2p.txt'), index=False, header=False)

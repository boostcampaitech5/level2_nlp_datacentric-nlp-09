import torch
import pandas as pd
from tqdm import tqdm
from roundtrip import backtranslate

def roundtrip(sentence):
    """
    roundtrip library를 이용해 backtranslate

    Args:
        sentence (str): backtranslate할 문장

    Returns:
        str: backtranslate 결과
    """
    translated_sen = backtranslate(phrase=sentence, from_language='ko')
    return translated_sen


def translate(train_path):
    data = pd.read_csv(train_path, encoding='utf-8')
    origin = data['text']
    translated_datas = [roundtrip(sen) for sen in tqdm(origin)]
    data['text'] = translated_datas
    return data

    
if __name__ == '__main__':
    version='v0.7.2'
    train_path = "../data/train_v0.12.1.csv"
    save_path = f"../data/train_{version}.csv"
    new_data = translate(train_path=train_path)
    new_data.to_csv(save_path, encoding='utf-8', index=False)

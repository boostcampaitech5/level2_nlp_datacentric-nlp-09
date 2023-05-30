import os
import re
import torch
import pandas as pd
from g2pk import G2p
from jamo import h2j, j2hcj
from tqdm import tqdm
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer


def temp_cleaning(df):
    """
        docstring 예시

        Args:
            df (DataType): 예시

        Returns:
            df (DataType): 예시
    """
    return df


def g2p_cleaning(df):
    BASE_DIR = os.getcwd()
    MODEL_DIR = os.path.join(BASE_DIR, '../p2g_model')

    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("DEVICE:", DEVICE)

    def find_g2p_suspicious(df):
        G2P_suspicious = pd.DataFrame()
        g2p = G2p()
        print("[Finding G2P Suspicious]")
        for idx, td in tqdm(df.iterrows(), total=len(df)):
            if g2p(td.text, descriptive=False) == td.text or g2p(td.text, descriptive=True) == td.text:
                td_df = pd.DataFrame(td).transpose()
                G2P_suspicious = pd.concat([G2P_suspicious, td_df], ignore_index=False)

        print("# of G2P Suspicious: {}".format(len(G2P_suspicious)))

        return G2P_suspicious
    
    g2p_suspicious = find_g2p_suspicious(df)

    # 모델과 토크나이저 초기화
    model_name = "facebook/m2m100_418M"
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(MODEL_DIR)
    model.to(DEVICE)

    # 문장 변환 함수
    def convert_text(text):
        input_ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
        outputs = model.generate(input_ids=input_ids, max_length=50)
        converted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return converted_text
    
    predictions = []
    for text in tqdm(g2p_suspicious["text"], total=len(g2p_suspicious)):
        converted_text = convert_text(text)
        converted_text = converted_text.replace("...", "…")
        print(text)
        print(converted_text)
        predictions.append(converted_text)

    # 예측 결과 저장
    predictions_df = pd.DataFrame({"text": predictions})
    g2p_converted = g2p_suspicious.copy()
    g2p_converted.text = predictions_df

    df_converted = df.copy()
    for idx, gc in g2p_converted.iterrows():
        df_converted = pd.concat([df_converted, pd.DataFrame(gc).transpose()], ignore_index=True)

    return df_converted
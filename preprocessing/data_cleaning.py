import os
import torch
import pandas as pd
from g2pk import G2p
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

  
def add_label_in_text(df):
    """
    text에 label을 설명하는 문장을 추가하는 함수

    Args:
        df (DataFrame): train data

    Returns:
        DataFrame: 함수 적용 된 train data
    """
    label_dict = {0: "IT과학", 1: "경제", 2: "사회",
                  3: "생활문화", 4: "세계", 5: "스포츠", 6: "정치"}

    for index in range(len(df)):
        label = df["target"][index]
        df["text"][index] = f"이 문장의 분류는 {label_dict[label]}입니다. " + \
            df["text"][index]

    return df

  
def g2p_cleaning(df, find_g2p=True):
    """
    (1) DataFrame 형태의 데이터를 받아 G2P로 의심되는 데이터들을 선별
    (2) G2P 의심 데이터들에 대해 Seq2Seq 모델을 통한 grapheme 복원
    소요 시간 크니 실행에 유의

    find_g2p: True인 경우 phoneme으로 의심되는 단어를 선별 후 P2G 복원, False인 경우 선별 없이 모두 P2G 복원 적용

    Args:
        df (pd.DataFrame): text 헤더를 포함한 DataFrame 데이터

    Returns:
        df_converted (pd.DataFrame): G2P 의심 데이터들에 대해 grapheme 복원을 적용한 DataFrame 데이터
    """
    BASE_DIR = os.getcwd()
    MODEL_DIR = os.path.join(BASE_DIR, '../p2g_model_m2m100_large')

    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("DEVICE:", DEVICE)

    # G2P 의심 데이터 탐색
    def find_g2p_suspicious(df):
        """
        모든 입력 데이터 텍스트에 G2P 변환을 사용하여, 변환시 원문과 차이가 없는 경우 G2P 의심 데이터로 선별
        G2P 변환은 prescriptive와 descriptive 변환 모두 적용되며, 두 변환 중 한 번이라도 차이가 없는 경우 선별됨

        Args:
            df (pd.DataFrame): text 헤더를 포함한 DataFrame 데이터

        Returns:
            G2P_suspicious (pd.DataFrame): DataFrame 형식의 선별된 G2P 의심 데이터
        """
        G2P_suspicious = pd.DataFrame()
        g2p = G2p()
        print("[Finding G2P Suspicious]")
        for idx, td in tqdm(df.iterrows(), total=len(df)):
            if g2p(td.text, descriptive=False) == td.text or g2p(td.text, descriptive=True) == td.text:
                td_df = pd.DataFrame(td).transpose()
                G2P_suspicious = pd.concat([G2P_suspicious, td_df], ignore_index=False)

        print("# of G2P Suspicious: {}".format(len(G2P_suspicious)))

        return G2P_suspicious
    
    g2p_suspicious = find_g2p_suspicious(df) if find_g2p else df

    # 모델과 토크나이저 초기화
    model_name = "facebook/m2m100_418M"
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(MODEL_DIR)
    model.to(DEVICE)

    # 문장 변환 함수
    def convert_text(text):
        """
        입력 텍스트에 대해 Seq2Seq 모델을 통한 P2G 변환을 적용

        Args:
            text (str): 입력된 DataFrame 데이터 내 text 헤더의 텍스트 데이터

        Returns:
            converted_text (str): P2G 변환이 적용된 text 데이터
        """
        input_ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
        outputs = model.generate(input_ids=input_ids, max_length=50)
        converted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return converted_text
    
    # 각 문장 변환
    print("[Converting Phonemes into Graphemes]")
    predictions = []
    for text in tqdm(g2p_suspicious["text"], total=len(g2p_suspicious)):
        converted_text = convert_text(text)
        converted_text = converted_text.replace("...", "…")
        predictions.append(converted_text)

    # 예측 결과 저장
    g2p_converted = g2p_suspicious.copy()
    g2p_converted.text = predictions

    df_converted = df.copy()
    df_converted = df_converted[~df_converted.ID.isin(g2p_suspicious.ID)]
    for idx, gc in g2p_converted.iterrows():
        df_converted = pd.concat([df_converted, pd.DataFrame(gc).transpose()], ignore_index=True)

    return df_converted

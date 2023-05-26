import os
import pandas as pd
from tqdm import tqdm
from PyKakao import KoGPT


def generate_sentence(data):
    """
    kakao KoGPT api를 사용해 기존 데이터를 비슷한 문장으로 생성해주는 함수

    Args:
        data (DataFrame): train data

    Returns:
        list: 생성된 문장 리스트
    """
    # 프롬프트
    prompt = """주어진 문장과 같은 의미의 문장을 생성
    예시
    입력:유튜브 내달 2일까지 크리에이터 지원 공간 운영
    출력:유튜브 내달 2일까지 크리에이터를 위한 공간 운영

    입력:어버이날 맑다가 흐려져 남부지방 옅은 황사
    출력:어버이날 날씨 맑다가 흐려져 남부지방은 옅은 황사

    입력:"""
    # 생성할 문장의 최대 토큰 수
    max_tokens = 20
    # 생성된 문장을 저장할 리스트
    result_list = []

    for text in tqdm(data["input_text"]):
        # 프롬프트에 input_text 추가
        input_text = prompt+text+"\n출력:"
        # 문장 생성 요청
        respond = api.generate(input_text, max_tokens,
                               temperature=0.1, top_p=0.9)
        # 결과 추출 후 저장
        result = respond["generations"][0]["text"]
        result = result.split("\n")[0]
        result_list.append(result)

    return result_list


def change2df(train_path, new_data):
    """
    입력받은 데이터를 train data 형식에 맞게 DataFrame으로 변경하는 함수

    Args:
        train_path (str): train data 경로
        new_data (list): 생성된 문장 리스트

    Returns:
        DataFrame: train data 형식에 맞게 변경된 데이터
    """
    # 원본 데이터에서 input_text만 변경
    ag_data = pd.read_csv(train_path)
    ag_data["input_text"] = new_data

    return ag_data


def concat_with_origin(train_path, save_path, data):
    """
    원본 데이터와 결합한 뒤 csv 파일로 저장하는 함수

    Args:
        train_path (str): 원본 데이터 경로
        save_path (str): 새로운 데이터 저장할 경로
        data (DataFrame): 생성된 문장 데이터
    """
    train_data = pd.read_csv(train_path)
    new_data = pd.concat((train_data, data), axis=0)
    new_data.to_csv(save_path, index=False)


if __name__ == "__main__":
    # 저장할 버전 선택
    version = "v0.3"

    api_key = os.environ.get("KoGPT_KEY")
    api = KoGPT(api_key)

    train_path = "/opt/ml/level2_nlp_datacentric-nlp-09/data/train_v0.csv"
    save_path = f"/opt/ml/level2_nlp_datacentric-nlp-09/data/train_{version}.csv"

    train_data = pd.read_csv(train_path)
    new_data_list = generate_sentence(train_data)
    new_data_df = change2df(train_path, new_data_list)
    concat_with_origin(train_path, save_path, new_data_df)

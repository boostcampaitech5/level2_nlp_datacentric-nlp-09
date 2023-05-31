import pandas as pd


def temp_cleaning(df):
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

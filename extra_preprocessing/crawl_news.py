import requests
from bs4 import BeautifulSoup as bs
import pandas as pd


def crawl_pages(topic):
    """
    뉴스 기사 제목 크롤링 함수

    Args:
        topic (list): 크롤링 할 뉴스 카테고리

    Returns:
        list : 각 뉴스 제목을 원소로 하는 list
    """
    title_list = []
    for page in range(1, topics_dict[topic][0]+1):
        url = topics_dict[topic][1]+str(page)

        # 서버 요청
        res = requests.get(url)
        res.raise_for_status()
        res.encoding = "utf-8"

        # 뉴스 제목 추출
        soup = bs(res.text, "html.parser")
        section = soup.find('div', attrs={"class": "section01"})
        titles = section.find_all('strong', attrs={"class": "tit-news"})

        title_list = title_list+[title.get_text() for title in titles]

    return title_list


def change2df(data, topic, target):
    """
    입력받은 데이터를 train data 형식에 맞게 DataFrame으로 변경하는 함수

    Args:
        data (list): 뉴스 제목 리스트
        topic (list): 뉴스 카테고리
        target (int): 뉴스 카테고리의 타겟 number

    Returns:
        DataFrame : train data 형식에 맞게 변경된 뉴스 데이터
    """
    # train.csv 형식
    df = pd.DataFrame(columns=["ID", "input_text", "label_text", "target",
                      "predefined_news_category", "annotations", "url", "date"])
    df["input_text"] = data
    df["label_text"] = topic
    df["target"] = target

    return df


def preprocessing(df):
    """
    전처리 함수

    Args:
        df (DataFrame): DataFrame 형태로 변환된 뉴스 데이터

    Returns:
        DataFrame: 네가지 전처리를 거친 데이터
    """
    # [] 내용 제거
    df['input_text'] = df['input_text'].str.replace(r'\[.*?\]', '', regex=True)
    # '(', ')' 제거
    df['input_text'] = df['input_text'].str.replace(r'\(|\)', '', regex=True)
    # 양쪽 공백 제거
    df['input_text'] = df['input_text'].str.strip()
    # 10글자 미만 제거
    # ex) "[오늘의 국회일정](24일·수)" 전처리하면 "24일·수"만 남음
    df = df[df['input_text'].str.len() > 10]

    return df


def concat_with_origin(train_path, df, version):
    """
    원본 데이터와 결합한 뒤 csv 파일로 저장하는 함수

    Args:
        train_path (str): 원본 데이터 경로
        df (DataFrame): 크롤링 한 뉴스 데이터
        version (str): 데이터 버전

    Returns:
        None
    """
    train_data = pd.read_csv(train_path)
    new_data = pd.concat((train_data, df), axis=0)
    new_data.to_csv(
        f"/opt/ml/level2_nlp_datacentric-nlp-09/data/train_{version}.csv", index=False)


if __name__ == "__main__":
    # 데이터 버전 변경한 뒤 실행
    train_path = "/opt/ml/level2_nlp_datacentric-nlp-09/data/train_v0.csv"
    version = "v0.3"

    topics = ["정치", "경제", "사회", "생활문화", "세계", "IT과학", "스포츠"]
    topics_dict = {"정치": [14, "https://www.yna.co.kr/politics/all/"],
                   "경제": [20, "https://www.yna.co.kr/economy/all/"],
                   "사회": [20, "https://www.yna.co.kr/society/all/"],
                   "생활": [14, "https://www.yna.co.kr/lifestyle/all/"],
                   "문화": [10, "https://www.yna.co.kr/culture/all/"],
                   "세계": [20, "https://www.yna.co.kr/international/all/"],
                   "IT과학": [20, "https://www.yna.co.kr/industry/technology-science/"],
                   "스포츠": [11, "https://www.yna.co.kr/sports/all/"]}

    # ag_dataset에 크롤링한 데이터를 모은 뒤 한번에 train data와 결합
    ag_dataset = pd.DataFrame(columns=["ID", "input_text", "label_text", "target",
                                       "predefined_news_category", "annotations", "url", "date"])
    for topic in topics:
        # 생활문화는 생활과 문화를 합친 카테고리
        if topic == "생활문화":
            ag_data = preprocessing(change2df(
                crawl_pages("생활")+crawl_pages("문화"), "생활문화", topics.index(topic)))
            ag_dataset = pd.concat((ag_dataset, ag_data), axis=0)
        else:
            ag_data = preprocessing(
                change2df(crawl_pages(topic), topic, topics.index(topic)))
            ag_dataset = pd.concat((ag_dataset, ag_data), axis=0)

    concat_with_origin(train_path, ag_dataset, version)

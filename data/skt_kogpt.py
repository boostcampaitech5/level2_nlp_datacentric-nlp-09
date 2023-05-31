from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import pandas as pd
from tqdm import tqdm


def set_model():
    """
    skt/kogpt2를 사용하기 위한 tokenizer와 model을 설정하는 함수

    Returns:
        model: kogpt model
        tokenizer: kogpt tokenizer
    """
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        "skt/kogpt2-base-v2", bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
    model = GPT2LMHeadModel.from_pretrained(
        'skt/kogpt2-base-v2').to(device='cuda', non_blocking=True)

    return model, tokenizer


def generation(model, tokenizer, data_path):
    """
    model로 문장을 생성하는 함수

    Args:
        model: kogpt2 model
        tokenizer: kogpt2 tokenizer
        data_path (str): 입력 문장 데이터 경로

    Returns:
        list: 생성된 문장 리스트
    """
    train_data = pd.read_csv(data_path)

    prompt = """비슷한 뉴스 제목 생성:"""

    model.eval()

    new_data = []
    for text in tqdm(train_data["text"]):
        if "…" in text:
            text = text.replace("…", "")
        input_text = prompt + "\n" + text + "\n"
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(
            device='cuda', non_blocking=True)
        gen_ids = model.generate(input_ids,
                                 max_length=40,
                                 repetition_penalty=1.0,
                                 pad_token_id=tokenizer.pad_token_id,
                                 eos_token_id=tokenizer.eos_token_id,
                                 bos_token_id=tokenizer.bos_token_id,
                                 use_cache=True)
        generated = tokenizer.decode(gen_ids[0])
        new_data.append(generated.split("\n")[2])

    return new_data


def change2df(data_path, new_data):
    """
    입력받은 데이터를 train data 형식에 맞게 DataFrame으로 변경하는 함수

    Args:
        data_path (str): 입력 문장 데이터 경로
        new_data (list): 생성된 문장 리스트

    Returns:
        DataFrame: train data 형식에 맞게 변경된 데이터
    """
    # 원본 데이터에서 text만 변경
    ag_data = pd.read_csv(data_path)
    ag_data["text"] = new_data

    return ag_data


def preprocessing(df):
    """
    전처리 함수

    Args:
        df (DataFrame): 생성된 문장

    Returns:
        DataFrame: 전처리 된 문장
    """
    # 양쪽 공백 제거
    df['text'] = df['text'].str.strip()
    # 10글자 미만 제거
    # 10글자 이하 불완전한 문장 제거
    df = df[df['text'].str.len() > 10]

    return df


if __name__ == "__main__":
    data_path = "./data/train_v0.csv"
    model, tokenizer = set_model()
    new_data = generation(model, tokenizer, data_path)
    new_df = change2df(data_path, new_data)
    new_df = preprocessing(new_df)
    new_df.to_csv("./data/train_v0.11.csv", index=False)

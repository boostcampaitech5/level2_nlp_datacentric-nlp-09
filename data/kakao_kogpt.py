import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm


def set_model():
    """
    kokoabrain/kogpt2를 사용하기 위한 tokenizer와 model을 설정하는 함수

    Returns:
        model: kogpt model
        tokenizer: kogpt tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        # or float32 version: revision=KoGPT6B-ryan1.5b
        'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',
        bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
    )
    model = AutoModelForCausalLM.from_pretrained(
        # or float32 version: revision=KoGPT6B-ryan1.5b
        'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',
        pad_token_id=tokenizer.eos_token_id,
        torch_dtype='auto', low_cpu_mem_usage=True
    ).to(device='cuda', non_blocking=True)

    return model, tokenizer


def generation(model, tokenizer, data_path):
    """
    model로 문장을 생성하는 함수

    Args:
        model: kogpt model
        tokenizer: kogpt tokenizer
        data_path (str): 입력 문장 데이터 경로

    Returns:
        list: 생성된 문장 리스트
    """
    train_data = pd.read_csv(data_path)

    prompt = """비슷한 의미의 뉴스 제목 생성:
    유튜브 내달 2일까지 크리에이터 지원 공간 운영 => 유튜브에서 다음달 2일까지 유튜버를 위한 공간을 운영합니다.
    어버이날 맑다가 흐려져 남부지방 옅은 황사 => 어버이날 날씨는 맑다가 흐려지며 남부지방은 옅은 황사가 예상됩니다.
    """

    model.eval()
    with torch.no_grad():
        new_data = []
        for text in tqdm(train_data["text"]):
            input_text = prompt + text + " =>"

            tokens = tokenizer.encode(input_text, return_tensors='pt').to(
                device='cuda', non_blocking=True)

            gen_tokens = model.generate(
                tokens, do_sample=False, temperature=0.9, max_length=110, top_p=0.9)

            output = tokenizer.batch_decode(gen_tokens)[0].split("=> ")[-1]
            output = output.split("\n")[0]
            output = output.strip()
            new_data.append(output)

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


if __name__ == "__main__":
    data_path = "./data/train_v0.csv"
    model, tokenizer = set_model()
    new_data = generation(model, tokenizer, data_path)
    new_df = change2df(data_path, new_data)
    new_df.to_csv("./data/train_v0.3.csv", index=False)

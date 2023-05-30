import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("DEVICE:", DEVICE)

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '../data')
MODEL_DIR = os.path.join(BASE_DIR, '../p2g_model')

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

# 테스트 데이터 로드
phonemes_test = pd.read_csv(os.path.join(DATA_DIR, "g2p_suspicious.csv"))

# 변환 예측
predictions = []
for text in tqdm(phonemes_test["text"], total=len(phonemes_test)):
    converted_text = convert_text(text)
    converted_text = converted_text.replace("...", "…")
    print(text)
    print(converted_text)
    predictions.append(converted_text)

# 예측 결과 저장
predictions_df = pd.DataFrame({"text": predictions})
phonemes_test_converted = phonemes_test.copy()
phonemes_test_converted.text = predictions_df
phonemes_test_converted.to_csv(os.path.join(DATA_DIR, "g2p_suspicious_converted.csv"), index=False)

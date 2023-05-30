import os
import torch
import pandas as pd
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '../data')
OUTPUT_DIR = os.path.join(BASE_DIR, '../p2g/output')

# 모델과 토크나이저 초기화
model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_1.2B")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B")

# 저장된 모델 가중치 로드
model_weights_path = os.path.join(OUTPUT_DIR, "best_p2g_model/pytorch_model.bin")
model.load_state_dict(torch.load(model_weights_path, map_location=DEVICE))

# 문장 변환 함수
def convert_text(text):
    input_ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
    outputs = model.generate(input_ids=input_ids, max_length=50)
    converted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return converted_text

# 테스트 데이터 로드
graphemes_test = pd.read_csv(os.path.join(DATA_DIR, "test_graphemes.csv"))
phonemes_test = pd.read_csv(os.path.join(DATA_DIR, "test_phonemes.csv"))

# 변환 예측
predictions = []
for text in graphemes_test["text"]:
    converted_text = convert_text(text)
    predictions.append(converted_text)

# 예측 결과 저장
output_df = pd.DataFrame(
    {"original_grapheme": graphemes_test["text"], 
     "original_phoneme": phonemes_test["text"], 
     "converted_phoneme": predictions}
    )
output_df.to_csv(os.path.join(OUTPUT_DIR, "p2g_results.csv"), index=False)

import sys
sys.path.append('/opt/ml/level2_nlp_datacentric-nlp-09')
from preprocessing.data_cleaning import g2p_cleaning

from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
from transformers import AutoModelForSequenceClassification
from tokenization_kobert import KoBertTokenizer
from cleanlab.filter import find_label_issues
from transformers import AutoModelForSequenceClassification, AutoConfig
    
train = pd.read_csv('../data/train_v0.12.1.csv')
valid = pd.read_csv('../data/valid_v0.12.1.csv')
dataset_total = pd.concat([train, valid], axis=0)

# 모델 체크포인트 파일과 구성 설정 파일 경로
config_file = "../output/checkpoint-2000/config.json"
model_checkpoint = "../output/checkpoint-2000/pytorch_model.bin"

# 구성 설정 파일 로드
config = AutoConfig.from_pretrained(config_file)

# 토크나이저 및 모델 로드
tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert")
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, config=config)


probs = []
print("\n[Cleanlab Label Error Detection]")
for _, dt in tqdm(dataset_total.iterrows(), total=len(dataset_total)):
    # 입력 문장
    input_text = dt["text"]

    # 입력 문장 토큰화 및 패딩
    input_ids = tokenizer.encode(input_text, add_special_tokens=True)
    input_ids = torch.tensor([input_ids])
    # print(input_ids)
    # 모델에 입력 전달
    outputs = model(input_ids)

    # 로짓(마지막 레이어 출력값)과 클래스 확률 계산
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)

    # 클래스 확률 출력
    class_probabilities = probabilities[0].tolist()
    # print(class_probabilities)
    probs.append(class_probabilities)


label_dict = {0:"IT과학", 1:"경제", 2:"사회", 3:"생활문화", 4:"세계", 5:"스포츠", 6:"정치"}

# cleanlab
train_pred_probs = torch.Tensor()

ordered_label_issues = find_label_issues(
    labels=dataset_total["target"],
    pred_probs=np.array(probs),
    return_indices_ranked_by="self_confidence",
)

print("\n[Print Changed Sentences and Labels]")
count = 0
for issue in ordered_label_issues:
    count += 1
    print(issue,dataset_total.iloc[issue]["text"],"|",
          label_dict[dataset_total.iloc[issue]["target"]],
          dataset_total.iloc[issue]["target"])
    # print(probs[issue])
print(count)

# 예측값으로 학습데이터 재 라벨링
new_target = [prob.index(max(prob)) for prob in probs[:1371]] + dataset_total.target.iloc[1371:]
total_converted = dataset_total.copy()
total_converted["target"] = new_target


train_g2p_cleaned = pd.read_csv('../data/train_v0.12.1.csv')

print("\n[Train Eval Splitting]")
train_g2p_label_cleaned = pd.DataFrame()
valid_g2p_label_cleaned = pd.DataFrame()
for idx, tc in tqdm(total_converted.iterrows(), total=len(total_converted)):
    if tc.ID in train_g2p_cleaned.ID.tolist():
        train_g2p_label_cleaned = pd.concat([train_g2p_label_cleaned, pd.DataFrame(tc).transpose()], ignore_index=True)
    else:
        valid_g2p_label_cleaned = pd.concat([valid_g2p_label_cleaned, pd.DataFrame(tc).transpose()], ignore_index=True)

print("Train Length:", len(train_g2p_label_cleaned))
print("Valid Length:", len(valid_g2p_label_cleaned))
train_g2p_label_cleaned.to_csv('../data/train_g2p_label_cleaned.csv', index=False)
valid_g2p_label_cleaned.to_csv('../data/valid_g2p_label_cleaned.csv', index=False)
print("# of Changed Target in Train:", sum(train.target != train_g2p_label_cleaned.target))
print("# of Changed Target in Valid:", sum(valid.target != valid_g2p_label_cleaned.target))

import torch
import random
import torch.nn as nn
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from nltk.translate.bleu_score import corpus_bleu

import os
import wandb


SEED = 456
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("DEVICE:", DEVICE)

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '../data')
OUTPUT_DIR = os.path.join(BASE_DIR, '../p2g/best_p2g_model')

model_name = "facebook/m2m100_418M"

# 데이터셋 클래스 정의
class PhonemeGraphemeDataset(Dataset):
    def __init__(self, phoneme_data, grapheme_data):
        self.phoneme_data = phoneme_data
        self.grapheme_data = grapheme_data
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    
    def __len__(self):
        return len(self.phoneme_data)
    
    def __getitem__(self, index):
        phoneme_text = self.phoneme_data[index]
        grapheme_text = self.grapheme_data[index]
        
        phoneme_inputs = self.tokenizer.encode_plus(
            phoneme_text,
            padding="max_length",
            truncation=True,
            max_length=50,
            return_tensors="pt"
        )
        
        grapheme_labels = self.tokenizer.encode_plus(
            grapheme_text,
            padding="max_length",
            truncation=True,
            max_length=50,
            return_tensors="pt"
        )
        
        return {
            "input_ids": phoneme_inputs["input_ids"].squeeze(),
            "attention_mask": phoneme_inputs["attention_mask"].squeeze(),
            "labels": grapheme_labels["input_ids"].squeeze()
        }

# 모델 초기화
model = M2M100ForConditionalGeneration.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
tokenizer = M2M100Tokenizer.from_pretrained(model_name)

# 데이터 불러오기
phoneme_data = pd.read_csv(os.path.join(DATA_DIR, "total_phonemes.csv")).text
grapheme_data = pd.read_csv(os.path.join(DATA_DIR, "total_graphemes.csv")).text

# 데이터 분할
train_phoneme, eval_phoneme, train_grapheme, eval_grapheme = train_test_split(
    phoneme_data, grapheme_data, test_size=0.2, random_state=SEED
)

train_phoneme = train_phoneme.reset_index(drop=True)
train_grapheme = train_grapheme.reset_index(drop=True)
eval_phoneme = eval_phoneme.reset_index(drop=True)
eval_grapheme = eval_grapheme.reset_index(drop=True)

# 데이터셋과 데이터로더 초기화
train_dataset = PhonemeGraphemeDataset(train_phoneme, train_grapheme)
eval_dataset = PhonemeGraphemeDataset(eval_phoneme, eval_grapheme)

# TrainingArguments 설정
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=1e-5,
    lr_scheduler_type='linear',
    weight_decay=0.01,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    save_total_limit=1,
    logging_dir="./logs",
    logging_strategy='steps',
    evaluation_strategy='steps',
    save_strategy='steps',
    logging_steps=100,
    eval_steps=500,
    save_steps=500,
    seed=SEED,
    report_to="wandb",
)

# 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()

# 학습 및 평가 함수 정의
def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    labels = pred.label_ids

    # Convert predictions and labels to numpy arrays
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    # Calculate BLEU score
    bleu_score = corpus_bleu([[label] for label in labels], preds)

    return {"BLEU": bleu_score}

# WandB 초기화
wandb.init(project="data_centric_p2g", name='baseline')

# Trainer 초기화
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=None,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 모델 훈련
trainer.train()

# 학습 완료 후 저장
best_model_path = OUTPUT_DIR
trainer.save_model(best_model_path)
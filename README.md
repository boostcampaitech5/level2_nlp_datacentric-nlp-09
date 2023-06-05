# 주제 분류 프로젝트

모델 구조의 변경 없이 Data-Centric 관점으로 텍스트의 주제를 분류하는 태스크입니다.

## 1. Introduction

---

### Competition Task & Details

- 주제 분류(topic classification)
- [모델] monologg/kobert - 변경 불가 / 베이스라인 F1 score 0.8720
- [데이터] KLUE-TC(YNAT, 연합 뉴스 기사 제목) 45,678개(train) + 9,107개(test) → train 내 15%에 인위적 노이즈 포함
    - [Train 데이터 구성] IT과학 5,309개 / 경제 6,119개 / 사회 5,180개 / 생활문화 5,760개 / 세계 8,250개 / 스포츠 7,668개 / 정치 7,372개

## 2. Team Members

---

**김세형_T5038**

- G2P 및 label 노이즈 탐지 및 복원

**이준선_T5157**

- 생성모델, 크롤링을 통한 데이터 증강

**정윤석_T5194**

- 전처리 툴, 특정 문자를 이용한 데이터 클리닝

**이동호_T5139**

- 생성모델, 외부 데이터를 통한 데이터 증강

**홍찬우_T5227**

- backtranslate, annotation bot, 특수문자 제거

## 3. Data Cleaning

---

### **3.1. G2P (Grapheme-to-Phoneme) Error Denoising(김세형)**

- 총 45,678개 train 데이터셋 내 12%(5,481개)의 G2P 형태 text perturbation 에러 존재
- **탐색**: 한국어 G2P 변환기(`Kyubyong/g2pK` [1]) 사용하여 prescriptive / descriptive phoneme 변환 시 원본과 동일한 경우 G2P 노이즈로 판단
    - 실제 5,481개 존재 / 약 6,600개 G2P 의심 데이터 선별 및 제거
    - **Public F1: 0.8778 (베이스라인 대비 0.0058 상승)**
- **복원**: 다국어 기계번역을 위한 `facebook/m2m100` [2] 모델의 두 버전(418M, 1.2B) 및 한국어 텍스트 생성 목적의 `kykim/bertshared-kor-base` [3] 모델을 이용하여 phoneme-to-grapheme 변환
    1. 전체 train 데이터 45,678개 중 phoneme으로 의심되는 데이터를 제거하고 남은 데이터 약 39,000개를 G2P 변환기를 통해 phoneme으로 변환(임의로 50%는 prescriptive, 나머지 50%는 descriptive)
    2. Seq2Seq 모델을 phoneme → grapheme으로 번역하도록 fine-tuning
    3. 제외한 6600개 데이터에 대해 fine-tuned 모델을 적용하여 grapheme 복원
    - **최고 public F1: 0.8780 (베이스라인 대비 0.0060 상승, `facebook/m2m100_1.2B`)**
        
        <aside>
        💡 **[왜 한국어 모델보다 다국어 모델 성능이 좋은가?]**
        Phoneme의 특성상 기존 한국어 모델의 vocab 내에 존재하지 않는 단어들이 다수 존재하여, 오히려 byte-pair tokenizing을 수행하는 다국어 번역 모델 성능이 더 좋았던 것으로 추측
        
        </aside>
        

### **3.2. Label Error Denoising(김세형)**

- 총 45,678개 train 데이터셋 내 3%(1,371개)의 label 에러 존재
- **탐색 및 복원**: `Cleanlab` [4] 라이브러리 이용
    - **Cleanlab**: 모델이 예측한 logit(확률) 값을 이용하여, 현재 label과 예측 label의 logit 값 차이가 큰 경우 label error로 판단하고 label을 모델이 예측한 결과로 변경
    - 전체 45,678개 데이터를 모두 모델에 넣어 logit 값과 비교 후, 실제 노이즈 개수(1,371개)와 오차범위를 고려해 상위 1,500개 데이터의 label을 제거 또는 복원
    - **제거시 Public F1: 0.8724 (베이스라인 대비 0.0004 상승, G2P 복원 데이터 대비 0.0056 감소)**
    - **복원시 Public F1: 0.8789 (베이스라인 대비 0.0069 상승, G2P 복원 데이터 대비 0.0009 상승)**

### **3.3. Hard Cleaning(정윤석)**

- 공백, “,” , “·” 을 기준으로 문장을 split 하여 자주 등장하는 단어들 확인
- 자주 등장하는 단어가 특정 주제를 가리킬 경우 해당 레이블링로 바꾸기
    - annotator의 주관이 아닌 주관이 들어가서 그런지 성능 감소

### 3.4. 특수문자 처리 (홍찬우)

- 기사 제목이 길면 …으로 끝나는 경우가 존재
    - 의미가 없다고 판단해 …을 제거하고 학습 (Public에서 가장 좋은 성능을 나타냄)
- 그 외 [UNK] token으로 변환되는 한자 및 특수 문자들을 제거
    - …을 제외한 다른 특수 문자들은 제거 시 성능 감소

## 4. Data Augmentation

---

### **4.1. Swap Text(정윤석)**

- … 을 기준으로 문장을 swap 하여 증강하기
    - 성능 감소

### **4.2. External Data**

- **AI HUB(이동호)**
    - AI HUB의 뉴스 기사 기계독해 데이터 활용 [5]
    - IT과학, 경제 라벨 데이터를 각각 3000개씩 추가
    - **Public F1: 0.8013 (베이스라인 대비 0.0707 하락)**
- **Crawling(이준선)**
    - klue 데이터셋이 연합뉴스 기사 제목이므로 최신 연합뉴스 기사 제목을 크롤링해 데이터 증강 시도
    - 오히려 성능이 하락

### **4.3. Generative Language Models**

- **`beomi/KoAlpaca-Polyglot-12.8B` (이동호)**
    - `**beomi/KoAlpaca-Polyglot-12.8B`** [6]를 한국어 데이터셋으로 fine-tuning한 생성모델 [7]
    - 데이터셋 전체 문장에 대해 각각 유사한 문장을 하나씩 생성
    - 성능 하락
- **`kakaobrain/kogpt`(이준선)**
    - 카카오브레인에서 개발한 gpt3 기반 한국어 언어모델[8]
    - api 사용 시도했으나 무료 제한으로 인해 huggingface에서 직접 불러서 사용
    - 학습 데이터를 1대1로 비슷한 문장 생성
    - 베이스 코드 대비 약 0.2% 성능 향상
- **`skt/kogpt2-base-v2`(이준선)**
    - skt의 gpt2 기반 한국어 언어모델[9]
    - 학습 데이터를 1대1로 비슷한 문장 생성
    - 베이스 코드 대비 약 0.07% 성능 향상

### **4.4. Back-Translatation (홍찬우)**

- **HuggingFace pre-trained machine translation model 이용**
    - 선학습 번역 모델을 train dataset으로 fine-tuning 후 한국어 → 영어 → 한국어 backtranslate 시도
    - 번역 성능이 매우 좋지 않았고, 그에 따른 성능 대폭 감소
- **Python Library 이용**
    - Googletrans 라이브러리를 이용한 backtranslate
        - 어려운 문장에 대해선 번역 성능이 좋지 않음
    - roundtrip 라이브러리 이용
        - 번역 성능이 다른 방법들에 비해 월등히 좋았고, baseline 기준 0.02 성능 향상

### **4.5. Annotation Bot (홍찬우)**

- **AI Hub 및 연합뉴스에서 수집한 외부 데이터를 학습 데이터의 annotation 추세와 비슷하게 labeling 시도**
    - HuggingFace klue/roberta-large 모델을 학습 데이터에 대해 fine-tuning
    - 학습된 model로 외부 데이터 labeling
    - 증강 데이터 + 원본 데이터로 학습 dataset을 재구성
    - 데이터를 4배 가량 증강했음에도 성능이 감소

## 5. Self-Evaluation

---

### **5.1. What’s Good**

- 이슈가 많은 대회였으나 마지막까지 팀원 모두가 끝내 놓지 않고 임했으며, 마지막 날 순위를 11위에서 3위까지 끌어올렸음.
- 첫 날 GitHub commit convention을 정하고 대회 기간 내 최대한 열심히 지키며 협업하기 위해 노력함.
- 첫 대회때 하지 못했던 결과 분석(confusion matrix, precision-recall curve 등)을 활용하여 진행 방향 결정에 활용함.

### **5.2. What’s Bad**

- GitHub flow로 대회를 진행하였으나 commit & push, 그리고 pull request의 처리가 늦은 감이 있음.
- 코드와 결과를 공유는 하지만 아직 진정한 의미의 ‘협업’과는 조금 거리감이 있음. 이번 대회 때 다들 초반 열정이 좋아서 한 층 발전할 수 있을 것 같았는데, 사실 운영 이슈가 의욕을 좀 꺾은 부분도 있음. 다음 대회가 기대됨.
- 성능이 하락하는 이유에 대한 깊은 분석이 부족함. 정확한 원인 분석을 토대로 성능 향상을 이루는 경험까지 이어질 필요가 있음.

### **5.3. What’s Learned**

- LLM 모델 기반 task의 데이터 처리에 LLM을 이용하는, LLM-to-LLM 방식의 학습을 진행한 경험을 쌓음. 앞으로의 대회나 프로젝트에 적극 활용해도 좋을 것 같음.
- 데이터를 증강했다면, 증강 데이터에 대한 정제를 거치거나 신뢰도를 확인한 후 적용할 필요가 있음(from 마스터 클래스).

## References

---

1. Park, K., 2019, *g2pK*, *GitHub*, [https://github.com/Kyubyong/g2pk](https://github.com/Kyubyong/g2pk)
2. Fan, A., Bhosale, S., Schwenk, H., Ma, Z., El-Kishky, A., Goyal, S., ... & Joulin, A. (2021). Beyond english-centric multilingual machine translation. *The Journal of Machine Learning Research*, *22*(1), 4839-4886.
(HuggingFace: [https://huggingface.co/facebook/m2m100_418M](https://huggingface.co/facebook/m2m100_418M))
3. Kim, K., 2020, Pretrained Language Models For Korean, *GitHub*, [https://github.com/kiyoungkim1/LMkor](https://github.com/kiyoungkim1/LMkor)
(HuggingFace: [https://huggingface.co/kykim/bertshared-kor-base](https://huggingface.co/kykim/bertshared-kor-base))
4. Cleanlab Inc., 2017, Cleanlab Documentation, [https://docs.cleanlab.ai/stable/index.html](https://docs.cleanlab.ai/stable/index.html)
5. [https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=577](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=577)
6. [https://huggingface.co/EleutherAI/polyglot-ko-12.8b](https://huggingface.co/EleutherAI/polyglot-ko-12.8b)
7. [https://huggingface.co/beomi/KoAlpaca-Polyglot-12.8B](https://huggingface.co/beomi/KoAlpaca-Polyglot-12.8B)
8. [https://huggingface.co/kakaobrain/kogpt](https://huggingface.co/kakaobrain/kogpt)
9. [https://huggingface.co/skt/kogpt2-base-v2](https://huggingface.co/skt/kogpt2-base-v2)

---

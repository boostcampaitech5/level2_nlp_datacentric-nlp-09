import torch
from transformers import pipeline, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm

MODEL = 'beomi/KoAlpaca-Polyglot-12.8B'

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device=f"cuda", non_blocking=True)
model.eval()

pipe = pipeline(
    'text-generation',
    model=model,
    tokenizer=MODEL,
    device=0
)

prompt_setting = """주어진 문장과 같은 의미의 문장을 생성
    예시
    입력:유튜브 내달 2일까지 크리에이터 지원 공간 운영
    출력:유튜브 내달 2일까지 크리에이터를 위한 공간 운영

    입력:"""


def ask(prompt):
    """
    koalpaca를 이용하여 prompt를 입력으로 받아 문장을 생성합니다
    Args:
        prompt (str): koalpaca에 들어가는 입력 문장입니다

    Returns:
        str: 생성된 문장입니다
    """
    ans = pipe(
        prompt,
        do_sample=True,
        max_new_tokens=512,
        temperature=0.4,
        top_p=0.4,
        return_full_text=False,
        eos_token_id=2,
    )

    return ans[0]['generated_text']


df = pd.read_csv("./data/train_v0.csv", index_col=0)

new_rows = []
for idx, row in tqdm(df.iterrows(), total=len(df)):
    input_text = row.input_text
    prompt = prompt_setting + input_text + "\n출력:"
    output_text = ask(prompt)
    print("inpupt_text:", input_text)
    print("output_text:", output_text)
    new_row = row.copy()
    new_row['input_text'] = output_text
    # print(new_row)
    new_rows.append(new_row)


new_df = pd.DataFrame(new_rows)
concat_df = pd.concat([df, new_df])

concat_df.to_csv("./data/data_aug_with_koalpaca.csv")

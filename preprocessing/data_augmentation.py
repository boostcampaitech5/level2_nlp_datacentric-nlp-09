
import pandas as pd
import re


def temp_aug(df):
    return df


def swap_text_aug(df):
    def swap_text(sentence):
        segment = re.split(r"(…)", sentence)
        segment.reverse()
        sentence = "".join(segment)
        return sentence
    df['input_text'] = df['input_text'].apply(lambda x: swap_text(x))
    return df


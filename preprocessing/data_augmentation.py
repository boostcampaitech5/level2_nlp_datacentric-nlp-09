import pandas as pd
import re


def temp_aug(df):
    return df


def swap_text_aug(df):
    new_df = None
    new_df = df[df['input_text'].str.contains("…")]

    def swap_text(sentence):
        segment = re.split(r"(…)", sentence)
        if len(segment) > 2:
            segment.reverse()
        sentence = "".join(segment)
        return sentence
    new_df['input_text'] = new_df['input_text'].apply(lambda x: swap_text(x))
    return new_df

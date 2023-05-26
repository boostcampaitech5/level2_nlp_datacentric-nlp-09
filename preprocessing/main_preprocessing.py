import pandas as pd
from typing import List
import data_augmentation as DA
import data_cleaning as DC
import yaml
from collections import namedtuple


class SequentialPreprocessing:
    def __init__(self, preprocessing_list: List[str] = []):
        """
        여러 개의 전처리 method를 데이터 프레임에 순차적으로 적용하는 클래스입니다.
        Args:
            preprocessing_list (List[str], optional): 적용할 preprocessing method의 이름 리스트입니다.

        Example:
            SP = SequentialPreprocessing(["remove_special_word", "remove_stop_word"])
            train_df = SP.process(train_df)
        """
        self.preprocessing_list = preprocessing_list

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        입력으로 받은 데이터프레임에 preprocessing 메소드를 순차적으로 적용해서 반환합니다.
        Args:
            df (pd.DataFrame)
        Returns:
            pd.DataFrame
        """
        if self.preprocessing_list:
            for method in self.preprocessing_list:
                if "DC" in method:
                    preprocessing_method = eval(method)
                    df = preprocessing_method(df)
                else:
                    preprocessing_method = eval(method)
                    aug_data = preprocessing_method(df)
                    df = pd.concat([df, aug_data])
        return df


def load_yaml():
    config_file = None
    with open(f"./config/config.yaml") as f:
        config_file = yaml.load(f, Loader=yaml.FullLoader)
    config = namedtuple("config", config_file.keys())
    config_tuple = config(**config_file)
    print(config_tuple)
    return config_tuple


if __name__ == "__main__":
    config_tuple = load_yaml()
    input_path = config_tuple.input_csv_path
    output_path = config_tuple.output_csv_path
    preprocessing_list = config_tuple.data_preprocessing

    df = pd.read_csv(input_path, encoding="utf-8")
    SP = SequentialPreprocessing(preprocessing_list)
    df = SP.process(df)
    df.to_csv(output_path)

import pandas as pd
from typing import List
import data_augmentation as DA
import data_cleaning as DC
import yaml
from collections import namedtuple


class SequentialPreprocessing:
  def __init__(self, preprocessing_list: List[str]=[]):
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
          preprocessing_method = eval(method)
          df = preprocessing_method(df)
      return df




def load_yaml():

    config_dict = None
    with open(f"./config/config.yaml") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)


    Config = namedtuple("config", config_dict.keys())
    config = Config(**config_dict)
    print(config)
    return config


if __name__ == "__main__":
  config = load_yaml()
  input_path = config.input_csv_path
  output_path = config.output_csv_path
  preprocessing_list = config.data_preprocessing

  df = pd.read_csv(input_path, encoding = "utf-8")
  SP=SequentialPreprocessing(preprocessing_list)
  df=SP.process(df)
  df.to_csv(output_path)

  

  
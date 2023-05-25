import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split


def temp_cleaning(df):
  """
    docstring 예시

    Args:
        df (DataType): 예시

    Returns:
        df (DataType): 예시
    """
  return df


def train_test_split_with_drop_duplicates(total_df: pd.DataFrame):
    """
    전체 데이터가 입력되었을 때 베이스라인 코드의 split 데이터를 유지하면서,
    모든 columns가 동일한 중복 데이터를 삭제하는 코드입니다.

    미리 split된 데이터에 각각 적용하기에는 전체 데이터셋 내의 중복 데이터가
    하나의 데이터셋 내에 둘 다 존재할지, 두 데이터셋에 나눠 존재할지 알기가 힘들기 때문에,
    과잉 제거를 방지하고자 전체 데이터셋을 입력으로 받고 내부에서 split해 반환합니다.

    Args:
        total_df (pd.DataFrame): pandas.read_csv로 읽어온 베이스라인 데이터를 split 없이 그대로 입력합니다.

    Returns:
        train_drop (pd.DataFrame), eval_drop (pd.DataFrame): 7:3 split된 채로 전체 데이터의 중복 데이터가 drop된 train_drop, eval_drop 데이터가 반환됩니다.
    """
    duplicated_indices = total_df[total_df.duplicated(keep='first')].index.tolist()
    train_df, eval_df = train_test_split(total_df, train_size=0.7, random_state=42)
    train_drop = train_df[~train_df.index.isin(duplicated_indices)]
    eval_drop = eval_df[~eval_df.index.isin(duplicated_indices)]

    return train_drop, eval_drop


def drop_label_errors(df_with_label_text):
    """
    label_text와 annotations column을 포함하는 데이터에서,
    label_text와 label_text_from_annotations가 다른 경우 label error로 판단하여 제거합니다.

    Args:
        df_with_label_text (pd.DataFrame): label_text와 annotations column이 존재하는 데이터를 입력받습니다.

    Returns:
        df_with_label_text (pd.DataFrame): label error가 제거된 데이터를 반환합니다.
    """
    def anno2df(df: pd.DataFrame):
        """
        데이터 df의 annotations 정보를 pd.DataFrame으로 변환한 annotations_df를 출렵합니다.

        Args:
            df (pd.DataFrame): annotations column이 존재하는 DataFrame을 입력 받습니다.

        Returns:
            annotations_df (pd.DataFrame): annotations 정보를 DataFrame으로 변환하여 출력합니다.
        """
        annotations_dict = defaultdict(list)
        for idx, d in df.iterrows():
            anno_info = eval(d.annotations)
            annos = anno_info['annotations']
            
            annotations_dict['annotators'].append(anno_info['annotators'])
            for anno_key, anno_values in annos.items():
                annotations_dict[anno_key].append(anno_values)

        annotations_df = pd.DataFrame.from_dict(annotations_dict)
        annotations_df

        return annotations_df
    

    def get_labels_from_annotations(annotations_df: pd.DataFrame):
        """
        annotations_df의 scope 정보로부터 label_text와 target을 추론하여,
        label_text_from_annotstions와 target_from_annotations를 출력합니다.

        Args:
            annotations_df (pd.DataFrame): annotations 정보가 DataFrame 형태로 변환된 annotations_df를 입력 받습니다.

        Returns:
            label_text_from_annotations (List): annotations_df의 정보로부터 추측된 label_text를 출력합니다.
            target_from_annotations (List): annotations_df의 정보로부터 추측된 label_text의 target을 출력합니다.
        """
        # category dict의 번호순으로 label 이름 정렬
        categories = ['정치', '경제', '사회', '생활문화', '세계', 'IT과학', '스포츠']
        cat2num = {cat: idx for idx, cat in enumerate(categories)}
        num2cat = {idx: cat for idx, cat in enumerate(categories)}
        
        label_text_from_annotations, target_from_annotations = [], []
        for idx, anno in annotations_df.iterrows():
            score1, score2, score3 = [0] * 7, [0] * 7, [0] * 7
            for s1, s2, s3 in zip(anno['first-scope'], anno['second-scope'], anno['third-scope']):
                if s1 != '해당없음':
                    score1[cat2num[s1]] += 1
                if s2 != '해당없음':
                    score2[cat2num[s2]] += 1
                if s3 != '해당없음':
                    score3[cat2num[s3]] += 1
                    
            if max(score1) > 1:
                score = score1
            else:
                print('first-scope 동점!')
                if max(score2) > 0:
                    score = score2
                else:
                    print('second-scope 동점!')
                    score = score3

            label_argmax = max(range(len(score)), key=lambda x: score[x])
            target_from_annotations.append(label_argmax)
            label_text_from_annotations.append(num2cat[label_argmax])

        return label_text_from_annotations, target_from_annotations

    
    labels_from_annotations, _ = get_labels_from_annotations(anno2df(df_with_label_text))
    label_error_data = df_with_label_text[df_with_label_text.label_text != labels_from_annotations]
    print("Label Error Data: {}".format(len(label_error_data)))

    df_with_label_text = df_with_label_text.drop(index=label_error_data.index)

    return df_with_label_text
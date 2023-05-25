import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from jamo import h2j, j2hcj
from g2pk import G2p
import re


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


def drop_label_errors(df_with_label_text): # 사용 금지
    """
    (사용 금지)
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


def drop_g2p_errors(df: pd.DataFrame):
    """
    데이터 내 G2P 에러 데이터를 제거합니다.
    원본과 비교했을 때,
    (1) 숫자가 사라졌는가(한글로 바뀌었는가)
    (2) 영어가 사라졌는가(한글로 바뀌었는가)
    (3) 받침이 사라졌는가(전체 자모 개수가 줄었는가)
    (4) 어절의 초성에 쌍자음이 생겼는가
    의 기준으로 phoneme 데이터를 판단하며, 위 과정 이후 남은 G2P 데이터는
    (5) G2P 변환기(g2pk.G2p())를 이용하여, ID가 동일한 두 문장 중 한 문장을 G2P 변환할 시 다른 하나로 변경되는지의 여부로 문장을 판별합니다.

    Args:
        df (pd.DataFrame): DataFrame 형태의 데이터를 입력 받습니다.

    Returns:
        df_ge (df.DataFrame): G2P 에러가 제거된 DataFrame 형태의 데이터를 출력합니다.
    """
    errors = df[df.duplicated(subset='ID', keep=False)].sort_index()
    errors_copy = errors.copy()

    g2p = G2p()
    error_indices = []
    for (idx1, e1), (idx2, e2) in zip(errors_copy.iloc[:-1:2].iterrows(), errors_copy.iloc[1::2].iterrows()):
        if e1.input_text == e2.input_text:
            continue
        
        input_words1, input_words2 = e1.input_text.split(), e2.input_text.split()

        error_idx = None
        for iw1, iw2 in zip(input_words1, input_words2):
            jamo_iw1, jamo_iw2 = j2hcj(h2j(iw1)), j2hcj(h2j(iw2))

            # 숫자가 사라졌는가
            numeric_pattern = r"\d"
            if not re.search(numeric_pattern, iw1) and re.search(numeric_pattern, iw2):
                error_idx = idx1
                break
            elif re.search(numeric_pattern, iw1) and not re.search(numeric_pattern, iw2):
                error_idx = idx2
                break

            # 영어가 사라졌는가
            alphabet_pattern = r"[a-zA-Z]"
            if not re.search(alphabet_pattern, iw1) and re.search(alphabet_pattern, iw2):
                error_idx = idx1
                break
            elif re.search(alphabet_pattern, iw1) and not re.search(alphabet_pattern, iw2):
                error_idx = idx2
                break

            # 받침이 사라졌는가
            len_iw1, len_iw2 = len(jamo_iw1), len(jamo_iw2)
            if len_iw1 < len_iw2:
                error_idx = idx1
                break
            elif len_iw1 > len_iw2:
                error_idx = idx2
                break

            # 쌍자음이 생겨났는가
            for single_iw1, single_iw2 in zip(list(iw1)[:min(len(iw1), len(iw2))], list(iw2)[:min(len(iw1), len(iw2))]):
                jamo_single_iw1, jamo_single_iw2 = j2hcj(h2j(single_iw1)), j2hcj(h2j(single_iw2))
                double_pattern = r"ㄲ|ㄸ|ㅃ|ㅆ|ㅉ"
                if re.search(double_pattern, jamo_single_iw1[0]) and not re.search(double_pattern, jamo_single_iw2[0]):
                    error_idx = idx1
                    break
                elif not re.search(double_pattern, jamo_single_iw1[0]) and re.search(double_pattern, jamo_single_iw2[0]):
                    error_idx = idx2
                    break
            if error_idx:
                break
        
        # 남은 문장들에 대해 G2P 변환기를 써서 일치하는 지 확인
        if not error_idx:
            if e2.input_text == g2p(e1.input_text, descriptive=False) or e2.input_text == g2p(e1.input_text, descriptive=True):
                error_idx = idx1
            elif e1.input_text == g2p(e2.input_text, descriptive=False) or e1.input_text == g2p(e2.input_text, descriptive=True):
                error_idx = idx2

        if error_idx:
            error_indices.append(error_idx)
            error_idx = None

    print("G2P Removed:", len(error_indices)) # G2P 에러로 판단되어 제거된 에러의 개수

    df_ge = df.drop(index=error_indices)

    return df_ge

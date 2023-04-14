import os

import pandas as pd


# find . -name '._*' -exec rm {} \;
class cfg:
    data_dir = "/opt/ml/input/data/train"
    img_dir = f"{data_dir}/images"
    df_path = f"{data_dir}/train.csv"
    new_df_path = f"{data_dir}/train_label.csv"  # 새로 저장할 파일 이름


def make_new_data(df):
    def get_mask_value(file_name):
        return 12 if file_name.startswith("normal") else 6 if file_name.startswith("incorrect_mask") else 0

    def get_mask_class(file_name):
        return 2 if file_name.startswith("normal") else 1 if file_name.startswith("incorrect_mask") else 0

    def get_gender_value(gender):
        return 0 if gender == "male" else 3

    def get_gender_class(gender):
        return 0 if gender == "male" else 1

    def get_age_value(age):
        return 0 if age < 30 else 1 if age < 60 else 2

    def get_age_class(age):
        return 0 if age < 30 else 1 if age < 60 else 2

    new_data = []
    for _, data in df.iterrows():
        for file_name in os.listdir(os.path.join(cfg.img_dir, data.path)):
            age_class = get_age_class(data.age)
            mask_class = get_mask_class(file_name)
            gender_class = get_gender_class(data.gender)
            value = get_age_value(data.age) + get_mask_value(file_name) + get_gender_value(data.gender)
            new_data.append([*data, os.path.join(data.path, file_name), value, age_class, mask_class, gender_class])
    return new_data


# train.csv 읽어오는 코드
df = pd.read_csv(cfg.df_path)

# 새로운 DataFrame을 위한 새로운 list
new_df = make_new_data(df)

labeled_df = pd.DataFrame(
    new_df,
    columns=["id", "gender", "race", "age", "path", "full_path", "class", "age_class", "mask_class", "gender_class"],
)
labeled_df.to_csv("new_train_data.csv", index=False)

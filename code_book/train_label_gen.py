import os

import pandas as pd

"""
Overall codes from data_label_gen.py.
csv will save [image_path, class, age_class, mask_class, gender_class]

[Update] mis-labeled data modified 
    (20, 4418 ID mask state, 001498-1, 1720, 4432, 6359, 6360, 6361, 6362, 6363, 6364 ID gender)
"""


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
        # 현재 폴더 순회하며 파일 반환
        for file_name in os.listdir(os.path.join(cfg.img_dir, data.path)):
            if file_name.split("/")[-1].startswith("._"):
                continue
            else:
                age_class = get_age_class(data.age)
                mask_class = get_mask_class(file_name)
                gender_class = get_gender_class(data.gender)
                value = get_age_value(data.age) + get_mask_value(file_name) + get_gender_value(data.gender)

                if data.id in [20, 4418]:
                    if file_name.split("/")[-1].startswith("incorrect_mask"):
                        mask_class = 2
                    elif file_name.split("/")[-1].startswith("normal"):
                        mask_class = 1
                if str(data.id) in ["001498-1", "1720", "4432", "6359", "6360", "6361", "6362", "6363", "6364"]:
                    if gender_class == 0:
                        gender_class = 3
                    else:
                        gender_class = 0

                new_data.append([os.path.join(data.path, file_name), value, age_class, mask_class, gender_class])
    return new_data


# train.csv 읽어오는 코드
df = pd.read_csv(cfg.df_path)

# 새로운 DataFrame을 위한 새로운 list
new_df = make_new_data(df)
# print(new_df)

labeled_df = pd.DataFrame(
    new_df,
    columns=["img_path", "class", "age_class", "mask_class", "gender_class"],
)
labeled_df.to_csv("train_path_label.csv", index=False)

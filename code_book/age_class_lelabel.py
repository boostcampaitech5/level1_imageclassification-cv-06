import pandas as pd

data = pd.read_csv("/opt/ml/level1_imageclassification-cv-06/code_book/train_path_label.csv")
age = data["img_path"].str
data["age"] = age.split("/").str[0].str.split("_").str[-1]

data["re_age_class1"] = data["age"].astype(int).apply(lambda age: 0 if age < 30 else (1 if 30 <= age < 59 else 2))
data["re_age_class2"] = data["age"].astype(int).apply(lambda age: 0 if age < 30 else (1 if 30 <= age < 58 else 2))
data["re_age_class3"] = data["age"].astype(int).apply(lambda age: 0 if age < 29 else (1 if 29 <= age < 58 else 2))

for i in range(1, 4):
    data[f"re_labeled_class{i}"] = data["class"].astype(int) - data["age_class"].astype(int) + data[f"re_age_class{i}"].astype(int)

data.to_csv("/opt/ml/level1_imageclassification-cv-06/code_book/re_labeled_data.csv")
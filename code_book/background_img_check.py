import os

BASE_ROOT = "/opt/ml/input/data/train/bg_sub/"
folder_lst = os.listdir(BASE_ROOT)

lst1 = []
lst2 = []
for i in folder_lst:
    img_foler_path = os.path.join(BASE_ROOT, i)

    img_lst = os.listdir(img_foler_path)

    for j in img_lst:
        img_path_name = os.path.join(img_foler_path, j)
        if img_path_name not in lst1:
            lst1.append(img_path_name)
        else:
            lst2.append(img_path_name)

print(len(lst1), len(lst2))

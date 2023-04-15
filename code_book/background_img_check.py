import os

BASE_ROOT = "/opt/ml/input/data/train/bg_sub/"  # bg_sub image가 존재할 위치
folder_lst = os.listdir(BASE_ROOT)

img = []
img_duplication = []  # 중복되는 이미지가 있는지 확인하기 위해 list 2개를 나눠서 선언
for lst in folder_lst:
    img_folder_path = os.path.join(BASE_ROOT, lst)

    img_lst = os.listdir(img_folder_path)  # img가 담겨 있는 각 폴더까지의 경로
    for j in img_lst:
        img_path_name = os.path.join(img_folder_path, j)  # 경로를 포함한 이미지 이름
        if img_path_name not in img:
            img.append(img_path_name)
        else:
            img_duplication.append(img_path_name)

# 중복되지 않는 경우, lst1만 18900이라는 값을 출력해야 정상
if len(img) == 18900 and len(img_duplication) == 0:
    print(f"Img is correctly set! \n{len(img)} background subtracted images exist.")

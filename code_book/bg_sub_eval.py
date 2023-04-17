import os

BASE_ROOT = '/opt/ml/input/data/eval/bg_sub'
CORRECT_NUM_FILES = 18900
CORRECT_NUM_DUPLICATE_FILES = 0

img_lst = os.listdir(BASE_ROOT)
print(len(img_lst)) # 12600 Correct!
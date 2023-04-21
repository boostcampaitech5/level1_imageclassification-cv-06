import collections
import os

BASE_ROOT = "/opt/ml/input/data/train/bg_sub/"
CORRECT_NUM_FILES = 18900
CORRECT_NUM_DUPLICATE_FILES = 0

file_counter = collections.Counter()

for root, dirs, files in os.walk(BASE_ROOT):
    for file in files:
        file_path = os.path.join(root, file)
        file_counter[file_path] += 1

num_files = len(file_counter)
num_duplicate_files = len(list(filter(lambda x: x[1] > 1, file_counter.items())))

if num_files == CORRECT_NUM_FILES and num_duplicate_files == CORRECT_NUM_DUPLICATE_FILES:
    print("All Files Correct!")
    print(f"Number of files: {num_files}, Number of duplicate files: {num_duplicate_files}")

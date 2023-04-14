from rembg import remove
from PIL import Image

import os
import time

# rembg
# find . -name '._*' -exec rm {} \;
# pip install rembg[gpu]

BASE_ROOT = '/opt/ml/input/data/train/images/'
SAVE_ROOT = '/opt/ml/input/data/train/bg_sub/'
folder_lst = os.listdir(BASE_ROOT)

start = time.time()
for i in folder_lst:
    img_foler_path = os.path.join(BASE_ROOT, i)
    save_path = os.path.join(SAVE_ROOT, i)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('processing: ', img_foler_path)
    print('save_path: ', save_path)

    img_lst = os.listdir(img_foler_path)
    for j in img_lst:
        input = Image.open(os.path.join(img_foler_path, j))
        output = remove(input)
        output = output.convert("RGB") # jpg는 투명도를 표현할 수 없는 file format이므로, RGBA->RGB로 변환 필요
        output.save(os.path.join(save_path, j))

print("Total: %.5f" %(time.time()-start))

'''
OUTPUT_PATH = './output'
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

start = time.time()
for i in img_lst:
    img_path = os.path.join(BASE_IMG_ROOT, i)
    input = Image.open(img_path)
    output = remove(input)
    output = output.convert("RGB") # jpg는 투명도를 표현할 수 없는 file format이므로, RGBA->RGB로 변환 필요
    output.save("./output/bg_removed_{}".format(i))
    
print('time :', time.time()-start) # time : 9.987520217895508
'''
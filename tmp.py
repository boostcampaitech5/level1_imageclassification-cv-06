from rembg import remove
from PIL import Image

import os
import time

# rembg
# find . -name '._*' -exec rm {} \;
#BASE_ROOT = '/opt/ml/input/data/train/images/'
#img_folder = os.listdir(BASE_ROOT)

BASE_IMG_ROOT = '/opt/ml/input/data/train/images/006106_male_Asian_19'
img_lst = os.listdir(BASE_IMG_ROOT)    
#print(img_lst)

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

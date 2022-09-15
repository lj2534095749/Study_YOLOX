import os
import random
import argparse

import os
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

base_path = str(ROOT / r'datasets\VOCClothes')
test_image_path = base_path + r'\test'
test_txt_path = base_path + r'\VOC2012\ImageSets\Main'

parser = argparse.ArgumentParser()
parser.add_argument('--test_image_path', default=test_image_path, type=str, help='output txt label path')
parser.add_argument('--test_txt_path', default=test_txt_path, type=str, help='output txt label path')
opt = parser.parse_args()

txtsavepath = opt.test_txt_path

total_test_image = os.listdir(test_image_path)
if not os.path.exists(txtsavepath):
    os.makedirs(txtsavepath)

num = len(total_test_image)
list_index = range(num)

file_test = open(txtsavepath + '/test.txt', 'w')

for i in list_index:
    name = total_test_image[i][:-4] + '\n'
    file_test.write(name)

file_test.close()

print("==========================>>> Finished.")

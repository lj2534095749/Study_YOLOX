import shutil
import os

import os
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

base_path = str(ROOT / r'datasets\VOCClothes\VOC2012')
labels_path = base_path + r'\labels'
annotations_path = base_path + r'\Annotations'
segmentation_path = base_path + r'\ImageSets'

file_List = ["train", "val", "test"]
for file in file_List:
    if not os.path.exists(base_path + '/images/%s' % file):
        os.makedirs(base_path + '/images/%s' % file)
    if not os.path.exists(base_path + '/labels/%s' % file):
        os.makedirs(base_path + '/labels/%s' % file)

    f = open(base_path + '/%s.txt' % file, 'r')
    lines = f.readlines()
    for line in lines:
        line = "/".join(line.split('/')[-5:]).strip()
        shutil.copy(line, base_path + "/images/%s" % file)
        line = line.replace('JPEGImages', 'labels')
        line = line.replace('jpg', 'txt')
        shutil.copy(line, base_path + "/labels/%s" % file)

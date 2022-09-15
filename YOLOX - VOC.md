## 0. 环境准备

**1. 下载YOLOX**

```cmd
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
```

**2. 下载Pytorch**

```cmd
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

**3. 下载requirements.txt（先删除pytorch依赖）**

```cmd
pip install -r requirements.txt
```

**4. 下载pycocotools**

```cmd
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pycocotools
```

**5. 初始模型下载**

> https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0

## 1. VOC数据集设置

### 1.1. 文件格式设置

**源码位置：YOLOX/yolox/data/datasets/voc.py**

```
- YOLOX
    - datasets
        - VOC2012
            - Annotations
                - xxx.xml
            - JPEGImages
                - xxx.jpg
            - ImageSets
                - Main
                    - train.txt
                    - val.txt
```

**注意（修改源码，否则报错“找不到文件”）：**

```python
# 源码为 annopath = os.path.join(rootpath, "Annotations", "{:s}.xml")
# 但是 {:s} 存在时，os.path.join 会忽略在其之前的路径
# 字符串前加 r 表示将转义字符当成普通字符
annopath = os.path.join(rootpath, r"Annotations/{:s}.xml")
```

### 1.2. 修改class

**源码位置：YOLOX/yolox/data/datasets/voc_classes.py**

```python
# 改为VOC数据集的标签，格式如下
VOC_CLASSES = (
    'missing_hole',
    'mouse_bite',
    'open_circuit',
    'short',
    'spur',
    'spurious_copper',
)
```

### 1.3. 修改class个数

**源码位置：YOLOX/exps/example/yolox_voc/yolox_voc_s_me.py（复制的同目录下的yolox_voc_s.py文件）**

```python
# 改为VOC数据集的标签格式，格式如下
self.num_classes = 6
```

### 1.4. 修改配置文件的数据集文件位置

**源码位置：YOLOX/exps/example/yolox_voc/yolox_voc_s_me.py**

**1. def get_data_loader()**

```python
dataset = VOCDetection(
    data_dir=os.path.join(get_yolox_datadir(), "VOCdevkit"),
    image_sets=[('2012', 'train')],  # TODO
    img_size=self.input_size,
    preproc=TrainTransform(
        max_labels=50,
        flip_prob=self.flip_prob,
        hsv_prob=self.hsv_prob),
    cache=cache_img,
)
```

**2. def get_eval_loader()**

```python
valdataset = VOCDetection(
    data_dir=os.path.join(get_yolox_datadir(), "VOCdevkit"),
    image_sets=[('2012', 'val')],  # TODO
    img_size=self.test_size,
    preproc=ValTransform(legacy=legacy),
)
```

## 2. 训练设置

### 2.1. 修改配置

**源码位置：YOLOX/exps/example/yolox_voc/yolox_voc_s_me.py**

```python
# 在 def __init__(self) 增加以下配置
self.max_epoch = 120  # 总轮数
self.print_interval = 10  # 每几个 iter 打印一次
self.eval_interval = 5  # 每几个 epoch 打印一次
self.data_num_workers = 1
```

### 2.2 修改参数

**源码位置：YOLOX/train.py（从YOLOX/tools/train.py复制）**

```python
# 批大小
parser.add_argument("-b", "--batch-size", type=int, default=4, help="batch size")
# 使用设备
parser.add_argument("-d", "--devices", default=0, type=int, help="device for training")
# 配置文件
parser.add_argument("-f", "--exp_file", default=r'exps\example\yolox_voc\yolox_voc_s_me.py', type=str, help="plz input your experiment description file",)
# 断点续训
parser.add_argument("--resume", default=False, action="store_true", help="resume training")
# 初始模型
parser.add_argument("-c", "--ckpt", default=r'yolox_s.pth', type=str, help="checkpoint file")
# 断点续训开始轮数
parser.add_argument(
    "-e",
    "--start_epoch",
    default=None,
    type=int,
    help="resume training start epoch",
)
```

## 3. 检测设置

**源码位置：YOLOX/demo.py（从YOLOX/tools/demo.py复制）**

**1. def make_parser()**

```python
# --demo
parser.add_argument("--demo", default="image", help="demo type, eg. image, video and webcam")
# 配置文件
parser.add_argument("-f", "--exp_file", default=r'exps\example\yolox_voc\yolox_voc_s_me.py', type=str, help="plz input your experiment description file",)
# 要检测的图片或图片在的文件夹
parser.add_argument("--path", default="datasets/test/VOC2012/val", help="path to images or video")
# 如果要保存检测结果，需要设置默认值为True
parser.add_argument(
    "--save_result",
    action="store_true",
    default=True,
    help="whether to save the inference result of image/video",
)
```

**2. class Predictor**

**先在 yolox/data/datasets/__init__.py 添加 from .voc_classes import VOC_CLASSES**

```python
from yolox.data.datasets import VOC_CLASSES

def __init__(
    self,
    model,
    exp,
    cls_names=VOC_CLASSES,
    trt_file=None,
    decoder=None,
    device="cpu",
    fp16=False,
    legacy=False,
):
```

**3. def main()**

```python
predictor = Predictor(
    model, exp, VOC_CLASSES, trt_file, decoder,
    args.device, args.fp16, args.legacy,
)
```
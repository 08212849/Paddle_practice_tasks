import pandas as pd
import math
import numpy as np
import os
import collections
from functools import partial
import random
import time
import inspect
import importlib
from tqdm import tqdm
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import IterableDataset
from paddle.utils.download import get_path_from_url
# 导入paddlenlp所需的相关包
import paddlenlp as ppnlp
from paddlenlp.data import JiebaTokenizer, Pad, Stack, Tuple, Vocab
from paddlenlp.datasets import MapDataset
from paddle.dataset.common import md5file
from paddlenlp.datasets import DatasetBuilder


train = pd.read_table('data/Train.txt', sep='\t',header=None)  # 训练集
test = pd.read_table('data/Test.txt', sep='\t',header=None)    # 测试集

print(f"train数据集长度： {len(train)}\t  test数据集长度{len(test)}")

# 添加列名便于对数据进行更好处理
train.columns = ["id","label",'text_a']
test.columns = ["text_a"]

def drawData():
    # 绘制分布图
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    #指定默认字体
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['font.family']='sans-serif'
    #解决负号'-'显示为方块的问题
    mpl.rcParams['axes.unicode_minus'] = False
    train['label'].value_counts().plot.bar()
    plt.show()
    plt.savefig("result/data.png")

    # 打印train_text字数最大最小值
    print(max(train['text_a'].str.len()))
    print(min(train['text_a'].str.len()))

    # 打印test_text字数最大最小值
    print(max(test['text_a'].str.len()))
    print(min(test['text_a'].str.len()))

    # 查看数据分布
    print(f"train处理后数据集长度： {len(train)}")
    # print(train['text_a'].map(len).describe())
    # print(test['text_a'].map(len).describe())
    
drawData()

# 过滤不含汉字的训练数据
# train['text_a'] = train['text_a'].astype(str)
# contains_chinese = train['text_a'].str.contains(r'[\u4e00-\u9fff]', na=False)
# train = train[contains_chinese]


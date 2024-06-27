import numpy as np
from functools import partial
import paddle
import paddlehub as hub
import paddle.nn as nn
from paddle.io import Dataset
from paddlenlp.datasets import MapDataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlehub.datasets.base_nlp_dataset import InputExample, TextClassificationDataset

n_epochs = 4
batch_size = 64
max_seq_length = 128
n_classes = 14
DATA_DIR="./data"

class ThuNews(TextClassificationDataset):
    def __init__(self, tokenizer, mode='train', max_seq_len=128):
        if mode == 'train':
            data_file = 'Train.txt'
        elif mode == 'test':
            data_file = 'test.txt'
        else:
            data_file = 'valid.txt'
        super(ThuNews, self).__init__(
            base_path=DATA_DIR,
            data_file=data_file,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            mode=mode,
            is_file_with_header=True,
            label_list=['财经','彩票','房产','股票','家居','教育','科技','社会','时尚','时政','体育','星座','游戏','娱乐']
        )

    
    # 加载数据集
    def _read_file(self, input_file, is_file_with_header: bool = False):
        with open(input_file, "r", encoding="utf-8") as f:
            examples = []
            seq_id = 0
            for line in f.readlines():
                if len(line.strip().split("\t", maxsplit=3)) != 3:
                        continue
                id, label, text = line.strip().split("\t", maxsplit=3)
                example = InputExample(guid=seq_id, text_a=text, label=label)
                seq_id += 1
                examples.append(example)
            return examples


model = hub.Module(name="chinese-bert-wwm", task='seq-cls', num_classes=14) 

optimizer = paddle.optimizer.Adam(learning_rate=5e-5, 
                                  parameters=model.parameters())  # 优化器的选择和参数配置
trainer = hub.Trainer(model, optimizer, 
                      checkpoint_dir='./ckpt', use_gpu=True)        # fine-tune任务的执行者

train_dataset = ThuNews(model.get_tokenizer(), mode='train', max_seq_len=128)

trainer.train(train_dataset, epochs=1, 
              batch_size=64, save_interval=1)   # 配置训练参数，启动训练，并指定验证集


# Data to be prdicted
data = []
data_path = './data/Test.txt'
with open(data_path, "r", encoding="utf-8") as f:
    for line in f.readlines():
        text = line.strip().split("\t", maxsplit=1)
        data.append(text)

label_list=['财经','彩票','房产','股票','家居','教育','科技','社会','时尚','时政','体育','星座','游戏','娱乐']
label_map = { 
    idx: label_text for idx, label_text in enumerate(label_list)
}

model = hub.Module(
    name='chinese-bert-wwm',
    task='seq-cls',
    load_checkpoint='./ckpt/best_model/model.pdparams',
    label_map=label_map)
results = model.predict(data, max_seq_len=128, batch_size=64, use_gpu=True)

with open('resulthub.txt', 'w', encoding='utf-8') as file:
    for idx, text in enumerate(data):
        print('Data: {} \t Lable: {}'.format(text[0], results[idx]))
        file.write(results[idx] + '\n')
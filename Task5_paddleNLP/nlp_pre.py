import numpy as np
from functools import partial
import paddle
import paddle.nn as nn
from paddle.io import Dataset
import paddle.nn.functional as F
import paddlenlp
from paddlenlp.datasets import MapDataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import LinearDecayWithWarmup


def convert_example(example, tokenizer, max_seq_length=128, is_test=False):
    encoded_inputs = tokenizer(text=example["text"], max_seq_len=max_seq_length)
    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        return input_ids, token_type_ids
    
def create_dataloader(dataset,
                      mode='train',
                      batch_size=128,
                      batchify_fn=None,
                      trans_fn=None):
    # trans_fn对应前边的covert_example函数，使用该函数处理每个样本为期望的格式
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)

    # 调用paddle.io.DataLoader来构建DataLoader
    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)

class ErnieForSequenceClassification(paddle.nn.Layer):
    def __init__(self, MODEL_NAME, num_class=14, dropout=None):
        super(ErnieForSequenceClassification, self).__init__()
        # 加载预训练好的ernie，只需要指定一个名字就可以
        self.ernie = paddlenlp.transformers.ErnieModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(dropout if dropout is not None else self.ernie.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.ernie.config["hidden_size"], num_class)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        _, pooled_output = self.ernie(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class THUCDataSet(Dataset):
    def __init__(self, data_path, isTrain=True):
        # 加载数据集
        self.isTrain = isTrain
        self.data = self._load_data(data_path)
        
    # 加载数据集
    def _load_data(self, data_path):
        data_set = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                if self.isTrain:
                    if len(line.strip().split("\t", maxsplit=3)) != 3:
                        continue
                    id, label, text = line.strip().split("\t", maxsplit=3)
                    example = {"text":text, "label": id}
                    data_set.append(example)
                else:
                    text = line.strip().split("\t", maxsplit=1)
                    example = {"text":text}
                    data_set.append(example)
        return data_set

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
    
# 超参设置
n_classes=14
dropout_rate = None

MODEL_NAME = "ernie-1.0"
# 检测是否可以使用GPU，如果可以优先使用GPU
use_gpu = True if paddle.get_device().startswith("gpu") else False
if use_gpu:
    paddle.set_device('gpu:0')

# 加载预训练模型ERNIE
# 加载用于文本分类的fune-tuning网络
model =  ErnieForSequenceClassification(MODEL_NAME, num_class=n_classes, dropout=dropout_rate)
model_name = "ernie_for_sequence_classification"

# 加载模型参数
model_dict = paddle.load("model/{}1.pdparams".format(model_name))
model.set_dict(model_dict)

# 加载优化器参数
optimizer_dict = paddle.load("model/{}1.optparams".format(model_name))
# optimizer.set_dict(optimizer_dict)

from paddlenlp.transformers import BertTokenizer
# 加载分词器
tokenizer = BertTokenizer.from_pretrained('./model/tokenizer1')
model.eval()

# 定义模型预测函数
def predict1(model, data, tokenizer, id2label, batch_size=312):
    examples = []
    # 将输入数据（list格式）处理为模型可接受的格式
    for text in data:
        input_ids, segment_ids = convert_example(
            text,
            tokenizer,
            max_seq_length=81,
            is_test=True)
        examples.append((input_ids, segment_ids))

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input id
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # segment id
    ): fn(samples)

    # Seperates data into some batches.
    batches = []
    one_batch = []
    for example in examples:
        one_batch.append(example)
        if len(one_batch) == batch_size:
            batches.append(one_batch)
            one_batch = []
    if one_batch:
        # The last batch whose size is less than the config batch_size setting.
        batches.append(one_batch)

    results = []
    model.eval()
    temp = 0
    for batch in batches:
        temp += 1
        input_ids, segment_ids = batchify_fn(batch)
        input_ids = paddle.to_tensor(input_ids)
        segment_ids = paddle.to_tensor(segment_ids)
        logits = model(input_ids, segment_ids)
        probs = F.softmax(logits, axis=1)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        labels = [id2label[i] for i in idx]
        results.extend(labels)
        if temp % 10 == 0:
            print('batch: ',temp,'number: ', temp*batch_size)
    return results  # 返回预测结果


def predict(data, id2label, batch_size=32):
    results = []
    model.eval()
    temp = 0
    for text in data:
        temp += 1
        input_ids, segment_ids = convert_example(
            text,
            tokenizer,
            max_seq_length=81,
            is_test=True)
        input_ids = paddle.to_tensor(input_ids)
        segment_ids = paddle.to_tensor(segment_ids)
        logits = model(input_ids, segment_ids)
        probs = F.softmax(logits, axis=1)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        labels = [id2label[i] for i in idx]
        results.extend(labels)
        if temp % 1000 == 0:
            print(temp)
    return results

test_set = THUCDataSet("./data/Test.txt",isTrain=False)
Testdata = test_set._load_data("Test.txt")
print(len(Testdata))

label_list=['财经','彩票','房产','股票','家居','教育','科技','社会','时尚','时政','体育','星座','游戏','娱乐']
# train_ds = paddlenlp.datasets.load_dataset('thucnews', splits=['train'])
# label_list = train_ds.label_list
label_dict={}
for i in range(0,len(label_list)):
    label_dict[i]=label_list[i]

results = predict(Testdata, label_dict)
# 对测试集进行预测
results = predict1(model, Testdata, tokenizer, label_dict, batch_size=128) 


with open('result2.txt', 'w', encoding='utf-8') as file:
    for result in results:
        # 写入每行数据，并添加换行符
        file.write(result + '\n')
        
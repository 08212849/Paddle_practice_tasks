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

MODEL_NAME = "ernie-1.0"
tokenizer = paddlenlp.transformers.ErnieTokenizer.from_pretrained(MODEL_NAME)
# MODEL_NAME = "roberta-wwm-ext-large"
# 超参设置
n_epochs = 4
batch_size = 312
max_seq_length = 128
n_classes = 14
dropout_rate = None

learning_rate = 5e-5
warmup_proportion = 0.1
weight_decay = 0.01

def convert_example(example, tokenizer, max_seq_length=128, is_test=False):
    encoded_inputs = tokenizer(text=example["text"], max_seq_len=max_seq_length)
    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]
    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        return input_ids, token_type_ids
trans_func = partial(convert_example, tokenizer=tokenizer, max_seq_length=max_seq_length)

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


batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),      # input
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id), # segment
    Stack(dtype="int64")
): [data for data in fn(samples)]

# 加载数据集，构造DataLoader
train_set = THUCDataSet("./data/Train.txt")
train_set = MapDataset(train_set)
train_data_loader = create_dataloader(train_set, mode="train", batch_size=batch_size, batchify_fn=batchify_fn, trans_fn=trans_func)


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




# 检测是否可以使用GPU，如果可以优先使用GPU
use_gpu = True if paddle.get_device().startswith("gpu") else False
if use_gpu:
    paddle.set_device('gpu:0')

# 加载预训练模型ERNIE
# 加载用于文本分类的fune-tuning网络
model =  ErnieForSequenceClassification(MODEL_NAME, num_class=n_classes, dropout=dropout_rate)

# 设置优化器
num_training_steps = len(train_data_loader) * n_epochs
lr_scheduler = LinearDecayWithWarmup(learning_rate, num_training_steps, warmup_proportion)
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    weight_decay=weight_decay,
    apply_decay_param_fun=lambda x: x in [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ])

# 定义统计指标
metric = paddle.metric.Accuracy()

def evaluate(model, metric, data_loader):
    model.eval()
    # 每次使用测试集进行评估时，先重置掉之前的metric的累计数据，保证只是针对本次评估。
    metric.reset()
    losses = []
    for batch in data_loader:
        # 获取数据
        input_ids, segment_ids, labels = batch
        # 执行前向计算
        logits = model(input_ids, segment_ids)
        # 计算损失
        loss = F.cross_entropy(input=logits, label=labels)
        loss= paddle.mean(loss)
        losses.append(loss.numpy())
        # 统计准确率指标
        correct = metric.compute(logits, labels)
        metric.update(correct)
        accu = metric.accumulate()
    print("eval loss: %.5f, accu: %.5f" % (np.mean(losses), accu))
    metric.reset()

def train(model):
    global_step=0
    for epoch in range(1, n_epochs+1):
        model.train()
        for step, batch in enumerate(train_data_loader, start=1):
            # 获取数据
            input_ids, segment_ids, labels = batch
            # 模型前向计算
            logits = model(input_ids, segment_ids)
            loss = F.cross_entropy(input=logits, label=labels)
            loss = paddle.mean(loss)

            # 统计指标
            # probs = F.softmax(logits, axis=1)
            # correct = metric.compute(probs, labels)
            # metric.update(correct)
            # acc = metric.accumulate()
            
            # 打印中间训练结果
            global_step += 1
            if global_step % 500 == 0 :
                # print("global step %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f" % (global_step, epoch, step, loss, acc))
                print("global step %d, epoch: %d, batch: %d, loss: %.5f" % (global_step, epoch, step, loss))
            
            # 参数更新
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
        
        # 模型评估
        # evaluate(model, metric, test_data_loader)
            
train(model)
# # 模型保存的名称
model_name = "ernie_for_sequence_classification"

paddle.save(model.state_dict(), "model/{}1.pdparams".format(model_name))
paddle.save(optimizer.state_dict(), "model/{}1.optparams".format(model_name))
tokenizer.save_pretrained('./model/tokenizer1')

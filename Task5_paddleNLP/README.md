# 文本分类任务

> 兼容版本：
>
> GPU							1080Ti
>
> cuda							11.6
>
> paddlehub  				2.4.0
>
> paddlenlp   				2.8.0
>
> paddlepaddle    		2.6.1
>
> paddlepaddle-gpu    2.6.1.post116

尝试paddlenlp和paddlehub框架，分别测试ernie、chinese_roberta_wwm_ext、chinese_bert_wwm模型上。

- paddlenlp在装载数据集上快于paddlehub

## paddleNLP

### 一、数据处理

- 构造ThucNews类，加载

  训练集格式如：`{“text": "天气真好", "label":"0"}`

  测试集格式如：`{“text": "天气真好"}`

- 加载对应的BertTokenizer，paddlenlp内置的 `paddlenlp.datasets.MapDataset` 的 `map()` 方法支持传入一个函数，对数据集内的数据进行统一转换。

  转换后一条数据格式如：

  ```
  {'input_ids': [101, 1599, 3614, 2802, 5074, 4413, 4638, 4511, 4495,
                 1599, 3614, 784, 720, 3416, 4638, 1957, 4495, 102],
   'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 
   					1, 1, 1, 1, 1, 1, 1, 1, 1],
   'label': [1]
   }
  ```

​		处理后的单条数据是一个**字典** ，包含 `input_ids` ， `token_type_ids` 和 `label` 三个key。

- `input_ids` 和 `token_type_ids` 进行 **padding** 操作， `label` **stack** 后传入loss function。
- 组装每个batch的数据。

```python
MODEL_NAME = "ernie-1.0"
tokenizer = paddlenlp.transformers.ErnieTokenizer.from_pretrained(MODEL_NAME)

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
```

### 二、Transformer预训练模型

- 加载预训练模型

- 加载用于下游任务的fune-tuning网络

- 设置适用于ERNIE这类Transformer模型的动态学习率和损失函数、优化算法、评价指标等。

- 加载数据集，使用 `paddle.io.DataLoader()` 接口多线程异步加载并开始训练

- 保存tokenizer、optimizer、model参数

  <img src="https://qinglan-1324038201.cos.ap-nanjing.myqcloud.com/images/202406261804596.png" alt="image-20240614093323724" style="zoom:50%;" />

```python
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
```

### 三、模型预测

```python
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
        # if temp % 1000 == 0:
        #    print(temp)
    return results
```

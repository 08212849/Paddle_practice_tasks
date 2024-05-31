# Task2 鲍鱼年龄预测-线性回归

使用PaddlePaddle建立起一个鲍鱼年龄线性回归预测模型。数据集共4177行，每行9列，前8列用来描述鲍鱼的各种信息，分别是性别、长度、直径、高度、总重量、皮重、内脏重量、克重，最后一列为该鲍鱼的年龄。

实验环境：windows 11 Python-3.8.9  paddle-2.2.2  matplotlib-3.7.2 numpy-1.24.4

主函数如下：

<img src="https://qinglan-1324038201.cos.ap-nanjing.myqcloud.com/images/202406010215116.png" alt="image-20240601010119423" style="zoom: 67%;" />

## step1.数据预处理
```
def data_process(path): ...
```

归一化处理为解决数据指标之间的可比性。各指标处于[0,1]之间的小数，适合进行综合对比评价。因为数据较为稳定，不存在极端的最大最小值，因此归一化要好于标准化。

## step2.定义线性模型

```
class Regressor(paddle.nn.Layer): ...
```

定义一层全连接层，输出维度是1，激活函数为None，即不使用激活函数

<img src="https://qinglan-1324038201.cos.ap-nanjing.myqcloud.com/images/202406010215468.png" alt="img" style="zoom: 67%;" />

## step3.模型训练

```
def train(model): ...
```

1. **初始化优化器**：使用SGD（随机梯度下降）优化器或者其他，学习率为0.0001，并配置模型参数。
2. **训练循环**：
   - 遍历从0到`EPOCH_NUM`的每个epoch。
   - 在每个epoch开始前，使用`np.random.shuffle(train_data)`将训练数据随机打乱，以提高模型的泛化能力。
   - 将打乱后的数据分割为多个mini-batch，每个batch包含`BATCH_SIZE`个样本。
3. **处理每个mini-batch**：
   - 从每个mini-batch中提取特征和标签，转换为NumPy数组，并指定数据类型为`np.float32`，将NumPy数组转换为PaddlePaddle的`Tensor`对象。
   - 执行前向传播计算，`y_pred = model(features)`通过模型计算预测结果。
   - 使用均方误差损失函数`F.mse_loss`计算预测值和实际标签之间的损失。
4. **反向传播和优化**：
   - 调用`cost.backward()`进行反向传播，计算损失相对于模型参数的梯度。
   - 使用`optimizer.step()`根据梯度更新模型参数。
   - 调用`optimizer.clear_grad()`清除已存储的梯度，为下一次迭代做准备。

## step4.绘制训练过程中Loss变化图像

```
def draw_train_process(iters, train_costs):...
```

iter表示使用训练的样本次数，最大为3341（总训练样本数）*100（epoch个数），cost记录MSE loss变化。

右图为使用SGD优化器，左图为使用Momentum优化器。

<img src="https://qinglan-1324038201.cos.ap-nanjing.myqcloud.com/images/202406010215117.png" alt="image-20240601013628780" style="zoom:67%;" /><img src="https://qinglan-1324038201.cos.ap-nanjing.myqcloud.com/images/202406010215118.png" alt="image-20240601015512833" style="zoom: 67%;" />

## step5.结果预测

```
def print_Predict():...
```

<img src="https://qinglan-1324038201.cos.ap-nanjing.myqcloud.com/images/202406010215119.png" alt="image-20240601021212437" style="zoom: 50%;" />

## step6.绘制预测结果
```
def draw_infer_result(groud_truths, infer_results):
```

<img src="https://qinglan-1324038201.cos.ap-nanjing.myqcloud.com/images/202406010215120.png" alt="image-20240601014638876" style="zoom:67%;" />
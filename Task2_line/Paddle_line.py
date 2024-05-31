import paddle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import paddle.nn.functional as F
print(paddle.__version__)
print(np.__version__)
print(matplotlib.__version__)

# 数据预处理
def data_process(path):
    data_X = []
    data_Y = []
    sex_map = {'I': 0, 'M': 1, 'F': 2}
    with open(path) as f:
        for line in f.readlines():
            line = line.split(',')
            line[0] = sex_map[line[0]]
            data_X.append(line[:-1])
            data_Y.append(line[-1:])
    data_X = np.array(data_X, dtype='float32')
    data_Y = np.array(data_Y, dtype='float32')

    # 归一化
    for i in range(data_X.shape[1]):
        _min = np.min(data_X[:, i])  # 每一列的最小值
        _max = np.max(data_X[:, i])  # 每一列的最大值
        data_X[:, i] = (data_X[:, i] - _min) / (_max - _min)  # 归一化到0-1之间

    X_train, X_test, y_train, y_test = train_test_split(data_X,  # 被划分的样本特征集
                                                        data_Y,  # 被划分的样本标签
                                                        test_size=0.2,  # 测试集占比
                                                        random_state=1)  # 随机数种子，在需要重复试验的时候，保证得到一组一样的随机数
    train_data = np.concatenate((X_train, y_train), axis=1)
    test_data = np.concatenate((X_test, y_test), axis=1)
    return train_data, test_data

# 网络搭建
class Regressor(paddle.nn.Layer):
    def __init__(self):
        super(Regressor, self).__init__()
        # 定义一层全连接层，输出维度是1，激活函数为None，即不使用激活函数
        self.fc = paddle.nn.Linear(16, 1)

    # 网络的前向计算函数
    # def forward(self, inputs):
    #     pred = self.fc(inputs)
    #     return pred
    def forward(self, inputs):
        # 根据需要修改输入特征，添加平方项
        # 假设 inputs 是形状为 [batch_size, 1] 的张量
        # 我们需要将其转换为 [batch_size, 2]，其中包含原始特征和它的平方项
        inputs_squared = inputs ** 2
        combined_inputs = paddle.concat([inputs, inputs_squared], axis=1)
        # 现在 combined_inputs 包含原始特征和平方项，可以作为全连接层的输入
        pred = self.fc(combined_inputs)
        return pred

# 绘制训练过程的损失值变化趋势
def draw_train_process(iters, train_costs):
    title = "training cost"
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=14)
    plt.ylabel("cost", fontsize=14)
    plt.plot(iters, train_costs, color='red', label='training cost')
    plt.grid()
    # plt.show()
    plt.savefig('training cost2.png')
    plt.clf()

# 网络训练
train_nums = []
train_costs = []
def train(model):
    BATCH_SIZE = 50
    print('start training ... ')
    # 开启模型训练模式
    model.train()
    EPOCH_NUM = 100
    train_num = 0
    optimizer = paddle.optimizer.SGD(learning_rate=0.0001, parameters=model.parameters())
    for epoch_id in range(EPOCH_NUM):
        # 在每轮迭代开始之前，将训练数据的顺序随机的打乱
        np.random.shuffle(train_data)
        # 将训练数据进行拆分，每个batch包含50条数据
        mini_batches = [train_data[k: k+BATCH_SIZE] for k in range(0, len(train_data), BATCH_SIZE)]
        for batch_id, data in enumerate(mini_batches):
            features_np = np.array(data[:, :8], np.float32)
            labels_np = np.array(data[:, -1:], np.float32)

            features = paddle.to_tensor(features_np)
            labels = paddle.to_tensor(labels_np)

            # 前向计算
            y_pred = model(features)
            cost = F.mse_loss(y_pred, label=labels)
            train_cost = cost.numpy()[0]
            # 反向传播
            cost.backward()
            # 最小化loss，更新参数
            optimizer.step()
            # 清除梯度
            optimizer.clear_grad()
            if batch_id % 30 == 0 and epoch_id % 50 == 0:
                print("Pass:%d,Cost:%0.5f" %(epoch_id, train_cost))

            train_num = train_num + BATCH_SIZE
            train_nums.append(train_num)
            train_costs.append(train_cost)

# 进行预测
def print_Predict():
    INFER_BATCH_SIZE = 15

    infer_features_np = np.array([data[:8] for data in test_data]).astype("float32")
    infer_labels_np = np.array([data[-1] for data in test_data]).astype("float32")

    infer_features = paddle.to_tensor(infer_features_np)
    infer_labels = infer_labels_np
    fetch_list = model(infer_features)
    sum_cost = 0
    infer_results = []    # 预测值
    ground_truths = []     # 真实值
    for i in range(INFER_BATCH_SIZE):
        infer_result = fetch_list[i]
        ground_truth = infer_labels[i]
        infer_results.append(infer_result)   # 预测值
        ground_truths.append(ground_truth)    # 真实值
        print("No.%d: infer result is %.2f,ground truth is %.2f" % (i, infer_result, ground_truth))
        cost = paddle.pow(infer_result - ground_truth, 2)
        sum_cost += cost
    mean_loss = sum_cost / INFER_BATCH_SIZE
    print("Mean loss is:", mean_loss.numpy())
    return infer_results, ground_truths

# 绘制预测结果
def draw_infer_result(groud_truths,infer_results):
    title='abalone'
    plt.title(title, fontsize=24)
    x = np.arange(1,20)
    y = x
    plt.plot(x, y)
    plt.xlabel('ground truth', fontsize=14)
    plt.ylabel('infer result', fontsize=14)
    plt.scatter(np.array(groud_truths).astype("float32"), np.array(infer_results).astype("float32"),color='green',label='training cost')
    plt.grid()
    plt.savefig('abalone result2.png')
    plt.clf()

if __name__ == '__main__':
    data_path = 'data/AbaloneAgePrediction.txt'
    # step1.数据预处理
    train_data, test_data = data_process(data_path)
    # step2.定义线性模型
    model = Regressor()
    # step3.模型训练
    train(model)
    # step4.绘制训练过程中Loss变化图像
    draw_train_process(train_nums, train_costs)
    # step5.结果预测
    infer_results, groud_truths = print_Predict()
    # step6.绘制预测结果
    draw_infer_result(groud_truths, infer_results)
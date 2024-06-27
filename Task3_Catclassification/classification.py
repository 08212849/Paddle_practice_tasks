import paddle
print(paddle.__version__)
import paddlehub as hub
#去官网选择模型 https://www.paddlepaddle.org.cn/hublist?filter=en_category&value=ImageClassification
module = hub.Module(name="vgg19_imagenet")


from paddlehub.dataset.base_cv_dataset import BaseCVDataset
class DemoDataset(BaseCVDataset):
   def __init__(self):
       # 数据集存放位置
       self.dataset_dir = "/home/aistudio"
       super(DemoDataset, self).__init__(
           base_path=self.dataset_dir,
           train_list_file="train_split_list.txt",
           validate_list_file="val_split_list.txt",
           test_list_file="test_split_list.txt",
           label_list_file="label_list.txt",
           )
dataset = DemoDataset()

data_reader = hub.reader.ImageClassificationReader(
    image_width=module.get_expected_image_width(),
    image_height=module.get_expected_image_height(),
    images_mean=module.get_pretrained_images_mean(),
    images_std=module.get_pretrained_images_std(),
    dataset=dataset)


config = hub.RunConfig(
    use_cuda=False,                              #是否使用GPU训练，默认为False；
    num_epoch=5,                                #Fine-tune的轮数；
    checkpoint_dir="cv_finetune_turtorial_demo",#模型checkpoint保存路径, 若用户没有指定，程序会自动生成；
    batch_size=124,                              #训练的批大小，如果使用GPU，请根据实际情况调整batch_size；
    eval_interval=100,                           #模型评估的间隔，默认每100个step评估一次验证集；
    strategy=hub.finetune.strategy.DefaultFinetuneStrategy())
    #Fine-tune优化策略；如下几种：
    # hub.finetune.strategy.AdamWeightDecayStrategy()
    # hub.finetune.strategy.DefaultFinetuneStrategy()
    # hub.finetune.strategy.L2SPFinetuneStrategy()
    # hub.finetune.strategy.ULMFiTStrategy()

input_dict, output_dict, program = module.context(trainable=True)
img = input_dict["image"]
feature_map = output_dict["feature_map"]
feed_list = [img.name]
# print(img)

task = hub.ImageClassifierTask(
    data_reader=data_reader,
    feed_list=feed_list,
    feature=feature_map,
    num_classes=dataset.num_labels,
    config=config)

run_states = task.finetune_and_eval()

# 准备预测数据list
predict_data = []
with open('predict_list.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        img_path= line.strip('\n')
        predict_data.append(img_path)

# 预测
import numpy as np
import pandas as pd
run_states = task.predict(data=predict_data)

#获得预测结果之后，作进一步的处理
results = [run_state.run_results for run_state in run_states]
# print(np.array(results[0]).shape)
    #返回结果按上面设置的batch_size封装为2份，每一份的shape均为(1,N,C),
    # 如上面的第一份(1,124,12),即表示batch_size=124,类别为12类
csvs = []
#标签字典表
label_map = dataset.label_dict()
index = 0
for batch_result in results:
    #把每个批次的array按第2维进行压缩，即压缩类别，返回最大值的角标
    batch_result = np.argmax(batch_result, axis=2)[0]
    # print(batch_result.shape)
    for result in batch_result:
        result = label_map[result]
        csvs.append((predict_data[index].split('/')[-1],result))
        index+=1

#保存结果
with open('result.csv','w') as f_result:
        for i in range(len(csvs)):
            f_result.write(csvs[i][0] + ',' + str(csvs[i][1]) + '\n')
import os
import shutil
# 原始CSV文件路径
input_csv_path = '../../result.csv'
# 新的CSV文件路径
# output_csv_path = '../result/newresult.csv'

# 数据集目录
dataset_dir = 'cat_12_test'
# 结果保存目录
save_dir = 'classified_train_images'

# 确保保存目录存在
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 读取CSV文件并分类保存图片
with open(input_csv_path, 'r', encoding='utf-8') as file:
    for line in file:
        file_name, class_name = line.split(',')
        class_name = class_name.replace("\n", "")
        print(file_name, class_name)
        # 构建图片的完整路径
        file_path = os.path.join(dataset_dir, file_name)
        print(file_path)
        # 构建目标保存路径
        class_dir = os.path.join(save_dir, class_name)
        print(class_dir)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        # 构建目标文件的完整路径
        destination = os.path.join(class_dir, file_name)
        # 复制图片到目标文件夹
        shutil.copy(file_path, class_dir)

print("图片分类保存完成。")



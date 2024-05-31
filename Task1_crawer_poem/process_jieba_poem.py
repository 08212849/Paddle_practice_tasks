import os
from collections import Counter
import numpy as np
from wordcloud import WordCloud
from PIL import Image
import jieba
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

txtfile_path = 'data/all_poem.txt'
filter_poem_path = 'data/filter_poem.txt'

stopwords = ["，", "。", "不", "而", "第", "何", "乎", "乃", "其",
             "且", "若", "于", "与", "也", "则", "者", "之", "无",
             "有", "来", "一", "中", "时", "上", "为", "自", "如",
             "此", "去", "下", "得", "多", "是", "子", "三", "已",
             "我", "在", "谁", "还", "亦", "既", "\n"]

# 提取所有诗内容到“all_poem.txt”文件中
def write_poem():
    if os.path.exists(txtfile_path) == False:
        # 初始化汇总文件，准备写入
        with open('data/all_poem.txt', 'w', encoding='utf-8') as outfile:
            # 遍历1到500的数字，构造文件名
                for i in range(1, 501):
                    filename = f'roll_{i}.txt'
                    # 打开每个文件并读取内容
                    with open(filename, 'r', encoding='utf-8') as infile:
                        for line in infile:
                            # 去除每行首尾的空白字符（包括换行符）
                            stripped_line = line.strip()

                            # 检查行是否包含特定符号或字数是否小于10
                            if '【' in stripped_line or len(stripped_line) < 10:
                                continue

                            # 忽略"--"与空格间的文字
                            poems = stripped_line.split()
                            for poem in poems:
                                if poem[:2] == "--": continue
                                if poem[-1:] != "，":
                                    poem = poem[:poem.rfind('。')+1]
                                outfile.write(poem + '\n')

# jieba分词,将分词结果保存至“data/filtered_poem.txt”文件，返回word_counts计数器
def write_filtered_poem():
    if os.path.exists(filter_poem_path):
    # 文件存在，读取分词文件
        with open(filter_poem_path, 'r', encoding='utf-8') as file:
            word_counts = []
            for line in file:
                word_counts.append(line.strip())
        word_counts = Counter(word_counts)
    else:
        # 文件不存在，执行分词
        # 读取全唐诗文本
        with open('data/all_poem.txt', 'r', encoding='utf-8') as file:
            all_poem_text = file.read()
        # 使用jieba进行分词
        words = jieba.lcut(all_poem_text)
        # 过滤停用词
        filtered_words = [word for word in words if word not in stopwords]
        word_counts = Counter(filtered_words)

        with open('data/filtered_poem.txt', 'w', encoding='utf-8') as file:
            for word in filtered_words:
                file.write(word + '\n')

    # 计算词组最大长度, 7
    # maxLenth = 0
    # for i in word_counts:
    #     maxLenth = max(len(i), maxLenth)
    #     print(i)
    # print(maxLenth)
    return word_counts

# jieba分词后，绘制任意长度为 lenth 词组的词云，lenth = 0时，绘制所有词组
def draw_wcloud(lenth):
    word_counts = write_filtered_poem()
    if lenth != 0:
        filter_counts = {k: v for k, v in word_counts.items() if len(k) == lenth}
        word_counts = Counter(filter_counts)
    # 加载图像文件并转换为灰度
    mask_image = Image.open('resource/image/bg1.jpg').convert('L')
    # 创建词云对象
    wordcloud = WordCloud(
        font_path='resource/font/songti.ttf',  # 指定中文字体路径
        width=1200,
        height=800,
        mask=np.array(mask_image),
        background_color='white',
        max_words=300  # 设置词云显示的最大词数
    )
    # 生成词云
    wordcloud.generate_from_frequencies(word_counts)
    wordcloud.to_file(f'result/wordcloud_char{lenth}_poem.png')  # 保存为PNG格式

# jieba分词后，绘制长度为 lenth ,前 num 个频率最高的词组柱状图
def draw_split_chars(char_lenth, num=15):
    word_counts = write_filtered_poem()
    filter_counts = {k: v for k, v in word_counts.items() if len(k) == char_lenth}
    filter_counts = Counter(filter_counts)
    num = min(num, len(filter_counts))
    most_common_element = filter_counts.most_common(num)
    chars = []
    counts = []
    for element in most_common_element:
        chars.append(element[0])
        counts.append(element[1])
    plt.figure(figsize=(15, 8))
    plt.bar(chars, counts, color='Teal')
    plt.xlabel('词组')
    plt.ylabel('频率')
    plt.title('高频词组统计')
    plt.savefig(f'result/lenth{char_lenth}_num{num}_frequency.png')
    plt.clf()

if __name__ == '__main__':
    write_poem()
    for i in range(1, 7):
        draw_split_chars(i)
        draw_wcloud(i)


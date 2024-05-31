import os
from collections import Counter
import numpy as np
from wordcloud import WordCloud
from PIL import Image
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
txtfile_path = 'data/all_poet.txt'

# 提取所有诗人到一个文件“all_poet.txt”下
def write_poet():
    if os.path.exists(txtfile_path) == False:
        # 初始化汇总文件，准备写入
        with open('data/all_poet.txt', 'w', encoding='utf-8') as outfile:
            # 遍历1到500的数字，构造文件名
            for i in range(1, 501):
                filename = f'poem/roll_{i}.txt'
                # 打开每个文件并读取内容
                with open(filename, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        # 去除每行首尾的空白字符（包括换行符）
                        stripped_line = line.strip()
                        # 检查行是否包含特定符号
                        if '】' not in stripped_line:
                            continue
                        # 提取作者名
                        poet = stripped_line[stripped_line.rfind('】') + 1:]
                        if poet.find(' ') != -1:
                            poet = poet[:poet.find(' ')]
                        if len(poet) > 5:
                            continue
                        outfile.write(poet + '\n')

# 绘制写诗数量词云
def draw_poet_wcloud():
    with open(txtfile_path, 'r', encoding='utf-8') as file:
        word_counts = []
        for line in file:
            word_counts.append(line.strip())
        word_counts = Counter(word_counts)
    # 加载图像文件并转换为灰度
    mask_image = Image.open('resource/image/bg.png').convert('L')
    # 创建词云对象
    wordcloud = WordCloud(
        font_path='resource/font/songti.ttf',  # 指定中文字体路径
        width=800,
        height=1200,
        mask=np.array(mask_image),
        background_color='white',
        max_words=100  # 设置词云显示的最大词数
    )
    # 生成词云
    wordcloud.generate_from_frequencies(word_counts)
    wordcloud.to_file('result/wordcloud_poet.png')  # 保存为PNG格式

# 绘制创作数量分布饼图, 为展示写诗数目最多的前 num 人及其他
def draw_poet_pie(NUM_POET = 20):
    with open(txtfile_path, 'r', encoding='utf-8') as file:
        word_counts = []
        for line in file:
            word_counts.append(line.strip())
        word_counts = Counter(word_counts)

    sumCount = sum(word_counts.values())
    info_poet = '共计作者：' + str(len(word_counts)) + '人'
    info_poem = '有作者记录的作品：' + str(sumCount) + '首'
    NUM_POET = min(NUM_POET, len(word_counts))
    print(info_poem)
    print(info_poet)
    chars = []
    counts = []
    most_common_element = word_counts.most_common(NUM_POET)
    for element in most_common_element:
        chars.append(element[0])
        counts.append(element[1])
    chars.append('其他')
    counts.append(sumCount - sum(counts))

    explode = [0 for _ in range(NUM_POET+1)]
    explode[0] = explode[1] = 0.02
    colors = ['#9C755F', '#4E79A7', '#F28E1C', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7']

    plt.figure(figsize=(15, 11))
    plt.pie(counts, explode=explode, labels=chars, colors=colors, autopct='%1.2f%%', shadow=True, startangle=90)
    plt.axis('equal')  # 显示为圆（避免比例压缩为椭圆）
    plt.title('诗人创作数分布', fontsize=20)
    plt.savefig(f'result/poet_num{NUM_POET}_pie.png')
    plt.clf()

if __name__ == '__main__':
    # write_poet()
    # draw_poet_wcloud()
    draw_poet_pie()

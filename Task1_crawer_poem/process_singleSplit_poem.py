import os
from collections import Counter
import numpy as np
from wordcloud import WordCloud
from PIL import Image
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

# 提取所有诗内容到一个文件
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
                                poem = poem[:poem.rfind('。') + 1]
                            outfile.write(poem + '\n')

# 单字分词，返回charCount计数器
def write_filtered_char():
    with open('data/all_poem.txt', 'r', encoding='utf-8') as file:
        all_poem_text = file.read()
    charList = []
    for char in all_poem_text:
        if char not in stopwords:
            charList.append(char)
    charCount = Counter(charList)
    return charCount

# 单字分词后，绘制单字前 num 个频率最高的词组柱状图
def draw_split_single(num=15):
    charCount = write_filtered_char()
    chars = []
    counts = []
    most_common_element = charCount.most_common(num)
    for element in most_common_element:
        chars.append(element[0])
        counts.append(element[1])
    plt.figure(figsize=(10, 6))
    plt.bar(chars, counts, color='Teal')
    plt.xlabel('单字')
    plt.ylabel('频率')
    plt.title('单字分词的高频单字')
    plt.savefig(f'result/singlechar_num{num}_frequency.png')
    plt.clf()

# 绘制季节分布饼图
def draw_split_season():
    charCount = write_filtered_char()
    seasons = ['春', '夏', '秋', '冬']
    counts = []
    for season in seasons:
        counts.append(charCount[season])

    colors = ['yellowgreen', 'gold', 'lightcoral', 'lightskyblue']
    explode = (0.1, 0, 0, 0)

    plt.pie(counts, explode=explode, labels=seasons, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
    plt.axis('equal')  # 显示为圆（避免比例压缩为椭圆）
    plt.title('四季分布')
    plt.savefig('result/season_pie.png')
    plt.clf()

if __name__ == '__main__':
    write_poem()
    draw_split_single()
    draw_split_season()


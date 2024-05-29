# 《全唐诗》文本爬取与可视化

《全唐诗》是一部范围较广的唐诗总集。全书以明胡震亨《唐音统签》及清季振宜《唐诗》为底本，又旁采碑、碣、稗史、杂书之所载拾遗补缺而成，共收录唐、五代350年间诗歌48900余首。

## 一、数据爬取

爬取网站 https://www.diyifanwen.com/guoxue/quantangshi/ 上所有唐诗，网站中以卷为单位分割为504卷，实际爬虫结果为500卷。缘于网站中有一些卷号不连续，如下图：

<img src="https://qinglan-1324038201.cos.ap-nanjing.myqcloud.com/images/202405231453621.png" alt="image-20240523145153271" style="zoom:50%;" />

导入所有包：

```
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from collections import Counter
from urllib.parse import urljoin
```

在索引页面结构中，每卷唐诗展示在一个`<ul>`标签中，找出索引分页URL链接的规律，从第2页至第20页网页链接都为 https://www.diyifanwen.com/guoxue/quantangshi/index_i.html ，i=2,3,...20。遍历20个索引页面中所有的`<ul>`标签，将卷号链接放入l列表` volume_urls`中。

```python
base_url = "https://www.diyifanwen.com/guoxue/quantangshi/" 

volume_urls = []  
headers = { 
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
} 
# 获取所有卷链接
def crawl_pagination(base_url, start_page, end_page):
    for i in range(start_page, end_page + 1):
        page_url = base_url
        if i != 1:
            page_url = f"{base_url}index_{i}.html"
        response = requests.get(page_url,headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            lis = soup.find_all('li')
            for li in lis:
                a_tag = li.find('a', href=True)
                if a_tag:
                    volume_link = a_tag['href'][2:]
                    volume_urls.append(volume_link)  


crawl_pagination(base_url, 1, 20)
```

在每个卷中，有若干个分页面，找出分页面URL链接的规律，从第2页至第20页网页链接都为 https://www.diyifanwen.com/guoxue/quantangshi/<卷首页>_i.htm ，i=2,3... 

处理每卷中所有内容，查看网页源代码，找到网页中具有特定类名`content`的元素并读取其内容，`requests`库来发送HTTP请求，并使用`BeautifulSoup`库来解析HTML并提取数据。处理`<p>`标签之间的换行，以确保段落之间在文本中有明显的分隔。

需要注意：

- `requests`库默认输出的编码类型为`ISO-8859-1`，并不是原网页的编码类型，需要修改为`gbk`，与原网页保持一致。
- 原网页卷一的内容格式与其他卷不同，不能直接处理`<p>`标签之间的换行，可以爬取后对卷一手工换行。

```python
# 爬取卷内的所有内容
def crawe_rolltext(num, roll_url):
    all_text = ""
    if not roll_url.startswith("http://") and not roll_url.startswith("https://"):
            roll_url = "http://" + roll_url
    response = requests.get(roll_url, headers=headers)
    
    if response.status_code == 200:
        for i in range(1,10):
            content_url = roll_url
            if i != 1:
                content_url = content_url.replace(".htm", f"_{i}.htm")
            response = requests.get(content_url, headers=headers)
            response.encoding = 'gbk'
            if response.status_code != 200:
                break
            soup = BeautifulSoup(response.text, 'html.parser')
            content_element = soup.find('div', class_='content')
            if content_element:
                if num == 1:
                    all_paragraphs_text = content_element.get_text(strip=True)
                else:
                    paragraphs = content_element.find_all('p')
                    all_paragraphs_text = "\n".join([para.get_text() for para in paragraphs])
                # content_text = content_element.get_text(strip=True)
                all_text += all_paragraphs_text + "\n"
                
    
    with open(f"./poem/roll_{num}.txt",'w', encoding='utf-8') as file:
        file.write(all_text)  
        
for index, volume_url in enumerate(volume_urls):
    crawe_rolltext(index+1, volume_url)
```

爬取部分数据结果如图例：

<img src="https://qinglan-1324038201.cos.ap-nanjing.myqcloud.com/images/202405231545978.png" alt="image-20240523154540544" style="zoom:50%;" />

## 二、数据可视化

### 1 古诗文可视化

#### **数据预处理**

提取网站500卷中的所有的诗文内容，作以下格式处理：

- 判断行中是否有符号'【'或’】‘，忽略标题和作者行

- 有合作诗则忽略"--"与空格间的文字，以空格分割每行后忽略以“--”开头的字符串。

  <img src="https://qinglan-1324038201.cos.ap-nanjing.myqcloud.com/images/202405291121305.png" alt="image-20240528195857509" style="zoom:50%;" />

- 忽略行尾标点号后的文字，使用`rfind`函数找到最后一个标点并截取其前所有内容。

  <img src="https://qinglan-1324038201.cos.ap-nanjing.myqcloud.com/images/202405291121307.png" alt="image-20240528195538917" style="zoom: 50%;" />

```python
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
                                    poem = poem[:poem.rfind('。')+1]
                                outfile.write(poem + '\n')
```

#### **分词和去除停用词**

考虑两种分词方法，一种以单字分词，一种使用jieba分词。

停用词参考网上常见虚词、叹词等，设置如下：<img src="https://qinglan-1324038201.cos.ap-nanjing.myqcloud.com/images/202405291121308.png" alt="image-20240528200453289" style="zoom:50%;" />

#### 可视化图表

**单字分词**后可统计诗词中提到季节和最多单字频率：

- 饼图展示四季分布，可见春秋两季占据提及季节的全唐诗92%，”伤春悲秋“的说法有一定数据支持。

<img src="https://qinglan-1324038201.cos.ap-nanjing.myqcloud.com/images/202405291124161.png" alt="season_pie" style="zoom:67%;" />

```
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
```



- 由柱状图展示单字频率最高的十项，分布如下。展现以人为本、借景抒情、踌躇满志等诗情表达方向。

  <img src="https://qinglan-1324038201.cos.ap-nanjing.myqcloud.com/images/202405291124162.png" alt="single_char_frequency" style="zoom:67%;" />
  
  ```
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
  ```

**结巴分词**，提取分词文件“filter_poem.txt”，分别对不同长度词组分别展示词云和柱状图。分词后词组最大长度为7。以长度为2和4的词组为例：

<img src="https://qinglan-1324038201.cos.ap-nanjing.myqcloud.com/images/202405291124163.png" alt="wordcloud_char2_poem" style="zoom: 33%;" />

<img src="https://qinglan-1324038201.cos.ap-nanjing.myqcloud.com/images/202405291124164.png" alt="lenth2_num15_frequency" style="zoom: 50%;" />

<img src="https://qinglan-1324038201.cos.ap-nanjing.myqcloud.com/images/202405291124165.png" alt="lenth4_num15_frequency" style="zoom:33%;" />

<img src="https://qinglan-1324038201.cos.ap-nanjing.myqcloud.com/images/202405291124166.png" alt="lenth4_num15_frequency" style="zoom:50%;" />

```python
# jieba分词后，绘制任意长度为 lenth 词组的词云，lenth = 0时，绘制所有词组
def draw_wcloud(lenth):
    word_counts = write_filtered_poem()
    if lenth != 0:
        filter_counts = {k: v for k, v in word_counts.items() if len(k) == lenth}
        word_counts = Counter(filter_counts)
    # 加载图像文件并转换为灰度
    mask_image = Image.open('bg1.jpg').convert('L')
    # 创建词云对象
    wordcloud = WordCloud(
        font_path='font/songti.ttf',  # 指定中文字体路径
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
```

### 2 创作数目可视化

#### 数据预处理

提取网站500卷中的所有作者名出现次数，作以下格式处理：

- 判断行中是否有符号'【'或’】‘，若不存在则忽略

- 提取’】'与其后空格间的文字，若无空格则直接切片为作者名

```python
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
```

#### 可视化图表

依据作者对诗人创作数目进行统计，有作者记录的作品22625首，共计作者1037人。选择创作数最多前20人绘制拼图，在有作者的诗中，发现创作数最多的前20人创造了近一半：

<img src="https://qinglan-1324038201.cos.ap-nanjing.myqcloud.com/images/202405291124167.png" alt="poet_num20_pie" style="zoom:50%;" />

生成词云：

<img src="https://qinglan-1324038201.cos.ap-nanjing.myqcloud.com/images/202405291124168.png" alt="wordcloud_poet" style="zoom: 50%;" /><img src="https://qinglan-1324038201.cos.ap-nanjing.myqcloud.com/images/202405291124169.jpg" alt="bg" style="zoom: 50%;" />

```python
# 绘制写诗数量词云
def draw_poet_wcloud():
    with open(txtfile_path, 'r', encoding='utf-8') as file:
        word_counts = []
        for line in file:
            word_counts.append(line.strip())
        word_counts = Counter(word_counts)
    # 加载图像文件并转换为灰度
    mask_image = Image.open('bg.png').convert('L')
    # 创建词云对象
    wordcloud = WordCloud(
        font_path='font/songti.ttf',  # 指定中文字体路径
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
```

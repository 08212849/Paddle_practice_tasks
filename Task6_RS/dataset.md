## data

### 预测/验证集

数据列间用 \t 分割，字段说明：

- q网民搜索词
- ad_id广告 ID
- click点击次数

<img src="https://qinglan-1324038201.cos.ap-nanjing.myqcloud.com/images/202406182202438.png" alt="image-20240611085715341" style="zoom: 50%;" />

<img src="https://qinglan-1324038201.cos.ap-nanjing.myqcloud.com/images/202406182202440.png" alt="image-20240618173435740" style="zoom:50%;" />

### 广告核心特征

- ad_id广告 ID
- core_terms核心词字面，不同核心词间用<sep>连接

<img src="https://qinglan-1324038201.cos.ap-nanjing.myqcloud.com/images/202406182202441.png" alt="image-20240611085906375" style="zoom:50%;" />

### 广告落地页特征

- ad_id 广告 ID
- lp_content 落地页内容，不同段落间用<block>连接

![image-20240611090150901](https://qinglan-1324038201.cos.ap-nanjing.myqcloud.com/images/202406182202442.png)



## 双塔模型

广告落地页特征:corpus.csv(28w)

![image-20240618174418670](https://qinglan-1324038201.cos.ap-nanjing.myqcloud.com/images/202406182202443.png)

dev（1000）

![image-20240618174238330](https://qinglan-1324038201.cos.ap-nanjing.myqcloud.com/images/202406182202444.png)

![image-20240618174322715](https://qinglan-1324038201.cos.ap-nanjing.myqcloud.com/images/202406182202445.png)

train(70w)

![image-20240618174540899](https://qinglan-1324038201.cos.ap-nanjing.myqcloud.com/images/202406182202446.png)

in_batch_negative策略项目结构：

<img src="https://qinglan-1324038201.cos.ap-nanjing.myqcloud.com/images/202406182202447.png" alt="image-20240618141816417" style="zoom:67%;" />

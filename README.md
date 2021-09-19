# nlp-beginner-task1-textClassification-LogisticAndSoftmax
实现基于Logistic/Softmax Regression的文本分类
## 任务一：基于机器学习的文本分类

实现基于logistic/softmax regression的文本分类

实验时间2021.9.13~2021.9.14
### 一、问题描述

1. 实现基于机器学习的文本分类，使用Logistic/Softmax Regression模型

2. 数据集：https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews

3. 分类要求，将sentence进行情感分类吗，一共五类

   0: negative

   1: somewhat negative

   2: neutral

   3: somewhat positive

   4: positive

### 二、数据处理

1. 数据读取

   ​    可以直接使用pandas对数据train.tsv进行读取和处理，并使用train_test_split函数进行数据集的划分。我此处使用的划分比例为train: test = 8: 2

2. 文本特征提取

   我此处文本向量化的方法是sklearn中的CountVectorize方法，也可以使用TfidfVectorizer。

   CountVectorizer：考虑词汇在文本中出现的频率

   TfidfVectorizer：考虑词汇在文本中出现的频率和包含某个词汇的文档数量

### 三、模型

#### 1.Logistic Regression模型

1. LogisticRegression是一个分类模型，通常用于二分类。当用于多分类时，可以训练多个二分类分类器。第k个二分类器将第k个分类作为正样本，将其他的样本均作为负样本。
2. 假设函数h(x）

![IMG_AB56AEADEC9D-1](/Users/seventeen/Downloads/IMG_AB56AEADEC9D-1.jpeg)

3. 代价函数：交叉熵误差

![IMG_2226](/Users/seventeen/Downloads/IMG_2226.jpg)

4. 训练方式：梯度下降
5. 实现：此处实现直接调用了sklearn中的LogisticRegression模型，选择的梯度下降方法为随机梯度下降。

#### 2. Softmax Regression模型

1. Logistic Regression的多分类模式，将假设函数中的sigmoid函数换成softmax函数

2. 训练：在'softmax/softmax.py'中定义并实现了SoftmaxRegression类。

   其中实现了两种训练方式，随机梯度下降train_sgd()和批量梯度下降train_bgd()

   在调用训练函数进行训练时，可以指定以下参数

   max_itr：训练迭代次数

   alpha：学习率learning rate

   lamda：正则化参数，defaul=0，表示不加入正则化

3. 预测：predict()方法，输入x，输出预测标签y
4. 测试：test()使用测试集进行测试，使用的衡量指标为准确率
5. 训练方式性能差异：批量梯度下降的训练方法比较耗时，且需要测迭代次数较多才能表现较好的性能；随机梯度下降训练较快，且迭代次数在很小时，也能表现处较好的性能。
6. 其他：由于时间较为紧张，没有进一步开展更多的实验，后续实验可以从学习率、正则化、梯度下降方法、特征提取方法等多方面进行研究对比。之后会找时间继续补上。



      

import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


# 读取数据
data_train = pd.read_csv('../data/train.tsv', sep='\t')
x = data_train['Phrase']
y = data_train['Sentiment']
# 划分数据集 train:cv:test=6:2:2
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# 文本向量化
vec = CountVectorizer(stop_words='english')
x_train_vec = vec.fit_transform(x_train)
x_test_vec = vec.transform(x_test)
# print('特征数量：', len(vec.get_feature_names()))

lr = LogisticRegression(solver='sag')  # 随机梯度下降
lr.fit(x_train_vec, y_train)
y_predict = lr.predict(x_test_vec)
print(y_predict)
print('accuracy:', metrics.accuracy_score(y_test, y_predict))


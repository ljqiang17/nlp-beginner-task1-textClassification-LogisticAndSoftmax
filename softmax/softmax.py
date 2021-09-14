import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

class SoftmaxRegression:
    def __init__(self):
        self.weights = np.random.uniform(0, 1, (5, 14942))  # 本项目的数据集类别数目为5， 特征数目为14942


    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def train_sgd(self, x_train, y_train, max_itr=100, alpha=0.1, lamda=0):  # shuffle gradient decent
        print("-----Training-----")
        for itr in range(max_itr):
            print('iteration:%d' % itr)
            for i in range(len(x_train)):
                x = x_train[i].reshape(-1, 1)  # 取出每一个样本的特征，并转置，设置为特征数n*1的向量形式
                y = np.zeros((5, 1))
                y[y_train.iloc[i]] = 1  # 将y设置为one-hot的向量形式
                h_y = self.softmax(np.dot(self.weights, x))  # 预测值
                # self.weights = self.weights - alpha * (np.dot((h_y - y), x.T))  # 随机梯度下降更新参数
                self.weights = self.weights - alpha * (np.dot((h_y - y), x.T)) + alpha * lamda * self.weights   # 加入了正则项,lamda默认为0，相当于没有正则化
        print('-----Train Finished-----')
        return self.weights

    def train_bgd(self, x_train, y_train, max_itr=100, alpha=0.1, lamda=0):  # batch gradient decent
        print("-----Training-----")
        for itr in range(max_itr):
            print('iteration:%d' % itr)
            err = np.zeros((5, 14942))
            for i in range(len(x_train)):
                x = x_train[i].reshape(-1, 1)  # 取出每一个样本的特征，并转置，设置为特征数n*1的向量形式
                y = np.zeros((5, 1))
                y[y_train.iloc[i]] = 1  # 将y设置为one-hot的向量形式
                h_y = self.softmax(np.dot(self.weights, x))  # 预测值
                singleErr = np.dot((h_y - y), x.T)
                err += singleErr
            err = err / len(x_train)
            regular = (alpha * lamda * self.weights) / len(x_train) #正则项
            self.weights = self.weights - alpha * err + regular
        print('-----Train Finished-----')
        return self.weights



    def predict(self, x_test):
        y_predict = []
        for i in range(len(x_test)):
            x = x_test[i]
            y = np.argmax(np.dot(self.weights, x))
            y_predict.append(y)

        return y_predict

    def test(self, y_test, y_predict):
        print('-----Testing-----')
        accuracy = 0
        for i in range(len(y_test)):
            if y_predict[i] == y_test.iloc[i]:
                accuracy += 1

        accuracy = accuracy / len(y_test)
        print('-----Test Finished-----')
        return accuracy




if __name__ == '__main__':

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
    x_train_arr = x_train_vec.toarray()
    x_test_arr = x_test_vec.toarray()

    # f = vec.get_feature_names()

    # 实例化softmaxregression
    sl = SoftmaxRegression()
    sl.train_sgd(x_train_arr, y_train, 5)
    # sl.train_bgd(x_test_arr, y_train, 100)
    y_predict = sl.predict(x_test_arr)
    accuracy = sl.test(y_test, y_predict)
    print(accuracy)
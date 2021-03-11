import pandas as pd
import numpy as np
#读取文件
def file_read(file_name = './datingTestSet.txt'):
    with open(file_name,'r',encoding = 'utf-8') as file:
        content = file.readlines()
        ls_data = []
        ls_label = []
        for i in content:
            ls_data.append((i.strip().split('\t'))[:3])
            ls_label.append((i.strip().split('\t'))[3])
        for index,value in enumerate(ls_label):
            if value == 'largeDoses':
                ls_label[index] = 2
            elif value == 'smallDoses':
                ls_label[index] = 1
            elif value == 'didntLike':
                ls_label[index] = 0
        return ls_data,ls_label

# 预处理数据集
def initial_data(ls_data, ls_label):
    data = pd.DataFrame(ls_data, columns=['ffmile', 'gametime', 'icecream'], dtype='float')
    # data['ffmile'] = data['ffmile'].apply(lambda x: float(x))
    # data['gametime'] = data['gametime'].apply(lambda x: float(x))
    # data['icecream'] = data['icecream'].apply(lambda x: float(x))
    label = pd.DataFrame(ls_label, columns=['label'])
    return data, label

#归一化
def Norm(data):
    for i in data.columns:
        max_min_ffmile = max(data[i])-min(data[i])
        data[i] = (data[i]-min(data[i]))/max_min_ffmile
    Norm_data = data
    return Norm_data

#计算点对点之间的欧氏距离
def dis(a,b):
    distance = np.sqrt(np.sum((a-b)**2,axis=1))
    return distance

#通过距离的排序，获取K个最近邻的点
def get_result(distance,label,k):
    res = np.argsort(distance)
    result = [int(label[i]) for i in res[:k]]
    from collections import Counter
    count = Counter(result).most_common(1)
    return count[0][0]

#划分数据集
def split_func(size,Norm_data,label):
    m = Norm_data.shape[0]
    train_size = int(m*(1-size))
    X_train = Norm_data.iloc[:train_size,:].values
    y_train = label.iloc[:train_size,:].values
    X_test = Norm_data.iloc[train_size:,:].values
    y_test = label.iloc[train_size:,:].values
    return X_train,y_train,X_test,y_test

#定义主函数
def main_func():
    ls_data,ls_label = file_read(file_name = './datingTestSet.txt')
    data,label = initial_data(ls_data,ls_label)
    Norm_data = Norm(data)
    X_train,y_train,X_test,y_test = split_func(0.3,Norm_data,label)
    result = []
    for i in range(X_test.shape[0]):
        distance = dis(X_test[i],X_train)
        result.append(get_result(distance,y_train,k = 5))
    w = 0
    for index,value in enumerate(result):
        if value != y_test[index]:
                w += 1
    return (f'Accuracy:{(1-(w/(y_test.shape[0])))*100}%')

print(main_func())
```python3
#导入波士顿房价的数据集
import numpy as pd
import pandas as pd
from sklearn.datasets import load_boston
data = load_boston()
X = pd.DataFrame(data['data'],columns = data['feature_names'])
y = pd.DataFrame(data['target'],columns = ['label'])

定义随机抽取函数，从训练集n个样本中中随机并有放回的抽取n个训练样本，每个训练样本从M个特征中随机抽取m个特征（m<<M）
import random
def select_data_feature(X,drop_fea = 4):
    feature = random.sample(X.columns.tolist(),k = len(X.columns)-drop_fea)
    row = np.random.choice(range(X.shape[0]),size = X.shape[0],replace = True)
    return row,X.iloc[row][feature]

#划分训练集和测试集   
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 33)

#建立弱分类器，训练模型，并算出每个弱分类器对于训练集和测试集的score
from sklearn.tree import DecisionTreeRegressor
def main_func(X_train,X_test,y_train,y_test):
    #取出弱分类器的训练集和对应的行数
    row,weak_X_train = select_data_feature(X_train)
    model = DecisionTreeRegressor()
    model.fit(weak_X_train,y_train.iloc[row])
    #输出弱分类器训练集的score
    train_score = model.score(weak_X_train, y_train.iloc[row])
    #输出弱分类器测试集的score
    test_score = model.score(X_test[weak_X_train.columns], y_test)
    
    print('train score = {}; test score = {}'.format(train_score, test_score))
    #获取弱分类器预测的y_pred
    y_pred = model.predict(X_test[weak_X_train.columns])
    return y_pred

#定义随机森林函数
def random_forest(X_train,X_test,y_train,y_test):
    #建立一个空列表，用来收集每个弱分类器预测的结果
    all_pred = []
    #做4个弱分类器
    for _ in range(4):
        y_pred = main_func(X_train,X_test,y_train,y_test)
        all_pred.append(y_pred)
     #取所有预测结果的平均值作为最终输出
    return np.mean(all_pred,axis = 0)

random_forest_pred = random_forest(X_train, X_test, y_train, y_test)
random_forest_pred
```
![image text](https://github.com/Vihagle/AI/blob/main/image/1615548391(1).jpg)
```python3
#从上图可以看出每个弱分类器的表现并不好，但是如果将他们ensemble起来，会如何呢？
from sklearn.metrics import r2_score
r2_score(y_test, random_forest_pred)
```
![image text](https://github.com/Vihagle/AI/blob/main/image/1615548396(1).jpg)

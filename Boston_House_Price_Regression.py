from sklearn.datasets import load_boston
import pandas as pd

#Loading Data
data = load_boston()
dataframe = pd.DataFrame(data['data'],columns=data['feature_names'])
#将数据集中离散型的数据转化成str数据类型
dataframe['CHAS'] = dataframe['CHAS'].astype('int')
dataframe['CHAS'] = dataframe['CHAS'].astype('category')
dataframe['RAD'] = dataframe['RAD'].astype('int')
dataframe['RAD'] = dataframe['RAD'].astype('category')

#将离散数据onehot编码化
dataframe = pd.get_dummies(dataframe,columns = ['RAD','CHAS'])

#划分输入X，输出y
X = dataframe
y = data['target']

#将数据做归一化/标准化
from sklearn.preprocessing import StandardScaler
stander = StandardScaler()
X = stander.fit_transform(X)

#将数据划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=33)

#从sklearn库中调用LinearRegression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

#训练、拟合训练集
lr.fit(X_train,y_train)

#评估模型
print(lr.score(X_train,y_train))
print(lr.score(X_test,y_test))

#预测测试数据
print(lr.predict([X_test[0]]))

#将测试集的预测结果和实际结果化成散点图，查看拟合程度
import numpy as np
import matplotlib.pyplot as plt
fig,ax = plt.subplots(8,3,figsize = (40,40))
for i in range(X_test.shape[1]):
    ix = np.unravel_index(i,ax.shape)
    plt.sca(ax[ix])
    ax[ix].title.set_text(f'feature{i}')
    plt.scatter(X_test[:,i],y_test,s=3)
    plt.scatter(X_test[:,i],lr.predict(X_test),s=3)
plt.show()
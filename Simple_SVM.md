```python3
import numpy as np
import matplotlib.pyplot as plt

#划分两堆数据点
data_1 = np.random.normal(6,2,size = (50,2))
data_2 = np.random.normal(-6,2,size = (50,2))
plt.scatter(*zip(*data_1))
plt.scatter(*zip(*data_2))
plt.show()
```
![Image text](https://github.com/Vihagle/AI/blob/main/image/1615543360(1).jpg)
```python3
#定义划分直线的线性函数
def func(k,x,b):
    return k*x+b

#获取满足y(wx+b)>1的一系列w,b
data_1_x = data_1[:,0]
data_2_x = data_2[:,0]
k_and_b = []
for _ in range(100):
    k,b = (np.random.random(size = (1,2))*10-5)[0]
    if np.max(func(k,data_1_x,b))<=-1 and np.min(func(k,data_2_x,b))>=1:
        k_and_b.append((k,b))
print(k_and_b)

#画出这些w,b所对应的分割直线
x = np.append(data_1_x,data_2_x)
plt.scatter(*zip(*data_1))
plt.scatter(*zip(*data_2))
for k ,b in k_and_b:
    plt.plot(x,func(k,x,b))
```
![Image text](https://github.com/Vihagle/AI/blob/main/image/1615543373(1).jpg)
```python3
#为了满足loss函数，我们需要找到w最小的函数，此时的margin最大
final_k,final_b = sorted(k_and_b,key = lambda x:abs(x[0]))[0]

#画出margin最大的分割直线
plt.scatter(*zip(*data_1))
plt.scatter(*zip(*data_2))
plt.plot(x,func(final_k,x,final_b))
```
![Image text](https://github.com/Vihagle/AI/blob/main/image/1615543380(1).jpg)

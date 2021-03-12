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

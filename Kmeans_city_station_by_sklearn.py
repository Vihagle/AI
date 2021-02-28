from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
from collections import defaultdict
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

coordination_source = """
{name:'兰州', geoCoord:[103.73, 36.03]},
{name:'嘉峪关', geoCoord:[98.17, 39.47]},
{name:'西宁', geoCoord:[101.74, 36.56]},
{name:'成都', geoCoord:[104.06, 30.67]},
{name:'石家庄', geoCoord:[114.48, 38.03]},
{name:'拉萨', geoCoord:[102.73, 25.04]},
{name:'贵阳', geoCoord:[106.71, 26.57]},
{name:'武汉', geoCoord:[114.31, 30.52]},
{name:'郑州', geoCoord:[113.65, 34.76]},
{name:'济南', geoCoord:[117, 36.65]},
{name:'南京', geoCoord:[118.78, 32.04]},
{name:'合肥', geoCoord:[117.27, 31.86]},
{name:'杭州', geoCoord:[120.19, 30.26]},
{name:'南昌', geoCoord:[115.89, 28.68]},
{name:'福州', geoCoord:[119.3, 26.08]},
{name:'广州', geoCoord:[113.23, 23.16]},
{name:'长沙', geoCoord:[113, 28.21]},
//{name:'海口', geoCoord:[110.35, 20.02]},
{name:'沈阳', geoCoord:[123.38, 41.8]},
{name:'长春', geoCoord:[125.35, 43.88]},
{name:'哈尔滨', geoCoord:[126.63, 45.75]},
{name:'太原', geoCoord:[112.53, 37.87]},
{name:'西安', geoCoord:[108.95, 34.27]},
//{name:'台湾', geoCoord:[121.30, 25.03]},
{name:'北京', geoCoord:[116.46, 39.92]},
{name:'上海', geoCoord:[121.48, 31.22]},
{name:'重庆', geoCoord:[106.54, 29.59]},
{name:'天津', geoCoord:[117.2, 39.13]},
{name:'呼和浩特', geoCoord:[111.65, 40.82]},
{name:'南宁', geoCoord:[108.33, 22.84]},
//{name:'西藏', geoCoord:[91.11, 29.97]},
{name:'银川', geoCoord:[106.27, 38.47]},
{name:'乌鲁木齐', geoCoord:[87.68, 43.77]},
{name:'香港', geoCoord:[114.17, 22.28]},
{name:'澳门', geoCoord:[113.54, 22.19]}
"""

city_location = {
    '香港': (114.17, 22.28)
}

import re
reg = r"name:'(\w+)',\sgeoCoord:\[(\d+.\d+),\s(\d+.\d+)\]"
for i in coordination_source.split('\n'):
    if not i :
        continue
    else:
        city,long,fat = (re.findall(reg,i))[0]
        city_location[city] = (float(long),float(fat))
print(city_location)

#取出所有城市名和坐标点，分别放到列表中
city = []
location = []
for key,value in city_location.items():
    location.append(value)
    city.append(key)
#此处是将列表转换为数组，方便后续处理使用
location = np.array(location).reshape((-1,2))
# print(location)

#调出Kmeans算法
kmeans = KMeans(n_clusters=5,random_state=0)
#训练kmeans模型，将坐标值做分类
kmeans.fit(location)
#输出所有归类好的坐标点的类别
labels = kmeans.labels_
print(f'The label is  {labels}')
#分别输出每个类的中心点
centers = kmeans.cluster_centers_
print(f'The center is {centers}')
#在图上画出中心点的位置
centers_x = centers[:,0]
centers_y = centers[:,1]
plt.scatter(centers_x,centers_y,c = 'black',marker='*',s=100)
#建立一个默认字典，用于存放每个类别的坐标点：{label：[points]}
closet_point = defaultdict(list)
for index,value in enumerate(labels):
    closet_point[value].append(location[index])
print(closet_point)
for key,value in closet_point.items():
    #此处是先将一段数组分解成一个个坐标点，然后在分解成x值，y值，然后在图上画出每个点
    plt.scatter(*zip(*value))
plt.show()
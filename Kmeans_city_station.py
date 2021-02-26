from pylab import mpl
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

#利用re正则化使字符串输出一个字典，{城市名：位置坐标}
import re
reg = r"name:'(\w+)',\sgeoCoord:\[(\d+.\d+),\s(\d+.\d+)\]"
for i in coordination_source.split('\n'):
    if not i :
        continue
    else:
        city,long,fat = (re.findall(reg,i))[0]
        city_location[city] = (float(long),float(fat))
# print(city_location)

#定义球面地理距离公式，方便后面计算坐标距离
import math
def geo_distance(origin, destination):
    lon1, lat1 = origin
    lon2, lat2 = destination
    radius = 6371  # km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c
    return d


#使用上面建立的字典，取出里面得到value（即位置坐标），分别传入x坐标的列表，y坐标的列表
def initial(city_location):
    all_x = []
    all_y = []
    for _,location in city_location.items():
        x,y = location
        all_x.append(x)
        all_y.append(y)
    return all_x,all_y

#在x坐标列表和y坐标列表里面的范围，随机创建初始中心点，
import random
def get_random_center(all_x,all_y):
    r_x = random.uniform(min(all_x),max(all_x))
    r_y = random.uniform(min(all_y),max(all_y))
    return r_x,r_y


#此步是将每个坐标划分到每个初始中心点的归类，输出一个字典
from collections import defaultdict
def built_dic(all_x,all_y,centers):
    #先初始化一个字典，key用来定义每个初始中心点，value是一个list，用来收集每个划分好后的坐标点
    closet_points = defaultdict(list)
    #压缩x列表，y列表，提取出每个坐标点的x点，y点
    for x,y in zip(all_x,all_y):
        #算出每个点到中心点的距离，然后取出距离最小的那个中心点和其距离
        closet_c,closet_dis = min([(k,geo_distance((x,y),centers[k])) for k in centers],key = lambda t:t[1])
        #提取中心点和坐标点放进初始化好的closet_points字典中，{中心点：[坐标点1，坐标点2...]}
        closet_points[closet_c].append([x,y])
    return closet_points

#开始迭代
import numpy as np
def iterate_once(centers,closet_points,threshold = 5):
    #此处定义一个迭代开停的开关，后面主函数会用到
    have_changed = False
    #遍历closet_points字典的key
    for c in closet_points:
        #提取原始中心点的坐标
        former_center = centers[c]
        #获取所属每个中心点的坐标点列表
        neighbors = closet_points[c]
        #算出每个列表所有坐标点的平均值，作为新的中心点的坐标
        neighbors_center = np.mean(neighbors,axis = 0)
        #如果新的中心点和旧的中心点的距离比设定的阈值大，则将旧中心点替换成新的中心点
        if geo_distance(neighbors_center,former_center) > threshold:
            centers[c]  = neighbors_center
            #此处理解为循环继续
            have_changed = True
        else:
            #如果两个中心点的距离比阈值小，则停止迭代
            pass
    return centers,have_changed

#定义kmeans主函数，输出最终中心点，和一个字典 {中心点n：[坐标点1，，坐标点2..坐标点n]}
def kmeans(Xs,k,threshold = 5):
    #提取数据中的x值，y值，各自放进列表中
    all_x,all_y = initial(city_location)
    #将x,y值的列表进行random处理，随机创建初始的中心点
    centers = {i + 1: get_random_center(all_x, all_y) for i in range(k)}
    #启动迭代开关
    have_changed = True
    while have_changed:
        #获取每个中心点所划分的坐标点列表 的一个字典
        closet_points = built_dic(all_x,all_y,centers)
        #迭代出新的中心点
        centers,have_changed = iterate_once(centers,closet_points,threshold)
        print('iteration')
    return centers,closet_points

centers,closet_point = kmeans(np.array(list(city_location.values())),k = 5,threshold = 5)
print(centers)


import matplotlib.pyplot as plt
plt.figure(figsize=(20,8))
all_x,all_y = initial(city_location)
# plt.scatter(all_x,all_y)
# plt.scatter([x for x in (list(zip(*(centers.values()))))[0]],[y for y in (list(zip(*(centers.values()))))[1]])
# plt.show()

#建立能源站字典{能源站n：最终中心点n}
city_location_with_station = {f'能源站-{i}':j for i,j in centers.items()}
print(city_location_with_station)

#开始画图
import networkx as nx
def draw_cities(cities,color = None):
    city_graph = nx.Graph()
    city_graph.add_nodes_from(list(cities.keys()))
    nx.draw(city_graph,cities,node_color = color,with_labels=True,node_size = 30)
draw_cities(city_location_with_station,color='green')
draw_cities(city_location,color='red')
plt.show()
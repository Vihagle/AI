![image text](https://github.com/Vihagle/AI/blob/main/image/123.jpg)

```python3
#拓扑排序函数
def topologic(graph):
    import random
    #建立一个空列表，用来收集拓扑排序后元素
    order = []
    while graph:
        #获取有输出的点，即字典中的键
        all_node_have_ouput = set(graph.keys())
        #获取有输入的点，即字典中值
        all_node_have_input = set(graph.values())
        #将两个集合相减，得到只有输出的点
        delete_node = set(graph.keys()) - set(graph.values())
        #从输出的点的集合中随机取一个点
        res = random.choice(list(delete_node))
        
        #如果此时字典中只存在一对键值对，则直接获取字典中的键与值（避免pop掉键后，值也被删除）
        if len(graph) == 1:
            order.append(res)
            order.append(graph[res])
            graph.pop(res)
            #打印此时的order
            print(f'Now the order is :{order}')
            #打印此时的graph
            print(f'Now the graph is None')
            print('*'*20)
        else:
            #将取出来的点中放到order列表中，并将其从字典中删除
            order.append(res)
            graph.pop(res)
            
            #打印此时的order
            print(f'Now the order is :{order}')
            #打印此时的graph
            print(f'Now the graph is :{graph}')
            print('*'*20)
    return order 
    
#根据上图建立Graph
graph = {'A':'C','B':'C','C':'E','D':'E','E':'F'}
topologic(graph)    
```
输出
```python3
Now the order is :['B']
Now the graph is :{'A': 'C', 'C': 'E', 'D': 'E', 'E': 'F'}
********************
Now the order is :['B', 'A']
Now the graph is :{'C': 'E', 'D': 'E', 'E': 'F'}
********************
Now the order is :['B', 'A', 'D']
Now the graph is :{'C': 'E', 'E': 'F'}
********************
Now the order is :['B', 'A', 'D', 'C']
Now the graph is :{'E': 'F'}
********************
Now the order is :['B', 'A', 'D', 'C', 'E', 'F']
Now the graph is None
********************
['B', 'A', 'D', 'C', 'E', 'F']
```


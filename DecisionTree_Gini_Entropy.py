import numpy as np
import pandas as pd
from collections import Counter

#算出每种类别在该特征发生的概率
def pr(e,es):
    return Counter(es)[e]/len(es)

#求出每个特征的Gini系数
def gini(elements):
    return 1-np.sum(pr(e,elements)**2 for e in set(elements))

#求出每个特征的熵Entropy
def entropy(elements):
    return -np.sum(pr(e,elements)*np.log2(pr(e,elements)) for e in set(elements))

feature_1 = ['R','R','Y','Y']
feature_2 = ['R','R','R','Y']
feature_3 = ['R','R','R','R']

print(gini(feature_1))
print(gini(feature_2))
print(gini(feature_3))
print(entropy(feature_1))
print(entropy(feature_2))
print(entropy(feature_3))
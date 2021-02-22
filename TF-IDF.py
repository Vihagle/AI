import numpy as np
import pandas as pd
#定义数据和预处理
docA = 'The cat sat on my bed'
docB = 'The dog sat on my knees'
bowA = docA.split(' ')
bowB = docB.split(' ') #去除空格
wordSet = set(bowA)|set(bowB) #合并，去重,构建完整的词库

#进行词数的统计
wordDictA = dict.fromkeys(wordSet,0)
wordDictB = dict.fromkeys(wordSet,0) #创建统计字典
for word in bowA: #遍历列表1，统计每个词的频数
    wordDictA[word] += 1
for word in bowB: #遍历列表2，统计每个词的频数
    wordDictB[word] += 1
# df =pd.DataFrame([wordDict_1,wordDict_2],index = ['var_1','var_2'])
# print(df)

#计算词频TF
def computeTF(wordDict,bow):
    tf_dict = {} #用一个字典对象记录tf，把所有在bow文档里的词的tf都算出来
    nbowCount = len(bow) #计算总词数
    for word,count in wordDict.items(): #遍历文档里面的词和频数
        tf_dict[word] = count/nbowCount #将频数除以总词数，得到词频
    return tf_dict
tfA = computeTF(wordDictA,bowA)
tfB = computeTF(wordDictB,bowB)

#计算逆文档频率IDF
def computeIDF(wordDictlist):
    idfDict = dict.fromkeys(wordDictlist[0],0) #初始化一个新的idf字典
    N = len(wordDictlist) #计算总文档数
    import math
    for wordDict in wordDictlist: #遍历所有wordDict
        for word,count in wordDict.items(): #遍历所有wordDict里面的词和频数
            if count>0:
                idfDict[word] +=1 #如果频数大于1，则加1，即计算所有文档里面的词的频数
    for word,Ni in idfDict.items():
        idfDict[word] = math.log10((N+1)/(Ni+1)) #使用公式求出每个词对应的IDF值
    return idfDict
idfs = computeIDF([wordDictA,wordDictB])

#计算TFIDF
def computeTFIDF(tf,idfs):
    tfidf = dict.fromkeys(tf,0) #初始化一个新的tfidf字典
    for word,tfval in tf.items(): #遍历tf里面的词和词频
        tfidf[word] = tfval*idfs[word] #利用公式，将tf里面的词频除以idf里面的文档频率
    return tfidf
tfidfA = computeTFIDF(tfA,idfs)
tfidfB = computeTFIDF(tfB,idfs)
print(pd.DataFrame([tfidfA,tfidfB]))
import numpy as np
import pandas as pd
#定义数据和预处理
docA = 'The cat sat on my bed'
docB = 'The dog sat on my knees'
bowA = docA.split(' ')
bowB = docB.split(' ') #去除空格
wordSet = set(bowA)|set(bowB) #合并，去重,构建完整的词库

#进行词数的统计
wordDictA = dict.fromkeys(wordSet,0)
wordDictB = dict.fromkeys(wordSet,0) #创建统计字典
for word in bowA: #遍历列表1，统计每个词的频数
    wordDictA[word] += 1
for word in bowB: #遍历列表2，统计每个词的频数
    wordDictB[word] += 1
# df =pd.DataFrame([wordDict_1,wordDict_2],index = ['var_1','var_2'])
# print(df)

#计算词频TF
def computeTF(wordDict,bow):
    tf_dict = {} #用一个字典对象记录tf，把所有在bow文档里的词的tf都算出来
    nbowCount = len(bow) #计算总词数
    for word,count in wordDict.items(): #遍历文档里面的词和频数
        tf_dict[word] = count/nbowCount #将频数除以总词数，得到词频
    return tf_dict
tfA = computeTF(wordDictA,bowA)
tfB = computeTF(wordDictB,bowB)

#计算逆文档频率IDF
def computeIDF(wordDictlist):
    idfDict = dict.fromkeys(wordDictlist[0],0) #初始化一个新的idf字典
    N = len(wordDictlist) #计算总文档数
    import math
    for wordDict in wordDictlist: #遍历所有wordDict
        for word,count in wordDict.items(): #遍历所有wordDict里面的词和频数
            if count>0:
                idfDict[word] +=1 #如果频数大于1，则加1，即计算所有文档里面的词的频数
    for word,Ni in idfDict.items():
        idfDict[word] = math.log10((N+1)/(Ni+1)) #使用公式求出每个词对应的IDF值
    return idfDict
idfs = computeIDF([wordDictA,wordDictB])

#计算TFIDF
def computeTFIDF(tf,idfs):
    tfidf = dict.fromkeys(tf,0) #初始化一个新的tfidf字典
    for word,tfval in tf.items(): #遍历tf里面的词和词频
        tfidf[word] = tfval*idfs[word] #利用公式，将tf里面的词频除以idf里面的文档频率
    return tfidf
tfidfA = computeTFIDF(tfA,idfs)
tfidfB = computeTFIDF(tfB,idfs)
print(pd.DataFrame([tfidfA,tfidfB]))
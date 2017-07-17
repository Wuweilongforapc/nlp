# -*- coding: utf-8 -*- 
import os  
import pandas as pd
import jieba
import numpy as np
import re
import string
import pymysql
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.naive_bayes import MultinomialNB  
from sklearn.cross_validation import train_test_split   
from sklearn.cross_validation import StratifiedKFold  
from sklearn.cross_validation import KFold  
from sklearn.metrics import precision_recall_curve    
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2 
from sklearn import preprocessing
import datetime
import pickle
import multiprocessing
from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec
import codecs


starttime = datetime.datetime.now()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#==============================加载=============================        
def read_lines(filename):
    text=codecs.open(filename,'r',encoding='utf-8')
    content=text.read()
    text.close()
    return content


#=====================================================================
def fileWordProcess(contents,stopwords):  
    wordsList = []    
    contents = re.sub(r'\s+',' ',contents) # trans 多空格 to 空格  
    contents = re.sub(r'\n',' ',contents)  # trans 换行 to 空格  
    contents = re.sub(r'\t',' ',contents)  # trans Tab to 空格
    contents = re.sub(r'https:\/\/+.*',' ',contents)#去除url
    contents = re.sub(r'&+[a-zA-Z]*',' ',contents)
    contents = re.sub(r'\d',' ',contents)    
    for seg in jieba.cut(contents):  
        #seg = seg.encode('utf8')          #转化为utf-8后，文件格式不统一，下面的判断是无效的
        if seg not in stopwords:           # remove 停用词  
            if seg!=' ':                   # remove 空格  
                wordsList.append(seg)      # create 文件词列表  
    file_string = ' '.join(wordsList)              
    return file_string

#===============创建词向量矩阵，创建tfidf值矩阵============================
def normalization(x):
    x[x>1]=1  
    return x

#==============================================================================
# def MaxMinNormalization(x):
#     Min=np.min(x)
#     Max=np.max(x)
#     x = (x - Min) / (Max - Min);
#     return x;
#==============================================================================
def build_sentence_vector(text, size,imdb_w2v):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

def get_w2v(words,category_list):
    w2v=Word2Vec(size=300)
#==============================================================================
#     words_list=pd.DataFrame(words_list)
#     cw=lambda x:list(jieba.cut(x))
#     words_list['words']=words_list[0].apply(cw)
#     words=np.array(words_list['words'])
#==============================================================================
    x=words    
    le = preprocessing.LabelEncoder()
    y=le.fit_transform(category_list)
    w2v.build_vocab(x)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    w2v.train(x_train,total_examples=len(x_train),epochs=5)
    x_train=np.concatenate([build_sentence_vector(i,300,w2v) for i in x_train])
    print('x_train')
    w2v.train(x_test,total_examples=len(x_test),epochs=5)
    x_test=np.concatenate([build_sentence_vector(j,300,w2v) for j in x_test])    
    print('x_test')
    return x_train,x_test,y_train,y_test
    



#==========================神经网络========================
from keras.models import Sequential
from keras.layers import Dense,Activation,Input,Dropout
from keras.optimizers import SGD,Adam
import keras
def nn_model():
    model=Sequential()
    model.add(Dense(200,input_shape=(300,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(150,input_shape=(300,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100,input_shape=(100,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50,input_shape=(80,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(30,input_shape=(50,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(18,input_shape=(40,)))
    model.add(Activation('softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    return model
         
def get_words_list(text,stopwords):
    try:
        #print '------>' + ' ' + str(i)
        seg_sent = fileWordProcess(text,stopwords)   # 分词            
    except:
        print('meet problem')
    return seg_sent
           
#=============================构建训练数据============================
if __name__=='__main__':
    cw=lambda x:list(jieba.cut(x))
    stopwords = read_lines("D:/chinese_stopword.txt")
    words_list = []                                      
    filename_list = []  
    category_list = []
    results=[]
    words=[]
    multiprocessing.freeze_support()
    pool=Pool(processes=3)
    rootpath="E:/训练数据3" 
    category = os.listdir(rootpath)
    for categoryName in category:              
        categoryPath = os.path.join(rootpath,categoryName) # 这个类别的路径  
        filesList = os.listdir(categoryPath)      # 这个类别内所有文件列表  
        for filename in filesList:
            print(filename)
            contents = pd.read_table(categoryPath+'/'+filename,header=None,encoding='utf-8')
            contents['words']=contents[1].apply(cw)
            for word in contents['words']:
                words.append(word)
                category_list.append(categoryName)




     y_pre=clf.predict(x_test)
     print (classification_report(y_test,y_pre))
     cm=confusion_matrix(y_test,y_pre)


endtime = datetime.datetime.now()
print((endtime - starttime).seconds)





















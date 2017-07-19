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
from keras.preprocessing.sequence import pad_sequences


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
                  
    return wordsList






def transform_to_matrix(words, padding_size, vec_size,w2v_model):
    res = []
    for sen in words:
        matrix = []
        for i in range(padding_size):
            try:
                matrix.append(w2v_model[sen[i]].tolist())
            except:
                matrix.append([0] * vec_size)
        res.append(matrix)
    return res

def get_w2v(words,category_list,size):
    w2v=Word2Vec(size=size)
    x=words    
    le = preprocessing.LabelEncoder()
    y=le.fit_transform(category_list)
    w2v.build_vocab(x)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    w2v.train(x_train,total_examples=len(x_train),epochs=5)
    x_train=transform_to_matrix(words=x_train,padding_size=250,vec_size=size,w2v_model=w2v)
    x_train=np.array(x_train)
    x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2],1)
    print('x_train')
    w2v.train(x_test,total_examples=len(x_test),epochs=5)
    x_test=transform_to_matrix(words=x_test,padding_size=250,vec_size=size,w2v_model=w2v)    
    x_test=np.array(x_test)
    x_test=x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1)
    print('x_test')
    return x_train,x_test,y_train,y_test
    



#==========================神经网络========================
from keras.models import Sequential
from keras.layers import Dense,Activation,Input,Dropout,Convolution2D,Flatten,MaxPooling2D
from keras.optimizers import SGD,Adam
import keras
def nn_model():
    model=Sequential()
    model.add(Convolution2D(10,5,5,input_shape=(250,300,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(15,5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(200,input_shape=(300,)))
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
    rootpath="E:/训练数据3" 
    category = os.listdir(rootpath)
    for categoryName in category:              
        categoryPath = os.path.join(rootpath,categoryName) # 这个类别的路径  
        filesList = os.listdir(categoryPath)      # 这个类别内所有文件列表  
        for filename in filesList:
            print(filename)
            contents = pd.read_table(categoryPath+'/'+filename,header=None,encoding='utf-8')
            content=contents[1]
            title=contents[0]
            for word in content:
                words.append(fileWordProcess(word,stopwords))
                category_list.append(categoryName)
#==============================================================================
#             contents['title']=contents[0].apply(cw)
#             for word in contents['title']:
#                 words.append(word)
#                 category_list.append(categoryName)
#==============================================================================
                


    x_train,x_test,y_train,y_test=get_w2v(words,category_list,300)
    model=nn_model()
    y_train_nn=keras.utils.to_categorical(y_train,18)
    model.fit(x_train,y_train_nn,batch_size=100,epochs=100)
    y_pre=model.predict(x_test)
    y_pre_class=model.predict_classes(x_test)
    print (classification_report(y_test,y_pre_class))
    cm=confusion_matrix(y_test,y_pre_class)


endtime = datetime.datetime.now()
print((endtime - starttime).seconds)





















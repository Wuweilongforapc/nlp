# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 20:37:03 2017

@author: Administrator
"""

import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
import jieba
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
import joblib




def load_file_and_preprocessing():
    neg=pd.read_excel('E:/neg.xls',header=None,index=None)
    pos=pd.read_excel('E:/pos.xls',header=None,index=None)

    cw = lambda x: list(jieba.cut(x))
    pos['words'] = pos[0].apply(cw)
    neg['words'] = neg[0].apply(cw)

    #print pos['words']
    #use 1 for positive sentiment, 0 for negative
    y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))

    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos['words'], neg['words'])), y, test_size=0.2)
    
    np.save('E:/y_train.npy',y_train)
    np.save('E:/y_test.npy',y_test)
    return x_train,x_test


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

#计算词向量
def get_train_vecs(x_train,x_test):
    n_dim = 300
    #初始化模型和词表
    imdb_w2v = Word2Vec(size=n_dim, min_count=10)
    imdb_w2v.build_vocab(x_train)
    
    #在评论训练集上建模(可能会花费几分钟)
    imdb_w2v.train(x_train)
    
    train_vecs = np.concatenate([build_sentence_vector(z, n_dim,imdb_w2v) for z in x_train])
    #train_vecs = scale(train_vecs)
    
    np.save('E:/train_vecs.npy',train_vecs)
    print (train_vecs.shape)
    #在测试集上训练
    imdb_w2v.train(x_test)
    imdb_w2v.save('E:/w2v_model/w2v_model.pkl')
    #Build test tweet vectors then scale
    test_vecs = np.concatenate([build_sentence_vector(z, n_dim,imdb_w2v) for z in x_test])
    #test_vecs = scale(test_vecs)
    np.save('E:/test_vecs.npy',test_vecs)
    print (test_vecs.shape)
    
def get_data():
    train_vecs=np.load('E:/train_vecs.npy')
    y_train=np.load('E:/y_train.npy')
    test_vecs=np.load('E:/test_vecs.npy')
    y_test=np.load('E:/y_test.npy') 
    return train_vecs,y_train,test_vecs,y_test




def svm_train(train_vecs,y_train,test_vecs,y_test):
    clf=SVC(kernel='rbf',verbose=True)
    clf.fit(train_vecs,y_train)
    joblib.dump(clf, 'E:/svm_model/model.pkl')
    print (clf.score(test_vecs,y_test))
    
def get_predict_vecs(words):
    n_dim = 300
    imdb_w2v = Word2Vec.load('E:/w2v_model/w2v_model.pkl')
    #imdb_w2v.train(words)
    train_vecs = build_sentence_vector(words, n_dim,imdb_w2v)
    #print train_vecs.shape
    return train_vecs

def svm_predict(string):
    words=jieba.lcut(string)
    words_vecs=get_predict_vecs(words)
    clf=joblib.load('E:/svm_model/model.pkl')
     
    result=clf.predict(words_vecs)
    
    if int(result[0])==1:
        print(' positive')
    else:
        print(' negative')
        
string='电池充完了电连手机都打不开.简直烂的要命.真是金玉其外,败絮其中!连5号电池都不如'
#string='牛逼的手机，从3米高的地方摔下去都没坏，质量非常好'    
svm_predict(string)

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 15:49:56 2017
实时监测数据库中有无新ID出现，若出现新ID，将其内容进行分类，并将分类结果返回数据库
@author: wwl
"""
import os  
import pandas as pd
import jieba
import numpy as np
import re
import string
import MySQLdb
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
from sklearn.metrics import confusion_matrix,classification_report 
from sklearn.feature_selection import SelectKBest, chi2 
from sklearn import preprocessing
import datetime
import pickle
import time
from multiprocessing import Pool
import multiprocessing

#==============================加载=============================        
def read_lines(filename):
	fp = open(filename, 'r')
	lines = []
	for line in fp.readlines():
		line = line.strip()
		line = line.decode("utf-8")
		lines.append(line)
	fp.close()
	return lines

stopwords = read_lines("D:/chinese_stopword.txt") 
#==============================================================
def fileWordProcess(contents,stopwords):  
    wordsList = []
    contents = re.sub(r'\[0-9]*','',contents)
    contents = re.sub(r'\s+',' ',contents) # trans 多空格 to 空格  
    contents = re.sub(r'\n',' ',contents)  # trans 换行 to 空格  
    contents = re.sub(r'\t',' ',contents)  # trans Tab to 空格
    contents = re.sub(r'https:\/\/+.*',' ',contents)#去除url
    contents = re.sub(r'&+[a-zA-Z]*',' ',contents)
    
    for seg in jieba.cut(contents):  
        #seg = seg.encode('utf8')          #转化为utf-8后，文件格式不统一，下面的判断是无效的
        if seg not in stopwords:           # remove 停用词  
            if seg!=' ':                   # remove 空格  
                wordsList.append(seg)      # create 文件词列表  
    file_string = ' '.join(wordsList)              
    return file_string


#===========================创建词向量矩阵，创建tfidf值矩阵，预测用================

def normalization(x):
    x[x>1]=1  
    return x


def tfidf_mat_pre(words_list,df_columns):
    freWord = CountVectorizer(stop_words='english',max_features=10000)  
    transformer = TfidfTransformer()  
    fre_matrix = freWord.fit_transform(words_list)  
    tfidf = transformer.fit_transform(fre_matrix)
    feature_names = freWord.get_feature_names()           #
    tfidf_df = pd.DataFrame(np.zeros((1,10000)),index=['未知'.decode('utf-8')],columns=range(10000))
    tfidf_df.columns=df_columns
    le = preprocessing.LabelEncoder()
    tfidf_df['label'] = le.fit_transform(['未知'.decode('utf-8')])
    for strr in feature_names:
        for i in range(len(tfidf_df.columns)):
            if strr == tfidf_df.columns[i]:
                tfidf_df[tfidf_df.columns[i]] = 1
    ch2 = SelectKBest(chi2, k=10000)
    nolabel_feature = [x for x in tfidf_df.columns if x not in ['label']]      
    ch2_sx_np = ch2.fit_transform(tfidf_df[nolabel_feature],tfidf_df['label'])
    return ch2_sx_np

#=============================返回3个特征位置和预测值=============================
def get_top_3(pre_list):
    first=pre_list.argsort()[0,len(pre_list[0,:])-1]
    second=pre_list.argsort()[0,len(pre_list[0,:])-2]
    third=pre_list.argsort()[0,len(pre_list[0,:])-3]
    first_of_list=pre_list[0,first]
    second_of_list=pre_list[0,second]
    third_of_list=pre_list[0,third]
    return first,second,third,first_of_list,second_of_list,third_of_list

def encode_data(data):
    try:
        result=data.encode('utf-8')
    except:
        result=str(data)
    return result

#===========================进行分类并插入数据库======================================
def classfier_insertSql(content,stopwords,words_bag,category_list):
    try:              
        seg_sent = fileWordProcess(content,stopwords)
        words_list = [seg_sent]
        tfidf_df=tfidf_mat_pre(words_list,words_bag)
        clf = model
        #print clf.predict(tfidf_df)
        pre_list=clf.predict_proba(tfidf_df)
        first,second,third,first_of_list,second_of_list,third_of_list=get_top_3(pre_list)
        if first_of_list<0.1:
            classfier='其他'
        if first_of_list >0.1:
            if first_of_list/second_of_list>2:
                classfier='%s'%(category_list[first])
            if first_of_list/second_of_list<2:
                if first_of_list/third_of_list<2:
                    classfier='%s,%s,%s'%(category_list[first],category_list[second],category_list[third])
                else:
                    classfier='%s,%s'%(category_list[first],category_list[second])
        print classfier
    except:
        print 'classfier error'
        
    
    try:       
        data_new.loc[j,u'分类结果']=classfier 
        #print 'get clasfier'
        title=encode_data(data_new.ix[j,'标题'])
        bankuai=encode_data(data_new.ix[j,'板块'])
        comment=encode_data(data_new.ix[j,'内容'])
        sql_time=encode_data(data_new.ix[j,'时间'])
        url=encode_data(data_new.ix[j,'url'])
        province=encode_data(data_new.ix[j,'省'])
        city=encode_data(data_new.ix[j,'市'])
        qu=encode_data(data_new.ix[j,'区'])
        come_from=encode_data(data_new.ix[j,'网站来源'])
        ITbankuai=encode_data(data_new.ix[j,'网站板块'])
        insert_time=encode_data(data_new.ix[j,'插入时间'])
        cur = conn.cursor()             
        cur.execute("insert into 分类结果总表 (标题,版块,内容,时间,url,省,市,区,网站来源,网站版块,插入时间,分类结果) values('%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s')"%(title,bankuai,comment,sql_time,url,province,city,qu,come_from,ITbankuai,insert_time,classfier))            
        print 'insert yet'
        cur.close()
    except:
        print 'insert error'        
    

        
    

if __name__ == '__main__':
    multiprocessing.freeze_support()
    pool=Pool(processes=3)
    count_err=0
    model=joblib.load('D:/work/model/normal_extratree100.m')
    category_list=['仓储物流','信贷融资','农业服务业','医疗健康','大数据','扶贫',
        '政策','新能源','星创天地','林业服务业','法规','渔业服务业','电子商务','畜牧服务业'
        ,'社交','科技成果','科技特派员','证券投资','财政公开','通知公告','金融资讯']
    words_bag=pickle.load(open('D:/pickle/df_columns1.txt','r'))
#==============================================================================
#     content_list={'anhui_content':0,'beijing_content':0,'chongqing_content':0,'fujian_content':0,'gansu_content':0,
#                   'guangdong_content':0,'guangxi_content':0,'guizhou_content':0,'hainan_content':0,'hebei_content':0,
#                   'heilongjiang_content':0,'henan_content':0,'hebei_content':0,'hunan_content':0,'jiangshu_content':0,
#                   'jiangxi_content':0,'jilin_content':0,'liaoning_content':0,'ningxia_content':0,'shan_xi_content':0,
#                   'shandong_content':0,'shanghai_content':0,'shanxi_content':0,'tianjin_content':0,'xinjiang_content':0,
#                   'xizang_content':0,'yunnan_content':0,'zhejiang_content':0,'guojia_content':0}
#==============================================================================
    content_list={'beijing_content':0,'chongqing_content':0} # //测试用
    while True:        
        for i in content_list:
            time.sleep(1)
            conn = MySQLdb.connect(
                host='60.191.74.66',
                port = 3306,
                user='lwj',
                passwd='123456',
                db ='zhejiang_zixun',
                charset='utf8'
                )
            table_name = str(i)
            #print table_name
            ID_last=content_list[i]
            #print ID_last
            sqlcmd = "SELECT * FROM " + '%s where id>%s and id<%s;'%(table_name ,ID_last,ID_last+10)
            data = pd.read_sql(sqlcmd,conn)
            print 'get data'
            conn.close()
            ID = data[data.columns[0]]
            #print ID.max()
            if ID_last == ID.max():            
                print 'ID_last==ID.max()'
            else:
                #print ID_last
                
                data_new=data
                data_new=data_new.drop(['id'],axis=1)
                contents=data_new['内容']
                content_list[i]=ID.max()
                #print 'new_idlast'+str(content_list[i])
                #print 'get contents'                
                conn = MySQLdb.connect(
                    host='127.0.0.1',
                    port = 3306,
                    user='root',
                    passwd='root',
                    db ='mysql',
                    charset='utf8'
                    )  
                for j in range(len(contents)):
                    content=contents[j]
                    #print j
                    #print ID_last
                    try:
                        classfier_insertSql(content,stopwords,words_bag,category_list)
                    except:
                        print None                                                                                                 
                conn.close()
        

    











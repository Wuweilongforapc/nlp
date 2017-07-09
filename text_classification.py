# -*- coding: utf-8 -*- 
  
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

stopwords = read_lines("stop_words.txt") 
#=====================================================================
def fileWordProcess(contents):  
    wordsList = []  
    contents = re.sub(r'\s+',' ',contents) # trans 多空格 to 空格  
    contents = re.sub(r'\n',' ',contents)  # trans 换行 to 空格  
    contents = re.sub(r'\t',' ',contents)  # trans Tab to 空格    
    for seg in jieba.cut(contents):  
        seg = seg.encode('utf8')  
        if seg not in stopwords:           # remove 停用词  
            if seg!=' ':                   # remove 空格  
                wordsList.append(seg)      # create 文件词列表  
    file_string = ' '.join(wordsList)              
    return file_string

#===============创建词向量矩阵，创建tfidf值矩阵============================
def normalization(x):
    x[x>1]=1  
    return x

def tfidf_mat(words_list,filename_list,category_list):
    freWord = CountVectorizer(stop_words='english')  
    transformer = TfidfTransformer()  
    fre_matrix = freWord.fit_transform(words_list)  
    tfidf = transformer.fit_transform(fre_matrix)
    feature_names = freWord.get_feature_names()           # 特征名  
    freWordVector_df = pd.DataFrame(fre_matrix.toarray()) # 全词库 词频 向量矩阵  
    tfidf_df = pd.DataFrame(tfidf.toarray())              # tfidf值矩阵    
    tfidf_sx_featuresindex = tfidf_df.sum(axis=0).sort_values(ascending=False)[:10000].index 
    freWord_tfsx_df = freWordVector_df.ix[:,tfidf_sx_featuresindex] # tfidf法筛选后的词向量矩阵  
    df_columns = pd.Series(feature_names)[tfidf_sx_featuresindex]     
    tfidf_df_1 = freWord_tfsx_df.apply(normalization)  
    tfidf_df_1.columns = df_columns    
    le = preprocessing.LabelEncoder()  
    tfidf_df_1['label'] = le.fit_transform(category_list)  
    tfidf_df_1.index = filename_list
    return tfidf_df_1,df_columns
#===========================卡方检验=========================== 
def Chi_square_test(tfidf_df_1):
    ch2 = SelectKBest(chi2, k=10000)  
    nolabel_feature = [x for x in tfidf_df_1.columns if x not in ['label']]  
    ch2_sx_np = ch2.fit_transform(tfidf_df_1[nolabel_feature],tfidf_df_1['label'])  
    label_np = np.array(tfidf_df_1['label']) 
    X = ch2_sx_np  
    y = label_np  
    skf = StratifiedKFold(y,n_folds=10)  
    y_pre = y.copy()  
    for train_index,test_index in skf:  
        X_train,X_test = X[train_index],X[test_index]  
        y_train,y_test = y[train_index],y[test_index]  
        clf = MultinomialNB().fit(X_train, y_train)    
        y_pre[test_index] = clf.predict(X_test)      
        joblib.dump(clf, "train_model.m")
        print '准确率为 %.6f' %(np.mean(y_pre == y))
    return y,y_pre,nolabel_feature

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):  
    plt.imshow(cm, interpolation='nearest', cmap=cmap)  
    plt.title(title)  
    plt.colorbar()  
    tick_marks = np.arange(len(category[0:]))  
    category_english=['agriculture','house','education','environment','finance']  
    plt.xticks(tick_marks, category_english, rotation=45)  
    plt.yticks(tick_marks, category_english)  
    plt.tight_layout()  
    plt.ylabel('True label')  
    plt.xlabel('Predicted label')  
    for x in range(len(cm)):   
        for y in range(len(cm)):  
            plt.annotate(cm[x,y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
            
#=============================构建训练数据============================
words_list = []                                      
filename_list = []  
category_list = []
rootpath="C:/Users/admin/Desktop/文本分类/训练数据"    
category = os.listdir(rootpath.decode('utf-8'))
for categoryName in category:              
    categoryPath = os.path.join(rootpath.decode('utf-8'),categoryName) # 这个类别的路径  
    filesList = os.listdir(categoryPath)      # 这个类别内所有文件列表  
    for filename in filesList:
        contents = pd.read_table(categoryPath+'/'+filename,header=None,encoding='utf-8')
        context = contents[1]
        for i in range(len(context)):
            print '------>' + ' ' + str(i)
            seg_sent = fileWordProcess(context[i])   # 分词            
            words_list.append(seg_sent)  
            filename_list.append(filename)  
            category_list.append(categoryName) 
print '<-------构建结束------->'
tfidf_df_1,df_columns = tfidf_mat(words_list,filename_list,category_list)
y,y_pre,nolabel_feature = Chi_square_test(tfidf_df_1)
#===========================精准率 召回率 F1score 混淆矩阵===========================  
print 'precision,recall,F1-score如下：》》》》》》》》'  
print classification_report(y,y_pre)                   
cm = confusion_matrix(y,y_pre)  
plt.figure()  
plot_confusion_matrix(cm)    
plt.show()
#===========================预测===========================

'''
conn = MySQLdb.connect(
            host='60.191.74.66',
            port = 3306,
            user='lwj',
            passwd='123456',
            db ='china_news',
            charset='utf8'
            )
table_names = "china_content"
sqlcmd = "SELECT * FROM " + table_names
data = pd.read_sql(sqlcmd,conn)
ID = data[data.columns[0]]
sources = data[data.columns[9]]
titles = data[data.columns[1]]
contexts = data[data.columns[3]] 
 
for j in range(120,200):
    seg_sent = fileWordProcess(contexts[j])
    words_list = [seg_sent]
    #filename_list.append(titles[j]) 
    freWord = CountVectorizer(stop_words='english') 
    transformer = TfidfTransformer()  
    fre_matrix = freWord.fit_transform(words_list)  
    tfidf = transformer.fit_transform(fre_matrix)
    features = freWord.get_feature_names()       
    tfidf_df_v = pd.DataFrame(np.zeros((1,10000)),index=['未知'.decode('utf-8')],columns=range(10000))
    tfidf_df_v.columns = df_columns 
    le = preprocessing.LabelEncoder()
    tfidf_df_v['label'] = le.fit_transform(['未知'.decode('utf-8')])
    for str in features:
        for i in range(len(tfidf_df_v.columns)):
            if str == tfidf_df_v.columns[i]:
                tfidf_df_v[tfidf_df_v.columns[i]] = 1
    ch2 = SelectKBest(chi2, k=10000)     
    ch2_sx_np = ch2.fit_transform(tfidf_df_v[nolabel_feature],tfidf_df_v['label'])
    clf = joblib.load("train_model.m")
    if clf.predict(ch2_sx_np) == 0:
        print '农业'
    if clf.predict(ch2_sx_np) == 1:
        print '房产'
    if clf.predict(ch2_sx_np) == 2:  
        print '教育'
    if clf.predict(ch2_sx_np) == 3:  
        print '环保'
    if clf.predict(ch2_sx_np) == 4:  
        print '金融'


'''








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




starttime = datetime.datetime.now()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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
#=====================================================================
def fileWordProcess(contents,stopwords):  
    wordsList = []
    contents = re.sub(r'\d',' ',contents)
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


def tfidf_mat(words_list,filename_list,category_list):
    freWord = CountVectorizer(stop_words='english',decode_error='ignore',max_features=20000)  
    transformer = TfidfTransformer()
    fre_matrix = freWord.fit_transform(words_list)  
    tfidf = transformer.fit_transform(fre_matrix)
    #tfidf = Normalizer.fit_transform(fre_matrix)
    feature_names = freWord.get_feature_names()           # 特征名  
    freWordVector_df = pd.DataFrame(fre_matrix.toarray()) # 全词库 词频 向量矩阵  
    freWordVector_df
    tfidf_df = pd.DataFrame(tfidf.toarray())              # tfidf值矩阵    
    tfidf_sx_featuresindex = tfidf_df.sum(axis=0).sort_values(ascending=True)[:20000].index 
    
    freWord_tfsx_df = freWordVector_df.ix[:,tfidf_sx_featuresindex] # tfidf法筛选后的词向量矩阵  
    normal_df=Normalizer().fit_transform(freWord_tfsx_df)
    df_columns = pd.Series(feature_names)     
    tfidf_df_1 = pd.DataFrame(normal_df)       
    tfidf_df_1.columns = df_columns    
    le = preprocessing.LabelEncoder()
    tfidf_df_1['label'] = le.fit_transform(category_list)
    tfidf_df
    #tfidf_df_1.index = filename_list      
    return tfidf_df_1,df_columns


#==========================神经网络========================
from keras.models import Sequential
from keras.layers import Dense,Activation,Input
from keras.optimizers import SGD
import keras
def nn_model():
    model=Sequential()
    model.add(Dense(1000,input_shape=(20000,)))
    model.add(Activation('relu'))
    model.add(Dense(100,input_shape=(1000,)))
    model.add(Activation('relu'))
    model.add(Dense(19,input_shape=(100,)))
    model.add(Activation('softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model
#==============================================================================
#     ch2 = SelectKBest(chi2, k=10000)
#     nolabel_feature = [x for x in tfidf_df_1.columns if x not in ['label']]  
#     ch2_sx_np = ch2.fit_transform(tfidf_df_1[nolabel_feature],tfidf_df_1['label'])  
#     label_np = np.array(tfidf_df_1['label']) 
#     X = ch2_sx_np  
#     y = label_np  
#     #skf = StratifiedKFold(y,n_folds=5)  #交叉验证，产生10个测试集
#     #y_pre = y.copy()  
#     #for train_index,test_index in skf:  
#     X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
#     y_train=keras.utils.to_categorical(y_train,num_classes=19)
#     clf=nn_model().fit(X_train,y_train,batch_size=50,epochs=200)
#     y_pre=clf.predict(X_test)
#     
#     return y_pre
#==============================================================================

    
#===========================卡方检验=========================== 

def Chi_square_test(tfidf_df_1):
    ch2 = SelectKBest(chi2, k=20000)
    nolabel_feature = [x for x in tfidf_df_1.columns if x not in ['label']]  
    ch2_sx_np = ch2.fit_transform(tfidf_df_1[nolabel_feature],tfidf_df_1['label'])  
    label_np = np.array(tfidf_df_1['label']) 
    X = ch2_sx_np  
    y = label_np  
    #skf = StratifiedKFold(y,n_folds=5)  #交叉验证，产生10个测试集
    #y_pre = y.copy()  
    #for train_index,test_index in skf:  
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
    #clf = MultinomialNB(alpha=0.5).fit(X_train, y_train)
    #clf=RandomForestClassifier(n_estimators=30).fit(X_train, y_train)
    #clf=ExtraTreesClassifier(n_estimators=20).fit(X_train,y_train)
    #clf=SVC().fit(X_train,y_train)
    #clf=GradientBoostingClassifier().fit(X_train,y_train)
    #y_pre = clf.predict(X_test)
    clf=RandomForestClassifier(n_estimators=30).fit(X_train, y_train)
    y_pre=clf.predict(X_test)      
    joblib.dump(clf, "D:/work/model/normal_extratree100.m",compress=3)
    print '准确率为 %.6f' %(np.mean(y_pre == y_test))
    return y_test,y_pre,nolabel_feature



#==============================================================================
# def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):  
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)  
#     plt.title(title)  
#     plt.colorbar()  
#     tick_marks = np.arange(len(category[0:]))
#     category_list=['仓储物流','信贷融资','农业服务业','医疗健康','大数据','扶贫',
#         '政策','新能源','星创天地','林业服务业','法规','渔业服务业','电子商务','畜牧服务业'
#         ,'社交','科技成果','科技特派员','证券投资','财政公开','通知公告','金融资讯']
#     plt.xticks(tick_marks, category_list, rotation=45)  
#     plt.yticks(tick_marks, category_list)  
#     plt.tight_layout()  
#     plt.ylabel('True label')  
#     plt.xlabel('Predicted label')  
#     for x in range(len(cm)):   
#         for y in range(len(cm)):  
#             plt.annotate(cm[x,y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
#==============================================================================

    
#==============================================================================
#     category_english=[]
#     category_english=['o2o','Cultivation_of_Chinese_medicinal_herbs','Warehousing_logistics','Enterprise_service',
#                       'sport_industry','Insurance','Credit_financing','Other_animal_husbandry','Agricultural_Service_Industry','Agricultural_Service_Industry'
#                       ,'Agricultural_Service_Industry','Fund_financing','bigdata',
#                       'preschool_education','Poultry_raising','Advertising_marketing','Video_entertainment','Adult_Education',
#                       'Home_improvement','Poverty_alleviation','policy','Educational_informatization','new_energy','Tourism_integration',
#                       'Star_creation_world','Intelligent_investment_adviser','Intelligent_hardware','Harvesting_of_timber_and_bamboo',
#                       'Forestry_Service_Industry','Forest_product_collection','Tree_breeding_and_seedling_raising','Cotton_linen_sugar_tobacco_growing','Forest_management_and_conservation',
#                       'Aquiculture','Fishery_capture','Fruit_growing','Auto_market','statute','Fishery_Services','Livestock_raising',
#                       'Electronic_Commerce','Livestock_Service_Industry','Social_contact','Scientific_achievements','Commissioner_for_science_and_technology',
#                       'Comprehensive_logistics','Vegetables_edible_fungi_and_horticultura_crops_are_grown','Portfolio_investment',
#                       'Grain_planting','Legumes_oilseeds_and_potato_growing','Public_finance','Cross-border_electricity_supplier',
#                       'Notice','Reforestation_and_regeneration','Hotel_integration','Financial_information']'''
#==============================================================================
#==============================================================================
#     category_english=['o2o','Internet_application_service','hoPEnterprise_serviceublic_financeuse',
#     'sport_industry','Insurance','Credit_financing','Agricultural_Service_Industry',
#     'Medical_health','preschool_education','Advertising_marketing','Video_entertainment','PoveViAdult_Education',
#     'Home_improvement','Poverty_alleviationing','policy','Educational_informatization',
#     'new_energy','Tourism_integration','Star_creation_world','Intelligent_hardware',
#     'Forest_product_collection','Fishery_capture','HotAuto_market','statute','Comprehensive_logistics',
#     'Fund_financing','ForeLivestock_Service_Industry','Social_contact','Cultivation_herbs',
#     'Scientific_achievements','Commissioner_for_science_and_technology','Public_finance',
#     'Cross-border_electricity_supplier','Notice','Hotel_integration']  
#==============================================================================

            
            
def get_words_list(text,stopwords):
    try:
        #print '------>' + ' ' + str(i)
        seg_sent = fileWordProcess(text,stopwords)   # 分词            
    except:
        print '------>has problem'
    return seg_sent
    
            
#=============================构建训练数据============================
if __name__=='__main__':
     
    stopwords = read_lines("D:/chinese_stopword.txt")
    words_list = []                                      
    filename_list = []  
    category_list = []
    results=[]
    multiprocessing.freeze_support()
    pool=Pool(processes=3)
    rootpath="D:/标签资讯" 

    category = os.listdir(rootpath.decode('utf-8'))
    for categoryName in category:              
        categoryPath = os.path.join(rootpath.decode('utf-8'),categoryName) # 这个类别的路径  
        filesList = os.listdir(categoryPath)      # 这个类别内所有文件列表  
        for filename in filesList:
            print filename
            contents = pd.read_table(categoryPath+'/'+filename,header=None,encoding='utf-8')
            context = contents[1]
            context=context.dropna()            
            for i in range(len(context)):
                try:
                    text=context[i]
                    seg_sent =pool.apply_async(get_words_list,(text,stopwords,))   # 分词
                    #print'get'+str(i)
                    results.append(seg_sent)
                    filename_list.append(filename)  
                    category_list.append(categoryName)
                except:
                    print 'meet na'
            
   
    print 'get results...'
    pool.close()
    pool.join()
    for result in results:
        words_list.append(result.get())
        print 'get'+str(result)
                
    pickle.dump(words_list,open('D:/pickle/word_list_labeldata07091.txt','w'))
    #pickle.dump(words_list,open('D:/pickle/word_list_labeldata1.txt','w'))
    #words_list=pickle.load(open('D:/pickle/word_list_labeldata07091.txt'','r'))
    print '<-------构建结束------->'
    tfidf_df_1,df_columns = tfidf_mat(words_list,filename_list,category_list)
    print '-------载入数据集至本地----->'
    pickle.dump(df_columns,open('D:/pickle/df_columns1.txt','w'))
    #df_columns.to_csv('E:/label_data_columns1.csv',encoding='utf-8')
    #y_test,y_pre,nolabel_feature = nn_model(tfidf_df_1)
    #===========================精准率 召回率 F1score 混淆矩阵===========================  
#==============================================================================
#     print 'precision,recall,F1-score如下：>>>>>>>>'  
#     print classification_report(y_test,y_pre)                   
#     cm = confusion_matrix(y_test,y_pre) 
#     print cm
#==============================================================================
    #plt.figure(figsize=(50,50))      
    #plot_confusion_matrix(cm)    
    #plt.show()



endtime = datetime.datetime.now()

print (endtime - starttime).seconds





















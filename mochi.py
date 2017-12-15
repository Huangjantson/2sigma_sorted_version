# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 10:52:11 2017

@author: Raku

functions for 2sigma comptetition
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import string
from sklearn.cluster import KMeans
import lightgbm as lgb
import xgboost as xgb

class CVstatistics(object):
    
    """
    self.result : the result dataframe storing the cv results
    self.endpoint : the first ending point for the validations
    self.turns: the turns for each validation
    
    validCurve : plot the validation curve,stop at the first endpoint
    errorsAt: return the average errors at a certain turn
    """
    def __init__(self,result_dict,metric,k=5):
        self.metric = metric
        if type(result_dict) == pd.DataFrame:
            self.result = result_dict
        else:
            temp_dict = {}
            for phase in ['train','test']:
                for turn in range(k):
                    temp_dict[phase+str(turn)]=result_dict[turn][phase][metric]
                    self.result=pd.DataFrame(dict([ (key,pd.Series(v)) for key,v in temp_dict.iteritems()]))    
        
        self.endpoint =len(self.result.filter(like = 'train').dropna())
        
        self.turns = self.result.filter(like = 'test').\
            apply(lambda x : ~np.isnan(x)).cumsum(axis=0).iloc[len(self.result)-1,:]

    def validCurve(self,start = 0, stop_at_first = True):
        if stop_at_first:
            eout = self.result.iloc[start:,:].filter(like = 'test').dropna().mean(axis=1)
            ein =  self.result.iloc[start:,:].filter(like = 'train').dropna().mean(axis=1)
        else:
            eout = self.result.iloc[start:,:].filter(like = 'test').mean(axis=1)
            ein =  self.result.iloc[start:,:].filter(like = 'train').mean(axis=1)
        plt.plot(map(lambda x :x+start,range(len(eout))), eout,
        map(lambda x :x+start,range(len(ein))), ein)
        plt.xlabel("turn")
        plt.ylabel(self.metric)
        plt.title('Validation Curve')
        
        plt.show()
    
    def eoutCurve(self,start = 0,stop_at_first = True):
        if stop_at_first:
            eout = self.result.iloc[start:,:].filter(like = 'test').dropna().mean(axis=1)
        else:
            eout = self.result.iloc[start:,:].filter(like = 'test').mean(axis=1)
        plt.plot(map(lambda x :x+start,range(len(eout))), eout)
        plt.xlabel("turn")
        plt.ylabel(self.metric)
        plt.title('Eout Curve')
        
        plt.show()

    def minAvgEout(self):
        meanTestError = self.result.filter(like='test').mean(axis=1)
        return meanTestError[meanTestError==np.min(meanTestError)]
    
    def errorsAt(self,turn):
        eout = self.result.filter(like = 'test').loc[turn].mean()
        ein = self.result.filter(like = 'train').loc[turn].mean()
        return eout,ein

def xgbImportance(model,factor_name):
    factors = model.get_score(importance_type=factor_name)
    factor_list = []
    total = sum(factors.values())
    for key in factors:
        factors[key] = factors[key]*1.0/total
        factor_list.append((key,factors[key]))
    return sorted(factor_list,key=lambda x : x[1],reverse=True)
    
def showFscore(model,normalize = True):
    factors = model.get_fscore()
    factor_list = []
    total = sum(factors.values())
    for key in factors:
        if normalize:
            factors[key] = factors[key]*1.0/total
        else:
            factors[key] = factors[key]
        factor_list.append((key,factors[key]))
    return sorted(factor_list,key=lambda x : x[1],reverse=True)
    
def removePunctuation(x):
    #filter the head or tail blanks
    x = re.sub(r'^\s+',r' ',x)
    x = re.sub(r'\s+$',r' ',x)
    
    # Lowercasing all words
    x = x.lower()
    # Removing non ASCII chars, warning if you are dealing with other languages!!!!!!!!!!!!!!!
    x = re.sub(r'[^\x00-\x7f]',r' ',x)
    #change all the blank to space
    x = re.sub(r'\s',r' ',x)
    # Removing (replacing with empty spaces actually) all the punctuations
    removing = string.punctuation#.replace('-','')# except '-'
    removed = re.sub("["+removing+"]", "", x)
    #removing the line-changing
    #removed = re.sub('\\n'," ",removed)    
    return removed

#feature processing functions
def proecessStreet(address):
    #remove the building number
    pattern = re.compile('^[\d-]*[\s]+')
    street = removePunctuation(pattern.sub('',address))
    
    #sub the st to street
    pattern = re.compile('( st)$')
    street = pattern.sub(' street',street)
    
    #sub the ave to avenue
    pattern = re.compile('( ave)$')
    street = pattern.sub(' avenue',street)
    
    pattern = re.compile('(\d+)((th)|(st)|(rd)|(nd))')
    street = pattern.sub('\g<1>',street)
    
    #deal with the w 14 street => west 14 street
    pattern = re.compile('(w)(\s+)(\d+)')    
    street = pattern.sub('west \g<3>',street)
    
    #deal with the e....
    pattern = re.compile('(e)(\s+)(\d+)')    
    street = pattern.sub('east \g<3>',street)
    
    return street
    
#from "this is a lit"s python version by rakhlin
def singleValueConvert(df1,df2,column,minimum_size=1):
    ps = df1[column].append(df2[column])
    grouped = ps.groupby(ps).size().to_frame().rename(columns={0: "size"})
    df1.loc[df1.join(grouped, on=column, how="left")["size"] <= minimum_size, column] = -1
    df2.loc[df2.join(grouped, on=column, how="left")["size"] <= minimum_size, column] = -1
    return df1, df2

#add ranking for this function
def performance_eval(train_df,test_df,feature,smoothing=True,k=5,g=1,f=1,
                     update_df =None,random = None):
    
    temp=pd.concat([train_df[feature],pd.get_dummies(train_df.interest_level)], axis = 1)\
         .groupby(feature).mean()
    
    new_feature = feature+'_perf'
    new_rank = feature+'_rank'
    new_nrank = feature+'_nrank'
    
    if smoothing:
        new_feature +='_s'
        new_rank +='_s'
        new_nrank +='_s'
        
    if random:
        new_feature +='_r'
        new_rank +='_r'
        new_nrank +='_r'
    
    temp.columns = ['tempHigh','tempLow', 'tempMed']
    
    temp[feature+'_origin'] = temp['tempHigh']*2 + temp['tempMed']
    mean_values = temp.loc[:, feature+'_origin'].mean()

    temp['count'] = train_df.groupby(feature).count().iloc[:,1]
    if smoothing:
        temp["lambda"] = g / (g + np.exp((k - temp["count"] )/f))
        temp[new_feature] = temp["lambda"]*temp[feature+'_origin']+(1-temp["lambda"])*mean_values
    else:
        temp[new_feature] = temp[feature+'_origin']
        
    temp[new_rank]=temp[new_feature].rank()
    temp[new_nrank]=temp[new_rank]/temp['count']
    
    # Add uniform noise. Not mentioned in original paper.adding to each manager
    if random:
        temp[new_feature] *= np.random.uniform(1 - random, 1 + random, len(temp))     

    value = test_df[[feature]].join(temp, on=feature, how="left")[[new_feature,new_rank,new_nrank]].fillna(mean_values)
    
    if update_df is None: update_df = test_df
    if new_feature not in update_df.columns: update_df[new_feature] = np.nan
    if new_rank not in update_df.columns: update_df[new_rank] = np.nan
    if new_nrank not in update_df.columns: update_df[new_nrank] = np.nan

    update_df.update(value)
    
    
#hcc encoding based on the targets
def hcc_scoring(train_df,test_df,feature,labelValue,randomize=0.01,k=5,f=1,g=1,update_df =None):    
    #input is the train dataframe with its labels mapped to dummies
    #such as:
    tempTrain = train_df.join(pd.get_dummies(train_df[u'interest_level']).astype(int))
    
    new_feature = '_'.join(['hcc',feature,labelValue])
    
    #take the mean  for the feature on the given featureValue which is mapped to dummies
    prob = tempTrain[labelValue].mean()
    
    #take the mean and count for each feature value
    grouped = tempTrain.groupby(feature)[labelValue].agg({'count':'size','mean':'mean'})
    
    #perform the transform for lambda and the final score
    grouped["lambda"] = 1 / (g + np.exp((k - grouped["count"]) / f))
    grouped[new_feature] = grouped['lambda']*grouped['mean']+(1-grouped['lambda'])*prob
    
    #adding to the test_df
    update_value  = test_df[[feature]].join(grouped,on = feature,how='left')[new_feature].fillna(prob)
    
    if randomize : update_value *= np.random.uniform(1 - randomize, 1 + randomize, len(test_df))
        
    #adding some noise to the new 
    #print 'New feature added:'+new_feature

    if update_df is None:
        update_df = test_df
    if new_feature not in update_df.columns: 
        update_df[new_feature] = np.nan
        
    update_df.update(update_value)
    return    
    
#functions for features
def featureList(train_df,test_df,limit = 0.001):
    #acquiring the feature lists
    features_in_train = train_df["features"].apply(pd.Series).unstack().reset_index(drop = True).dropna().value_counts()
    features_in_test = test_df["features"].apply(pd.Series).unstack().reset_index(drop = True).dropna().value_counts()
    
    filtered_features_in_train = features_in_train[features_in_train > limit*len(train_df)]
    filtered_features_in_test = features_in_test[features_in_test > limit*len(test_df)]
    accept_list = set(filtered_features_in_train.index).union(set(filtered_features_in_test.index))
    return accept_list

def featureMapping(train_df,test_df,feature_list):
    for feature in feature_list:
        #add the feature column for both
        #if feature in the row, then set the value for (row,feature) to 1
        train_df['with_'+feature]=train_df['features'].apply(lambda x : 1 if feature in x else 0)
        test_df['with_'+feature]=test_df['features'].apply(lambda x : 1 if feature in x else 0)
    return

#new function for clustering
def getCluster(train_df,test_df,k):
    train_test = pd.concat([train_df.drop('interest_level',axis=1),test_df])
    #processMap(train_test)
    cluster = KMeans(k,random_state = 2333)
    cluster.fit(train_test[['latitude', 'longitude']].dropna())
    train_df['cluster_id_'+str(k)]=cluster.predict(train_df[['latitude', 'longitude']].fillna(-1))
    test_df['cluster_id_'+str(k)]=cluster.predict(test_df[['latitude', 'longitude']].fillna(-1))
    train_df['cluster_id_'+str(k)+'_d']=np.amin(cluster.transform(train_df[['latitude', 'longitude']]),axis=1)
    test_df['cluster_id_'+str(k)+'_d']=np.amin(cluster.transform(test_df[['latitude', 'longitude']]),axis=1)
    
#setting the outliers to be nan. to be test
def processMap(df):
    for i in ['latitude', 'longitude']:
        Q1 = df[i].quantile(0.005)
        Q3 = df[i].quantile(0.995)
        upper = Q3
        lower = Q1
        df.ix[(df[i]>upper)|(df[i]<lower),i] = np.nan
        #df.ix[:,i] =  df[i].round(3) 
    return 
       
#try performance instead of high&medium
def temporalManagerPerf(train_df,test_df,update_df =None,filling=None):
    temp=pd.concat([train_df,pd.get_dummies(train_df.interest_level)], axis = 1)
    tempTrain = temp[['manager_id','dayofyear','high','low','medium']].set_index('manager_id')
    tempTest = test_df[['manager_id','dayofyear']]
    tempJoin = tempTest.join(tempTrain,on='manager_id',how='left', rsuffix='_toSum')
    
    #historical_performances
    #3 day performance
    #performance_3 = tempJoin[tempJoin['dayofyear'] - tempJoin['dayofyear_toSum']<4]
    performance_3 = tempJoin[(tempJoin['dayofyear'] - tempJoin['dayofyear_toSum']<4) & \
                     (tempJoin['dayofyear'] - tempJoin['dayofyear_toSum']>0) ]
    performance_3 = performance_3.groupby(performance_3.index).sum()[['high','low','medium']]
    performance_3['m3total'] = performance_3['high']+performance_3['low']+performance_3['medium']
    performance_3['m3perf'] = (2*performance_3['high']+performance_3['medium'])*1.0/performance_3['m3total']
    
    #performance_7 = tempJoin[tempJoin['dayofyear'] - tempJoin['dayofyear_toSum']<8]
    performance_7 = tempJoin[(tempJoin['dayofyear'] - tempJoin['dayofyear_toSum']<8) & \
                     (tempJoin['dayofyear'] - tempJoin['dayofyear_toSum']>0)  ]
    performance_7 = performance_7.groupby(performance_7.index).sum()[['high','low','medium']]
    performance_7['m7total'] = performance_7['high']+performance_7['low']+performance_7['medium']
    performance_7['m7perf'] = (2*performance_7['high']+performance_7['medium'])*1.0/performance_7['m7total']
    
    #performance_14 = tempJoin[tempJoin['dayofyear'] - tempJoin['dayofyear_toSum']<15]
    performance_14 = tempJoin[(tempJoin['dayofyear'] - tempJoin['dayofyear_toSum']<15) & \
                     (tempJoin['dayofyear'] - tempJoin['dayofyear_toSum']>0)  ]
    performance_14 = performance_14.groupby(performance_14.index).sum()[['high','low','medium']]
    performance_14['m14total'] = performance_14['high']+performance_14['low']+performance_14['medium']
    performance_14['m14perf'] = (2*performance_14['high']+performance_14['medium'])*1.0/performance_14['m14total']

    
    #performance_30 = tempJoin[tempJoin['dayofyear'] - tempJoin['dayofyear_toSum']<31]
    performance_30 = tempJoin[(tempJoin['dayofyear'] - tempJoin['dayofyear_toSum']<31) & \
                     (tempJoin['dayofyear'] - tempJoin['dayofyear_toSum']>0)  ]
    performance_30 = performance_30.groupby(performance_30.index).sum()[['high','low','medium']]
    performance_30['m30total'] = performance_30['high']+performance_30['low']+performance_30['medium']
    performance_30['m30perf'] = (2*performance_30['high']+performance_30['medium'])*1.0/performance_30['m30total']

    update = pd.concat([performance_3[['m3perf','m3total']],performance_7[['m7perf','m7total']],\
                        performance_14[['m14perf','m14total']],performance_30[['m30perf','m30total']]],axis=1)

    if update_df is None: update_df = test_df
    
    new_features = ['m3perf','m7perf','m14perf','m30perf','m3total','m7total','m14total','m30total']
    
    for f in new_features:
        if f not in update_df.columns: 
             update_df[f] = np.nan
    
    update_df.update(update)
    
    #filling with manager_id_perf
    if filling:
        for f in ['m3perf','m7perf','m14perf','m30perf']:
            update_df.loc[np.isnan(update_df[f]),f] = update_df[filling]
    for f in ['m3total','m7total','m14total','m30total']:
        update_df.loc[np.isnan(update_df[f]),f] = 0
        
    #future performances
    performance_3 = tempJoin[(tempJoin['dayofyear_toSum'] - tempJoin['dayofyear']<4) & \
                     (tempJoin['dayofyear_toSum'] - tempJoin['dayofyear']>0) ]
    performance_3 = performance_3.groupby(performance_3.index).sum()[['high','low','medium']]
    performance_3['m3total_f'] = performance_3['high']+performance_3['low']+performance_3['medium']
    performance_3['m3perf_f'] = (2*performance_3['high']+performance_3['medium'])*1.0/performance_3['m3total_f']

    
    performance_7 = tempJoin[(tempJoin['dayofyear_toSum'] - tempJoin['dayofyear']<8) & \
                     (tempJoin['dayofyear_toSum'] - tempJoin['dayofyear']>0)]
    performance_7 = performance_7.groupby(performance_7.index).sum()[['high','low','medium']]
    performance_7['m7total_f'] = performance_7['high']+performance_7['low']+performance_7['medium']
    performance_7['m7perf_f'] = (2*performance_7['high']+performance_7['medium'])*1.0/performance_7['m7total_f']
    
    performance_14 = tempJoin[(tempJoin['dayofyear_toSum'] - tempJoin['dayofyear']<14) & \
                     (tempJoin['dayofyear_toSum'] - tempJoin['dayofyear']>0) ]
    performance_14 = performance_14.groupby(performance_14.index).sum()[['high','low','medium']]
    performance_14['m14total_f'] = performance_14['high']+performance_14['low']+performance_14['medium']
    performance_14['m14perf_f'] = (2*performance_14['high']+performance_14['medium'])*1.0/performance_14['m14total_f']

    
    performance_30 = tempJoin[(tempJoin['dayofyear_toSum'] - tempJoin['dayofyear']<31) & \
                     (tempJoin['dayofyear_toSum'] - tempJoin['dayofyear']>0)]
    performance_30 = performance_30.groupby(performance_30.index).sum()[['high','low','medium']]
    performance_30['m30total_f'] = performance_30['high']+performance_30['low']+performance_30['medium']
    performance_30['m30perf_f'] = (2*performance_30['high']+performance_30['medium'])*1.0/performance_30['m30total_f']

    performance_d = tempJoin[tempJoin['dayofyear_toSum'] == tempJoin['dayofyear']]
    performance_d = performance_d.groupby(performance_d.index).sum()[['high','low','medium']]
    performance_d['mdaytotal'] = performance_d['high']+performance_d['low']+performance_d['medium']
    performance_d['mperf_day'] = (2*performance_d['high']+performance_d['medium'])*1.0/performance_d['mdaytotal']

    update = pd.concat([performance_3[['m3perf_f','m3total_f']],performance_7[['m7perf_f','m7total_f']],\
                        performance_14[['m14perf_f','m14total_f']],performance_30[['m30perf_f','m30total_f']],\
                        performance_d[['mperf_day','mdaytotal']]],axis=1)

    if update_df is None: update_df = test_df
    
    new_features = ['m3perf_f','m7perf_f','m14perf_f','m30perf_f','mperf_day',\
                    'm3total_f','m7total_f','m14total_f','m30total_f','mdaytotal']
    
    for f in new_features:
        if f not in update_df.columns: 
             update_df[f] = np.nan  
                
    update_df.update(update)
    
    #filling with manager_id_perf
    if filling:
        for f in ['m3perf_f','m7perf_f','m14perf_f','m30perf_f','mperf_day']:
            update_df.loc[np.isnan(update_df[f]),f] = update_df[filling]
    for f in ['m3total_f','m7total_f','m14total_f','m30total_f','mdaytotal']:
        update_df.loc[np.isnan(update_df[f]),f] = 0

#the old one only filtering the passed
def temporalManagerPerf_old(train_df,test_df,update_df =None):
    temp=pd.concat([train_df,pd.get_dummies(train_df.interest_level)], axis = 1)
    tempTrain = temp[['manager_id','dayofyear','high','low','medium']].set_index('manager_id')
    tempTest = test_df[['manager_id','dayofyear']]
    tempJoin = tempTest.join(tempTrain,on='manager_id',how='left', rsuffix='_toSum')
    
    #3 day performance
    performance_3 = tempJoin[tempJoin['dayofyear'] - tempJoin['dayofyear_toSum']<4]
    performance_3 = performance_3.groupby(performance_3.index).sum()[['high','low','medium']]
    performance_3['total'] = performance_3['high']+performance_3['low']+performance_3['medium']
    performance_3['m3perf'] = (2*performance_3['high']+performance_3['medium'])*1.0/performance_3['total']

    
    performance_7 = tempJoin[tempJoin['dayofyear'] - tempJoin['dayofyear_toSum']<8]
    performance_7 = performance_7.groupby(performance_7.index).sum()[['high','low','medium']]
    performance_7['total'] = performance_7['high']+performance_7['low']+performance_7['medium']
    performance_7['m7perf'] = (2*performance_7['high']+performance_7['medium'])*1.0/performance_7['total']
    
    performance_14 = tempJoin[tempJoin['dayofyear'] - tempJoin['dayofyear_toSum']<15]
    performance_14 = performance_14.groupby(performance_14.index).sum()[['high','low','medium']]
    performance_14['total'] = performance_14['high']+performance_14['low']+performance_14['medium']
    performance_14['m14perf'] = (2*performance_14['high']+performance_14['medium'])*1.0/performance_14['total']

    
    performance_30 = tempJoin[tempJoin['dayofyear'] - tempJoin['dayofyear_toSum']<31]
    performance_30 = performance_30.groupby(performance_30.index).sum()[['high','low','medium']]
    performance_30['total'] = performance_30['high']+performance_30['low']+performance_30['medium']
    performance_30['m30perf'] = (2*performance_30['high']+performance_30['medium'])*1.0/performance_30['total']

    update = pd.concat([performance_3[['m3perf']],performance_7[['m7perf']],\
                        performance_14[['m14perf']],performance_30[['m30perf']]],axis=1).fillna(-1)

    if update_df is None: update_df = test_df
    
    new_features = ['m3perf','m7perf','m14perf','m30perf']
    
    for f in new_features:
        if f not in update_df.columns: 
             update_df[f] = np.nan
    
    update_df.update(update)

#try performance instead of high&medium
def temporalManagerPerf_f(train_df,test_df,update_df =None,filling=None):
    temp=pd.concat([train_df,pd.get_dummies(train_df.interest_level)], axis = 1)
    tempTrain = temp[['manager_id','dayofyear','high','low','medium']].set_index('manager_id')
    tempTest = test_df[['manager_id','dayofyear']]
    tempJoin = tempTest.join(tempTrain,on='manager_id',how='left', rsuffix='_toSum')
    
    #historical_performances
    #3 day performance
    #performance_3 = tempJoin[tempJoin['dayofyear'] - tempJoin['dayofyear_toSum']<4]
    performance_3 = tempJoin[(tempJoin['dayofyear'] - tempJoin['dayofyear_toSum']<4)]
    performance_3 = performance_3.groupby(performance_3.index).sum()[['high','low','medium']]
    performance_3['m3total'] = performance_3['high']+performance_3['low']+performance_3['medium']
    performance_3['m3perf'] = (2*performance_3['high']+performance_3['medium'])*1.0/performance_3['m3total']
    
    #performance_7 = tempJoin[tempJoin['dayofyear'] - tempJoin['dayofyear_toSum']<8]
    performance_7 = tempJoin[(tempJoin['dayofyear'] - tempJoin['dayofyear_toSum']<8)]
    performance_7 = performance_7.groupby(performance_7.index).sum()[['high','low','medium']]
    performance_7['m7total'] = performance_7['high']+performance_7['low']+performance_7['medium']
    performance_7['m7perf'] = (2*performance_7['high']+performance_7['medium'])*1.0/performance_7['m7total']
    
    #performance_14 = tempJoin[tempJoin['dayofyear'] - tempJoin['dayofyear_toSum']<15]
    performance_14 = tempJoin[(tempJoin['dayofyear'] - tempJoin['dayofyear_toSum']<15)]
    performance_14 = performance_14.groupby(performance_14.index).sum()[['high','low','medium']]
    performance_14['m14total'] = performance_14['high']+performance_14['low']+performance_14['medium']
    performance_14['m14perf'] = (2*performance_14['high']+performance_14['medium'])*1.0/performance_14['m14total']

    
    #performance_30 = tempJoin[tempJoin['dayofyear'] - tempJoin['dayofyear_toSum']<31]
    performance_30 = tempJoin[(tempJoin['dayofyear'] - tempJoin['dayofyear_toSum']<31)]
    performance_30 = performance_30.groupby(performance_30.index).sum()[['high','low','medium']]
    performance_30['m30total'] = performance_30['high']+performance_30['low']+performance_30['medium']
    performance_30['m30perf'] = (2*performance_30['high']+performance_30['medium'])*1.0/performance_30['m30total']

    update = pd.concat([performance_3[['m3perf','m3total']],performance_7[['m7perf','m7total']],\
                        performance_14[['m14perf','m14total']],performance_30[['m30perf','m30total']]],axis=1)

    if update_df is None: update_df = test_df
    
    new_features = ['m3perf','m7perf','m14perf','m30perf','m3total','m7total','m14total','m30total']
    
    for f in new_features:
        if f not in update_df.columns: 
             update_df[f] = np.nan
    
    update_df.update(update)
    
    #filling with manager_id_perf
    if filling:
        for f in ['m3perf','m7perf','m14perf','m30perf']:
            update_df.loc[np.isnan(update_df[f]),f] = update_df[filling]
    for f in ['m3total','m7total','m14total','m30total']:
        update_df.loc[np.isnan(update_df[f]),f] = 0
        
    #future performances
    performance_3 = tempJoin[(tempJoin['dayofyear_toSum'] - tempJoin['dayofyear']<4)]
    performance_3 = performance_3.groupby(performance_3.index).sum()[['high','low','medium']]
    performance_3['m3total_f'] = performance_3['high']+performance_3['low']+performance_3['medium']
    performance_3['m3perf_f'] = (2*performance_3['high']+performance_3['medium'])*1.0/performance_3['m3total_f']

    
    performance_7 = tempJoin[(tempJoin['dayofyear_toSum'] - tempJoin['dayofyear']<8)]
    performance_7 = performance_7.groupby(performance_7.index).sum()[['high','low','medium']]
    performance_7['m7total_f'] = performance_7['high']+performance_7['low']+performance_7['medium']
    performance_7['m7perf_f'] = (2*performance_7['high']+performance_7['medium'])*1.0/performance_7['m7total_f']
    
    performance_14 = tempJoin[(tempJoin['dayofyear_toSum'] - tempJoin['dayofyear']<14)]
    performance_14 = performance_14.groupby(performance_14.index).sum()[['high','low','medium']]
    performance_14['m14total_f'] = performance_14['high']+performance_14['low']+performance_14['medium']
    performance_14['m14perf_f'] = (2*performance_14['high']+performance_14['medium'])*1.0/performance_14['m14total_f']

    
    performance_30 = tempJoin[(tempJoin['dayofyear_toSum'] - tempJoin['dayofyear']<31)]
    performance_30 = performance_30.groupby(performance_30.index).sum()[['high','low','medium']]
    performance_30['m30total_f'] = performance_30['high']+performance_30['low']+performance_30['medium']
    performance_30['m30perf_f'] = (2*performance_30['high']+performance_30['medium'])*1.0/performance_30['m30total_f']


    update = pd.concat([performance_3[['m3perf_f','m3total_f']],performance_7[['m7perf_f','m7total_f']],\
                        performance_14[['m14perf_f','m14total_f']],performance_30[['m30perf_f','m30total_f']],\
                        ],axis=1)

    if update_df is None: update_df = test_df
    
    new_features = ['m3perf_f','m7perf_f','m14perf_f','m30perf_f',\
                    'm3total_f','m7total_f','m14total_f','m30total_f']
    
    for f in new_features:
        if f not in update_df.columns: 
             update_df[f] = np.nan  
                
    update_df.update(update)
    
    #filling with manager_id_perf
    if filling:
        for f in ['m3perf_f','m7perf_f','m14perf_f','m30perf_f']:
            update_df.loc[np.isnan(update_df[f]),f] = update_df[filling]
    for f in ['m3total_f','m7total_f','m14total_f','m30total_f']:
        update_df.loc[np.isnan(update_df[f]),f] = 0

#another verision for statistics including some leakage
def categorical_statistics(train_df,test_df,cf,nf,\
                           get_median=True,get_min = True,get_max = True,\
                           get_size = False,get_std=False,mini_size = 20):
    statistics ={}
    statistics['mean']='mean'

    if get_max:
        statistics['max']='max'
    if get_min:
        statistics['min']='min'
    if get_median:
        statistics['median']='median'
    if get_std:
        statistics['std']='std'
    if get_size:
        statistics['size']='size'

        
    values = pd.concat([train_df[[cf,nf]],test_df[[cf,nf]]])\
               .groupby(cf)[nf].agg(statistics)
    values = values.add_prefix(cf+'_'+nf+'_')
    
    new_feature = list(values.columns)
    
    #consider using -1 for others
    updateTest = test_df[[cf]].join(values, on = cf, how="left")[new_feature]#.fillna(-1)
    updateTrain = train_df[[cf]].join(values, on = cf, how="left")[new_feature]#.fillna(-1)
        
    for f in new_feature:
        if f not in test_df.columns: 
            test_df[f] = np.nan
        if f not in train_df.columns:
            train_df[f] = np.nan
    #update the statistics excluding the normalized value
    test_df.update(updateTest)
    train_df.update(updateTrain)
    
#without leakage, size for train instead of both 
def categorical_size(train_df,test_df,cf):
    values =train_df.groupby(cf)['interest_level'].agg({'size':'size'})
    values = values.add_prefix(cf+'_')
    new_feature = list(values.columns)
    updateTest = test_df[[cf]].join(values, on = cf, how="left")[new_feature].fillna(-1)
    updateTrain = train_df[[cf]].join(values, on = cf, how="left")[new_feature]#.fillna(-1)
    
    for f in new_feature:
        if f not in test_df.columns: 
            test_df[f] = np.nan
        if f not in train_df.columns:
            train_df[f] = np.nan
    #update the statistics excluding the normalized value
    test_df.update(updateTest)
    train_df.update(updateTrain)

def rank_on_categorical(train_df,test_df,cf,nf,mini_size=20,random=None):
    base = train_df.groupby(cf)[nf].agg({'rank':'rank','size':'size'})
    base['nrank'] = base['rank']/base['size']
    
    if mini_size:
        base.ix[base['size']<mini_size,:] = -1
    
    updateTrain = train_df[[cf]].join(base, on = cf, how="left").fillna(-1)
    updateTest = test_df[[cf]].join(base,on=cf,how = 'left').fillna(-1)

    n_feature = cf+'_'+nf+'_nrank'
    r_feature = cf+'_'+nf+'_rank'
    
    train_df[n_feature] =  updateTrain['rank']
    train_df[r_feature] =  updateTrain['nrank']
    
    test_df[n_feature] =  updateTest['rank']
    test_df[r_feature] =  updateTest['nrank']

#another version for man lat lon including some data leakage
def manager_lon_lat(train_df,test_df):
    
    #adding the features about distance and location
    temp=pd.concat([train_df[['manager_id',"latitude", "longitude"]],\
                     test_df[['manager_id',"latitude", "longitude"]]]).dropna()
    mean_value = temp.groupby('manager_id')[["latitude", "longitude"]].mean().round(4)
    mean_value.columns = ['mlat','mlon']
    std_value = train_df.groupby('manager_id')[["latitude", "longitude"]].std()
    mstd = std_value[["latitude", "longitude"]].mean()
    std_value['latitude']=std_value['latitude'].fillna(mstd['latitude'])
    std_value['longitude']=std_value['longitude'].fillna(mstd['longitude'])
    #manager mean distance
    std_value['m_m_distance'] = map(lambda x,y:np.sqrt(x**2+y**2).round(4),\
                                    std_value['latitude'],std_value['longitude'])
                                    
    
    updateMTest = test_df[['manager_id']].join(mean_value, on = 'manager_id', how="left")[['mlat','mlon']].fillna(-1)
    updateDTest = test_df[['manager_id']].join(std_value, on='manager_id', how="left")['m_m_distance'].fillna(-1)
    updateMTrain = train_df[['manager_id']].join(mean_value, on = 'manager_id', how="left")[['mlat','mlon']].fillna(-1)
    updateDTrain = train_df[['manager_id']].join(std_value, on='manager_id', how="left")['m_m_distance'].fillna(-1)
    
    for f in ['mlat','mlon','m_m_distance']:
        if f not in test_df.columns: 
            test_df[f] = np.nan
        if f not in train_df.columns: 
            train_df[f] = np.nan
    
    test_df.update(updateDTest)
    test_df.update(updateMTest)
    
    train_df.update(updateDTrain)
    train_df.update(updateMTrain)
    
    train_df['m_c_distance'] = map(lambda x,y,i,j: np.sqrt((x-i)**2+(y-j)**2),\
                train_df['latitude'],train_df['longitude'],\
                train_df['mlat'],train_df['mlon'])
    test_df['m_c_distance'] = map(lambda x,y,i,j: np.sqrt((x-i)**2+(y-j)**2),\
                test_df['latitude'],test_df['longitude'],\
                test_df['mlat'],test_df['mlon'])
def runLGBM(train_X, train_y, test_X, test_y=None, feature_names=None,
           seed_val=0, num_rounds=10000,watch_dict = None,max_bin=50000,
           max_depth=4,num_of_leaves=31,early_stop=64,verbose=True,eta=0.1,
           n_threads = 8,colsample_bytree = 0.7,subsample = 0.7):
    param = {'learning_rate':eta, 
             'max_depth':max_depth,
             'num_leaves':num_of_leaves,
             'application':'multiclass',
             'metric':'multi_logloss',
              'num_class':3, 
               'num_threads':n_threads,
              'feature_fraction':colsample_bytree,
              'bagging_fraction':subsample}
    
    num_rounds = num_rounds

    #plst = list(param.items())
    lgbtrain = lgb.Dataset(train_X, label=train_y,max_bin=max_bin,feature_name=feature_names)

    if test_y is not None:
        lgbtest = lgb.Dataset(test_X, label=test_y,max_bin=max_bin,feature_name=feature_names)
        watchlist = [lgbtrain,lgbtest]
        watchlist_name=['train','test']
        model = lgb.train(param, lgbtrain, num_rounds, watchlist,watchlist_name,
                          early_stopping_rounds=early_stop,\
                         evals_result = watch_dict,verbose_eval=verbose)
    else:
        #lgbtest = lgb.Dataset(test_X,feature_name=feature_names)
        model = lgb.train(param, lgbtrain, num_rounds,[lgbtrain],['train']
        ,verbose_eval = verbose )

    pred_test_y = model.predict(test_X)
    return pred_test_y, model
    
def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, 
     seed_val=0, early_stop = 20,num_rounds=10000, eta = 0.1,gamma=1,
     max_depth = 6,watch_dict = None,verbose=True,
     subsample = 0.7,colsample_bytree =0.7):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = eta
    param['gamma']=gamma
    param['max_depth'] = max_depth
    param['silent'] = 1
    param['num_class'] = 3 
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['subsample'] = subsample
    param['colsample_bytree'] = colsample_bytree
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y,feature_names=feature_names)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y,feature_names=feature_names)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist,\
        early_stopping_rounds=early_stop,evals_result = watch_dict,verbose_eval = verbose)
    else:
        xgtest = xgb.DMatrix(test_X,feature_names=feature_names)
        model = xgb.train(plst, xgtrain, 
                          num_rounds,[(xgtrain,'train')],
                          verbose_eval = verbose)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model            

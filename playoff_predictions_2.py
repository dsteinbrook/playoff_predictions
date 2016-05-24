
# coding: utf-8

# In[1]:

import numpy as np
import sys
import urllib
from urllib import urlopen
import bs4
from bs4 import BeautifulSoup #to install: "pip install BeautifulSoup"
import pandas as pd #to install: http://pandas.pydata.org/getpandas.html
import html5lib #to install: "pip install html5lib"
import scipy
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
import math
#get_ipython().magic(u'matplotlib inline')
import seaborn as sns

from read_data import read_csv
from read_data import read_features
from read_data import get_results
from read_data import get_features


#function to output model predictions to a dictionary labeled by
#the teams and season of the matchup

def output_predictions(path,start,end,values):
    keys=[]
    df=read_csv(path)
    #start,end=test[0],test[1]
    criterion=df['Season'].map(lambda x:x in range(start,end+1))
    df2=df[criterion]
    df2=df2.reset_index(drop=True)
    for index,row in df2.iterrows():
        keys.append((row['Home/Neutral'],row['Visitor/Neutral'],row['Season']))
    output=dict(zip(keys,values))
    return output

params = {'n_estimators': 1000,
              'max_depth':2,
              'min_samples_split':4,
              'learning_rate':.002,
              'subsample':0.5,}

path="playoff_results_1980_2015.csv"

#runs a gradient boosting classifier. Inputs are the training period, 
#test period, and paramaters for the model

def gradient_boost(train,test,params):
   
    
    x_train=get_features(path,train[0],train[1])
    x_test=get_features(path,test[0],test[1])
    y_train=get_results(path,train[0],train[1])
    y_test=get_results(path,test[0],test[1])
   
    clf=ensemble.GradientBoostingClassifier(**params)
    clf.fit(x_train,y_train)
    model=clf.fit(x_train,y_train)
    
    predictions=clf.predict_proba(x_test)
    predictions2=predictions[:,1:]
    values=predictions2.tolist()
    output=output_predictions(path,test[0],test[1],values)
    
    test_deviance = np.zeros((params['n_estimators'],), dtype=np.float64)
    for i, y_pred in enumerate(clf.staged_predict_proba(x_test)):
       
        y_pred2=y_pred[:,1]

        test_deviance[i] = -1.0*np.sum(y_test*map(math.log,y_pred2)+(1.0-y_test)
                                       *map(math.log,(1.0-y_pred2)))/y_test.shape[0]
    
    minimum=(test_deviance.argmin(),test_deviance.min())
    #print minimum
    return [output,model,test_deviance]

test_deviance=gradient_boost([1980,2010],[2011,2015],params)[2]
model=gradient_boost([1980,2010],[2011,2015],params)[1]
features=read_features("regular_season_stats_1980_2016.csv",['SRS'])

#outputs the model predictions for user input home
#and away teams for the 2016 playoffs

def predictions_2016(features,model,teams):
    home, away = teams['home'], teams['away']
    x_test=np.zeros((1,2))
    x_test[0][0]=features[2016,home][0]-features[2016,away][0]
    if x_test[0][0]>=0:
        x_test[0][1]=1.0
    else:
        x_test[0][1]=0.0
   
    predictions=model.predict_proba(x_test)
    output=home+':'+str(predictions[0][1])
    return output


teams = {'home': raw_input("Home team (full name, e.g. Toronto Raptors): "),
         'away': raw_input("Away team: ")}

a=predictions_2016(features,model,teams)
print a

#runs a grid search to tune the parameters for the gradient boosting classifier
#using cross validation

def grid_search(x,y,i):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.5,random_state=i)
    tuned_parameters = [{'n_estimators':[1000],'max_depth':[2], 
                         'min_samples_split':[2],
                        'learning_rate':[.003],
                         'subsample':[1.0]}]
    clf = GridSearchCV(ensemble.GradientBoostingClassifier(),tuned_parameters,cv=5,
                      scoring='log_loss')
    clf.fit(x_train,y_train)
    best_params=clf.best_params_
    for params, mean_score, scores in clf.grid_scores_:
        print mean_score,scores.std(),params
    return best_params
#x=get_features(path,1980,2015)
#y=get_results(path,1980,2015)


    
#test=grid_search(x,y,0)
#test

#tests the sensitivity of the model to choice of training set by
#randomly splitting the data into train and test samples

def train_test(x,y):
    scores=np.zeros(10)
    for i in range(0,10):
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.5,random_state=i)
        clf=ensemble.GradientBoostingClassifier(**params)
        clf.fit(x_train,y_train)
        score=clf.score(x_test,y_test)
        scores[i]=score
    return scores.mean(),scores.std()

#test=train_test(x,y)
#test
        


#plots the log-loss deviance on the test set as a function of boosting iterations

#plt.figure()
#plt.plot((np.arange(test_deviance.shape[0]) + 1)[::5], test_deviance[::5],
#             '-', color='orange')
#plt.xlabel('Boosting iterations')
#plt.ylabel('Deviance (log loss)')
    



    

        

    
    
    




# In[ ]:




# In[ ]:




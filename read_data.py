import numpy as np
import sys
import urllib
from urllib import urlopen
import bs4
from bs4 import BeautifulSoup #to install: "pip install BeautifulSoup"
import pandas as pd #to install: http://pandas.pydata.org/getpandas.html
import html5lib #to install: "pip install html5lib"

#reads csv into dataframe
def read_csv(filename):
    df=pd.read_csv(filename,index_col=0)
    return df


#for given data file "path" and desired stats "params" creates a 
#dictionary of stats for each team and season
def read_features(path, params):
    d=dict()
    df=read_csv(path)
    for index, row in df.iterrows():
        d[(row['Season'],row['Team'])] = [row[p] for p in params]
    return d
 
        

#creates the target numpy array y for the gradient boosting model.
#target is 1 if the home team wins; 0 otherwise


def get_results(path,start,end):
    df=read_csv(path)
    criterion=df['Season'].map(lambda x:x in range(start,end+1))
    df2=df[criterion]
    df2=df2.reset_index(drop=True)
    length=df2.shape[0]
    y=np.zeros(length)
    for index, row in df2.iterrows():
        if row['HPTS'] - row['PTS'] > 0:
            y[index] = 1
        else:
            y[index] = 0
    return y

#creates the features numpy array x for the gradient boosting model.

features=read_features("regular_season_stats_1980_2016.csv",['SRS'])

def get_features(path,start,end):
    df=read_csv(path)
    criterion=df['Season'].map(lambda x:x in range(start,end+1))
    df2=df[criterion]
    df2=df2.reset_index(drop=True)
    length=df2.shape[0]
    x=np.zeros((length,2))
    for i,game in df2.iterrows():
        # Feature 1: difference in SRS
        x[i][0] = features[game['Season'],game['Home/Neutral']][0] - features[game['Season'],game['Visitor/Neutral']][0]
        # Feature 2: home field advantage
        if x[i][0] >= 0:
            x[i][1] = 1
        else:
            x[i][1] = 0
    return x
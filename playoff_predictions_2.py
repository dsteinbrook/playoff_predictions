
import numpy as np
import sys
import urllib
from urllib import urlopen
import bs4
from bs4 import BeautifulSoup #to install: "pip install BeautifulSoup"
import pandas as pd #to install: http://pandas.pydata.org/getpandas.html
import html5lib #to install: "pip install html5lib"




#scrapes playoff game results from basketball reference.com
#for a given range of seasons, outputs to csv file

def download_playoff_results(start,end):

    url_template="http://www.basketball-reference.com/leagues/NBA_{year}_games.html"
    playoff_df=pd.DataFrame()
    for year in range(start,end+1):
        url=url_template.format(year=year)
        html=urlopen(url)
        soup=BeautifulSoup(html)
        
        #gets column headers
    
        col_headers_raw = soup.findAll('tr', limit=2)[0].findAll('th')
        col_headers = [th.getText() for th in col_headers_raw]
        col_headers[6]='HPTS'

        #loads data into dataframe
        table = soup.findAll('table',id='games_playoffs')[0]
        data_rows=table.findAll('tr')[1:]
        playoff_data = [[td.getText() for td in data_rows[i].findAll('td')]
            for i in range(len(data_rows))]
        year_df = pd.DataFrame(playoff_data, columns=col_headers)
  


        year_df = year_df[year_df.Date.notnull()]

       

        year_df= year_df[:].fillna(0)
        year_df = pd.DataFrame(year_df, columns=col_headers)


        
        year_df.insert(0,'Season',year)

 
        playoff_df=playoff_df.append(year_df, ignore_index=True)
        print year

    filename="playoff_results_{start}_{end}.csv"
    filename=filename.format(start=start,end=end)
    playoff_df.to_csv(filename)
    
#test=download_playoff_results(1980,2015)
    



#scrapes team regular season summary statistics
#from basketball reference.com for given range of seasons, outputs to csv
def download_regular_season_stats(start,end):

    url_template="http://www.basketball-reference.com/leagues/NBA_{year}.html"
    misc_df=pd.DataFrame()
    for year in range(start,end+1):
        url=url_template.format(year=year)
        html=urlopen(url)
        soup=BeautifulSoup(html)
        
        #gets column headers
        table=soup.findAll('tr',class_="over_header")[0]
        col_headers_raw = table.find_next_siblings('tr')[0].findAll('th')


        #loads data into dataframe
        misc = soup.findAll(class_='stw',id='all_misc_stats')[0]
        data_rows=misc.findAll('tr')[1:]
        year_data = [[td.getText() for td in data_rows[i].findAll('td')]
            for i in range(len(data_rows))]
            

        col_headers = [th.getText() for th in col_headers_raw]
        year_df = pd.DataFrame(year_data, columns=col_headers)
        year_df.insert(0,'Season',year)
    
        originals=year_df['Team']
        teams=year_df['Team'].tolist()
        
        #removes asterisks from team names
    
        for i,team in enumerate(teams):
            team=str(team)
    
   
   
            if team.endswith("*")==True:
                team=team[:-1]
                teams[i]=unicode(team)
            else:
                pass

    

        vals_to_replace=dict(zip(originals,teams))

        year_df['Team']=year_df['Team'].map(vals_to_replace)
    
    
    
        misc_df=misc_df.append(year_df,ignore_index=True)
        print year
    misc_df.head()
    
    filename="regular_season_stats_{start}_{end}.csv"
    filename=filename.format(start=start,end=end)
    misc_df.to_csv(filename)
    

#test=download_regular_season_stats(1980,2016)



#reads csv into dataframe
def read_csv(filename):
    df=pd.read_csv(filename,index_col=0)
    return df

#for given data file "path" and desired stats "params" creates a 
#dictionary of params for each team and season

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
        x[i][0] = features[game['Season'],game['Home/Neutral']][0] -         features[game['Season'],game['Visitor/Neutral']][0]
        # Feature 2: home field advantage
        if x[i][0] >= 0:
            x[i][1] = 1
        else:
            x[i][1] = 0
    return x
    


#get_features("playoff_results_1980_2015.csv",2013, 2015)


import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn import datasets
from sklearn import cross_validation
import math
import seaborn as sns


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
              'subsample':0.5}

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

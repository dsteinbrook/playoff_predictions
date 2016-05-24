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
    


# In[ ]:




# In[3]:

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
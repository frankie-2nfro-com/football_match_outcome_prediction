# AiCore Project #3 - Football Match Outcome Prediction

## Milestone 1 - EDA and Data Cleaning
In this project, I start with three data source files namely Result csv for several football leagues, Matches information csv and Team information csv. And in this milestone, I will try to experience the EDA and clean up procedures.

### Data source files
#### 1) Football.zip
The file includes all result information for several leagues. After unzipping the file, it contains different league folders with different season results in different csv. The file naming convention is Results_{Season}_{League}.csv. For example, for 2021 result of Premier League is in /Results/premier_league/Results_2021_premier_league.csv. In the csv file, it contains fields as follows: 
[Home_Team], [Away_Team], [Result], [Link], [Season], [Round] and [League]

##### Combine the data into dataframe
To better explore the data, I will combine all result csv files into a single pandas dataframe as follows:
```python
# load all directory as league name list
dir = "./Results"
leagues = [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]

# loop to open csv
result_pd = pd.DataFrame()
for league in leagues:
    league_folder = os.path.join(dir, league)
    csv_file_for_league = [os.path.join(league_folder, name) for name in os.listdir(league_folder) if name.endswith('.csv')]
    league_pds = [pd.read_csv(csv_filename, skiprows=[0], names=["Home_Team", "Away_Team", "Result", "Link", "Season", "Round", "League"]) for csv_filename in csv_file_for_league]
    whole_list_df = pd.concat(league_pds)
    result_pd = pd.concat([result_pd, whole_list_df])
```

#### 2) Match_Info.csv
In this csv file, it contains fields as follows: 
[Link], [Date_New], [Referee], [Home_Yellow], [Home_Red], [Away_Yellow], and [Away_Red]

Because of a single csv file, I just read the data to a single dataframe as follows:
```python
match_pd = pd.read_csv("./Matches/Match_Info.csv")
```

#### 3) Team_Info.csv
In this csv file, it contains fields as follows: 
[Team], [City], [Country], [Stadium], [Capacity], [Pitch]

Because of a single csv file, I just read the data to a single dataframe as follows:
```python
team_pd = pd.read_csv("./Matches/Match_Info.csv")
```

So, I have three dataframes namely result_pd, match_pd and team_pd.

### Peek at the data
In the very begining, there is no substitute for looking at the raw data. Here is the first 20 records of those dataframes.

Result first 20 records:
![Result first 20 records](https://github.com/frankie-2nfro-com/football_match_outcome_prediction/blob/main/Screens/Results_records_20.png)

Match first 20 records:
![Match first 20 records](https://github.com/frankie-2nfro-com/football_match_outcome_prediction/blob/main/Screens/Match_records_20.png)

Team first 20 records:
![Team first 20 records](https://github.com/frankie-2nfro-com/football_match_outcome_prediction/blob/main/Screens/Team_records_20.png)

Structure and summary of data:

result_pd.info()
```
<class 'pandas.core.frame.DataFrame'>
Int64Index: 146641 entries, 0 to 379
Data columns (total 7 columns):
 #   Column     Non-Null Count   Dtype 
---  ------     --------------   ----- 
 0   Home_Team  146641 non-null  object
 1   Away_Team  146641 non-null  object
 2   Result     146641 non-null  object
 3   Link       146641 non-null  object
 4   Season     146641 non-null  object
 5   Round      146641 non-null  object
 6   League     146641 non-null  object
dtypes: object(7)
memory usage: 9.0+ MB
```

match_pd.info()
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 143348 entries, 0 to 143347
Data columns (total 7 columns):
 #   Column       Non-Null Count   Dtype  
---  ------       --------------   -----  
 0   Link         143348 non-null  object 
 1   Date_New     143348 non-null  object 
 2   Referee      143348 non-null  object 
 3   Home_Yellow  122798 non-null  float64
 4   Home_Red     122798 non-null  float64
 5   Away_Yellow  122798 non-null  float64
 6   Away_Red     122798 non-null  float64
dtypes: float64(4), object(3)
memory usage: 7.7+ MB
```

team_pd.info()
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 544 entries, 0 to 543
Data columns (total 6 columns):
 #   Column    Non-Null Count  Dtype 
---  ------    --------------  ----- 
 0   Team      544 non-null    object
 1   City      544 non-null    object
 2   Country   544 non-null    object
 3   Stadium   447 non-null    object
 4   Capacity  544 non-null    object
 5   Pitch     447 non-null    object
dtypes: object(6)
memory usage: 25.6+ KB
```

To find out some relationships amongst the data, I create some functions to filter result_pd:
```python
def getLeagueDataFrame(data, league):
    return data[data["League"]==league]

def getLeagueSeasonDataFrame(data, league, season):
    return data[(data["League"]==league) & (data["Season"]==season)]
     
def getLeagueSeasonTeamDataFrame(data, league, season, team):
    return data[(data["League"]==league) & ((data["Home_Team"]==team) | (data["Away_Team"]==team)) & (data["Season"]==season)]

def getLeagueTeamCount(data, league, season):
    league_season_df = getLeagueSeasonDataFrame(data, league, season)
    return len(league_season_df["Home_Team"].sort_values().unique())
    
...
```

Some high level relationships amongst the variables, e.g. for each league, how many teams in a particular season:
```python
leagues = result_pd["League"].sort_values().unique()
for league in leagues:
    league_pd = getLeagueDataFrame(result_pd, league)
    seasons = league_pd["Season"].sort_values().unique()
    for season in seasons:
        print(league, season, getLeagueTeamCount(result_pd, league, season))
```

For each year, does the trend of the number of home wins change:
![Home Win Trend for Premier League](https://github.com/frankie-2nfro-com/football_match_outcome_prediction/blob/main/Screens/home_win_trend_sample.png)

### Problem found at the data
After scanning the data, some obvious problems could be located like encoding problem, format problem and NaN. It may need to solve if the data is useful. Another big problem I found is that "Result" field in the result_pd actually is not suitable to be stored like that. It is in a string format and the data itself is aggregated with some important data. So I will break down this field into some new fields i.e. Home_Score, Away_Score, Home_Win, Away_Win and Draw.

### Prelimilary direction

### Data cleaning and refine

### Hypothesis Testing




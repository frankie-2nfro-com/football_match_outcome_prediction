# AiCore Project #3 - Football Match Outcome Prediction

## Milestone 1 - Setup the environment
In this project, we'll use GitHub to track changes to our code and save them online in a GitHub repo. 


<br />    
<br />  





## Milestone 2 - EDA and Data Cleaning
In this project, I start with three data source files namely Result csv for several football leagues, Matches information csv and Team information csv. And in this milestone, I will try to experience the EDA and clean up procedures. Although the data in this stage is not rich enough, it could be a very good starting for learning the process of understanding data in order to get a best results. I will try to understand the data with some descriptive statistics.

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
Actually, the data I have is quite simple and limited. Before enhancing the source information, what features I consider to be more important for predicting the outcome of the match? In my view, I think the recent performance of a team is one of the key to predict the team's outcome. The question is how to get recent performance? In the data, I could decompose result field into Home_Score and Away_Score, so I can get the goal difference. If I get latest n matches goal difference as the team recent performance, I can use this to predict which team have better performance. Apart from this, Home_Yellow, Home_Red, Away_Yellow and Away_Red in match_pd may be kind of implication of the team performance. 

### Data cleaning and refine
Firstly, I will try to decompose result field into Home_Score, Away_Score, Home_Win, Away_Win and Draw. Definition is as follows:
Home_Score (Integer): Home team goals
Away_Score (Integer): Away team goals
Home_Win (Boolean): Is home team win
Away_Win (Boolean): Is away team win
Draw (Boolean): Is draw

The format of Result is (Home team goal)-(Away team goal). So I will try to extract home team goal and away team goal by regular expression. Code is as follows:
```python
# Divide result into home_score and away_score
df_score =  result_pd['Result'].str.extract(r'(\d)-(\d)')
result_pd.insert(loc=3, column="Home_Score", value=df_score[0].astype('Int64'))     # use Int64 as it support NaN
result_pd.insert(loc=4, column="Away_Score", value=df_score[1].astype('Int64')) 

# calculate score difference
df_h_a = result_pd["Home_Score"] - result_pd["Away_Score"]
df_h_a.isna().sum()

# add home_win, away_win and draw column
hw_array = []
aw_array = []
d_array = []
for minus_score in df_h_a:
    if pd.isna(minus_score):
        hw_array.append(pd.NA)
        aw_array.append(pd.NA)
        d_array.append(pd.NA)
    else:
        if minus_score>0:
            hw_array.append(True)
            aw_array.append(False)
            d_array.append(False)
        elif minus_score<0:
            hw_array.append(False)
            aw_array.append(True)
            d_array.append(False)
        else:
            hw_array.append(False)
            aw_array.append(False)
            d_array.append(True)

result_pd.insert(loc=5, column="Home_Win", value=hw_array) 
result_pd.insert(loc=6, column="Away_Win", value=aw_array) 
result_pd.insert(loc=7, column="Draw", value=d_array)
```

After running, the result_df becomes:
![Decompose result_df](https://github.com/frankie-2nfro-com/football_match_outcome_prediction/blob/main/Screens/Results_decompose.png)

To find the NaN or Null value in the new result_df:
```python 
result_null_pd = result_pd[result_pd.isna().any(axis=1)]
```
![null value](https://github.com/frankie-2nfro-com/football_match_outcome_prediction/blob/main/Screens/Results_decompose_null.png)
There is 96 records with NaN. I need to find a way to filling the missing data. Best way is getting correct information from other data source. If not possible, could i just remove the record with missing data? Or I need to fill data by some imputing approaches such as by average, mean, median, KNNImputer, Time Series Imputation or else.

Let's have some brief information about the new numeric fields:
```python
result_pd.describe()
```
![describe](https://github.com/frankie-2nfro-com/football_match_outcome_prediction/blob/main/Screens/score_describe.png)

To visualize the distribution:
```python
from matplotlib import pyplot
result_pd.hist()
pyplot.show()
```
![hist chart](https://github.com/frankie-2nfro-com/football_match_outcome_prediction/blob/main/Screens/result_pd_hist_chart.png)

### Hypothesis Testing
Like software engineering, we need to perform testing to make sure the software performs as expected. In data field, we also need testings to make sure the prepared data can produce finding as our assumption. This can be done by Hypothesis Testing. After EDA, a data analysis report with data assumption will be produced. And there are some hypothesis testings to prove the assumptions. Here is an example:

Problem: Is there any way to use data on hand to imporve the guessing of home team win? 

Assumption: I guess the data to represent the team recent performance should help to improve the prediction (Definition of recent performance is defined as goal difference for last 6 games)

#### Step 1 - Define null and alternative hypothesis

H0: There is no difference in win rate in home team with better or equal recent performance

H1: There is increase in win rate in home team with better or equal recent performance

Significance level: 5% (alpha level)

We have to collect enough evidence through our tests to reject the null hypothesis H0.

#### Step 2 - Examine data, check assumptions
In this step, the result data was filtered to records in different league and season. And then I need to calculate the recent performance and add to the dataframe. 

```python
def findRecentPreviousRounds(currentRound, limit):
    if currentRound<=limit:
        return None
    else:
        r = []
        for l in range(limit):
            r.append(currentRound - (limit-l))
        return r


def findLeagueSeasonTeamRecentPreviousRounds(data, league, season, team, round):
    rounds = findRecentPreviousRounds(round, 6)     # by definition is 6, can change for optimization
    if rounds is None:
        return None

    previous_matches_pd =  data[(data["League"]==league) & ((data["Home_Team"]==team) | (data["Away_Team"]==team)) & (data["Season"]==season) & (data["Round"].isin(rounds))]
    recent_perf = 0
    for index, row in previous_matches_pd.iterrows():
        hteam = row['Home_Team']
        ateam = row['Away_Team']
        if hteam==team:
            recent_perf = recent_perf + (row['Home_Score']-row['Away_Score'])
            #print('HOME', row['Home_Score']-row['Away_Score'])
        else:
            recent_perf = recent_perf + (row['Away_Score']-row['Home_Score'])
            #print('AWAY', row['Away_Score']-row['Home_Score'])

    return recent_perf


def addRecentPerfToLeagueSeason(data, league, season):
    league_season_pd = getLeagueSeasonDataFrame(data, league, season)

    hperf_list = []
    aperf_list = []
    for index, row in league_season_pd.iterrows():
        hteam = row['Home_Team']
        ateam = row['Away_Team']
        round = row["Round"]

        htperf = findLeagueSeasonTeamRecentPreviousRounds(league_season_pd, league, season, hteam, round)
        atperf = findLeagueSeasonTeamRecentPreviousRounds(league_season_pd, league, season, ateam, round)
        hperf_list.append(htperf)
        aperf_list.append(atperf)

    league_season_pd.insert(loc=2, column="Home_Recent", value=hperf_list) 
    league_season_pd.insert(loc=3, column="Away_Recent", value=aperf_list) 
    return league_season_pd
```

So I can get different league and season data by function addRecentPerfToLeagueSeason(). Here is the data I generated by random:

```python
result_premier_league_2021_pd = addRecentPerfToLeagueSeason(result_pd, 'premier_league', 2021)
result_premier_league_2020_pd = addRecentPerfToLeagueSeason(result_pd, 'premier_league', 2020)
result_premier_league_2019_pd = addRecentPerfToLeagueSeason(result_pd, 'premier_league', 2019)
result_premier_league_2018_pd = addRecentPerfToLeagueSeason(result_pd, 'premier_league', 2018)
result_premier_league_2017_pd = addRecentPerfToLeagueSeason(result_pd, 'premier_league', 2017)
```

Here is the sample of result_premier_league_2021_pd:

![Recent performance data sample](https://github.com/frankie-2nfro-com/football_match_outcome_prediction/blob/main/Screens/Recent_perf_sample.png)

To calculate the final win rate, code as follows:

```python
def getLeagueSeasonHomeTeamWinRate(data):
    HW = 0
    HWC = 0
    HWTOTAL = 0
    C = 0
    for index, row in data.iterrows():
        hperf = row["Home_Recent"]
        aperf = row["Away_Recent"]
        hwin = row["Home_Win"]

        # skip NaN
        if pd.isna(hperf) or pd.isna(aperf) or pd.isna(hwin):
            continue

        C = C + 1
        if hperf>=aperf:
            HWC = HWC + 1
            if hwin:
                # TP
                HW = HW + 1

        if hwin:
            HWTOTAL = HWTOTAL + 1

    return (HW/HWC * 100), (HWTOTAL/C * 100)  
```

To see the win rate for the H0 situation and general home win rate comparison:

```python
print(getLeagueSeasonHomeTeamWinRate(result_premier_league_2021_pd))
print(getLeagueSeasonHomeTeamWinRate(result_premier_league_2020_pd))
print(getLeagueSeasonHomeTeamWinRate(result_premier_league_2019_pd))
print(getLeagueSeasonHomeTeamWinRate(result_premier_league_2018_pd))
print(getLeagueSeasonHomeTeamWinRate(result_premier_league_2017_pd))
```

Output:

```python
(49.6, 38.429752066115704)
(55.49132947976878, 45.9375)
(59.45945945945946, 48.75)
(53.48837209302325, 46.25)
(58.82352941176471, 50.0)
```

Just see the sample result, it actually improve the win rate. But can it fulfill the hypothesis test? It needs to prove after calculate the testing statistic in next step. Before that, I need to prepare the dataframe by follow code:

```python
win_rate_table = []
leagues = result_pd["League"].sort_values().unique()
for league in leagues:
    league_pd = getLeagueDataFrame(result_pd, league)
    seasons = league_pd["Season"].sort_values().unique()
    for season in seasons:
        result_with_perf_pd = addRecentPerfToLeagueSeason(result_pd, league, season)
        perf_home_win_rate, general_home_win_rate = getLeagueSeasonHomeTeamWinRate(result_with_perf_pd)
        increase = perf_home_win_rate - general_home_win_rate
        win_rate_table.append([league, season, perf_home_win_rate, general_home_win_rate, increase])
        
win_rate_pd = pd.DataFrame(win_rate_table, columns=["League", "Season", "Pref_Home_Win_Rate", "General_Home_Win_Rate", "Win_Rate_Increase"])
```

![Win rate increase sample](https://github.com/frankie-2nfro-com/football_match_outcome_prediction/blob/main/Screens/win_rate_sample_data.png)

To find out some statistic:

```python
win_rate_pd["General_Home_Win_Rate"].describe()

count    404.000000
mean      46.543962
std        3.627231
min       34.615385
25%       44.270833
50%       46.428571
75%       48.989899
max       61.419753
Name: General_Home_Win_Rate, dtype: float64
```
So statistic of home team winning rate for different league and season could be found in the above output. The mean is 46.543962. The std is 3.627231.

Now we need to extract the sample records to prove our test. I randomly filter records by:

```python
test_sample_win_rate_pd = win_rate_pd.sample(frac=0.5)
```
To find out the statistic:

```python
test_sample_win_rate_pd["Pref_Home_Win_Rate"].describe()

count    202.000000
mean      52.569771
std        5.574470
min       37.391304
25%       48.418576
50%       52.563087
75%       56.044560
max       70.072993
Name: Pref_Home_Win_Rate, dtype: float64
```

#### Step 3 - Calculate Test Statistic
First of all, I try to find out Z value. 

Z = (Mean of home win rate with better recent performance) - (Mean of normal home win rate) / (standard deviation of normal home win rate)

Z = (52.569771 - 46.543962) / 3.627231

Z = 1.66

Base on the table, z value will be 0.9515. 

#### Step 4 - Determine the corresponding p-value

P = 1 - z value

P = 1 − 0.9515 

P = 0.0485

So, we have a 4.85% chance of finding home team winning rate is not increase when recent performance is better or equal the away team. 

#### Step 5 - make a decision about the null hypothesis

In hypothesis testing, we need to judge whether it is a one-tailed or a two-tailed test. Generally speaking, it depends on the original claim
in the question. A single tailed test looks for "increase" or "decrease" in the hypothesis. (Two-tailed test usually look for "change" which could be increase or decrease) 

So in this case, only 4.85% records of win rate of home team with better or equal recent performance decreased. So it is smaller than the significance level. And H0 will be rejected. 

  
<br />    
<br />    
  





## Milestone 3 - Feature Engineering
Apart from the new features I created in milestone 1, I have few more new features created in this milestone namely ELO_HOME, ELO_AWAY, HOME_TOTAL_GOAL_SO_FAR and AWAY_TOTAL_GOAL_SO_FAR. As the heavy nested looping, it is not able to create a big list before creating the new features. So I will calculate the total goal feature league by league. Also, I will try to make use pandas internal loop for better performance. So the apply() function for dataframes will be called to handle each row of the league data. 

To pipeline the whole process, I created a notebook file as pipeline.ipynb. So that I can be easier to check result step by step. When the process has been proved well running, I can put all logic to a python file and run as a whole. 

### ELO

For adding ELO features, I need to load the pickle database first: 
```python 
d = pickle.load(open('./ELO/elo_dict.pkl', 'rb'))
```

To map the field 'Link' in the pickle database, I create this function:
```python
def fillWithELO(link):
    if link not in d:
        return [pd.NA, pd.NA]
    else:
        return [d[link]['Elo_home'], d[link]['Elo_away']]
```

So for adding the ELO_HOME and ELO_AWAY features:
```python
# merge with ELO
result_elo_pd = current_league_season_pd['Link'].apply(fillWithELO)   
elo_list = np.array(result_elo_pd.values.tolist())
elo_df = pd.DataFrame(elo_list, columns=["ELO_HOME", "ELO_AWAY"])
current_league_season_pd.insert(loc=8, column="ELO_HOME", value=elo_df["ELO_HOME"]) 
current_league_season_pd.insert(loc=9, column="ELO_AWAY", value=elo_df["ELO_AWAY"]) 
```

For each records in the dataframe, it will call fillWithELO(link) function with the link value. And the function will get Elo_home and Elo_away from pickle database. The list of Elo_home, Elo_away will be returns. After that, I will insert the features to the dataframe. 

### Total Goals So Far

For adding total goals so far for both home and away team, I create the following functions:
```python
def getLeagueSeasonTeamBeforeRoundTotalGoal(data, league, season, team, round):
    # determine home or away and get the score 
    # get home game of the team
    home_pd = data[(data["League"]==league) & (data["Home_Team"]==team) & (data["Season"]==season) & (data["Round"]<round)]
    df_home_score_sofar =  home_pd['Result'].str.extract(r'(\d)-\d')
    home_total_score = df_home_score_sofar[0].astype('Int64').sum()

    # get away game of the team
    away_pd = data[(data["League"]==league) & (data["Away_Team"]==team) & (data["Season"]==season) & (data["Round"]<round)]
    df_away_score_sofar =  away_pd['Result'].str.extract(r'\d-(\d)')
    away_total_score = df_away_score_sofar[0].astype('Int64').sum()

    # calculate total goals
    return (home_total_score + away_total_score)


def fillWithTotalGoalSoFar(record, data):
    # get home team and away team and round
    league = record['League']
    season = record['Season']
    round = record['Round']
    hteam = record['Home_Team']
    ateam = record['Away_Team']
    
    home_goal_so_far = getLeagueSeasonTeamBeforeRoundTotalGoal(data, league, season, hteam, round)
    away_goal_so_far = getLeagueSeasonTeamBeforeRoundTotalGoal(data, league, season, ateam, round)

    return [home_goal_so_far, away_goal_so_far]
```

Also, I will call apply() for the dataframe. So each record will be looped to calculate the total goals:

```python
# get home team and away team total goal so far
home_away_total_goal_sofar = current_league_season_pd.apply(fillWithTotalGoalSoFar, data=current_league_season_pd, axis=1)
goal_so_far_list = np.array(home_away_total_goal_sofar.values.tolist())         # convert to list
home_away_total_goal_sofar_pd = pd.DataFrame(goal_so_far_list, columns=["HOME_GOAL_SO_FAR", "AWAY_GOAL_SO_FAR"])    # convert to dataframe
current_league_season_pd.insert(loc=5, column="HOME_TOTAL_GOAL_SO_FAR", value=home_away_total_goal_sofar_pd["HOME_GOAL_SO_FAR"]) 
current_league_season_pd.insert(loc=6, column="AWAY_TOTAL_GOAL_SO_FAR", value=home_away_total_goal_sofar_pd["AWAY_GOAL_SO_FAR"])   
```

### Recent performance

For adding recent goal difference (last 6 games) for both home and away team, I create the following functions:

```python
def findRecentPreviousRounds(currentRound, limit):
    if currentRound<=limit:
        return None
    else:
        r = []
        for l in range(limit):
            r.append(currentRound - (limit-l))
        return r


def findLeagueSeasonTeamRecentPreviousRounds(data, league, season, team, round):
    rounds = findRecentPreviousRounds(round, 6)         # by definition is 6, can change for optimization
    if rounds is None:
        return None

    previous_matches_pd =  data[(data["League"]==league) & ((data["Home_Team"]==team) | (data["Away_Team"]==team)) & (data["Season"]==season) & (data["Round"].isin(rounds))]
    recent_perf = 0
    for index, row in previous_matches_pd.iterrows():
        hteam = row['Home_Team']
        ateam = row['Away_Team']
        if hteam==team:
            recent_perf = recent_perf + (row['Home_Score']-row['Away_Score'])
        else:
            recent_perf = recent_perf + (row['Away_Score']-row['Home_Score'])

    return recent_perf


def fillWithRecentPerformance(record, data):
    # get home team and away team and round
    league = record['League']
    season = record['Season']
    round = record['Round']
    hteam = record['Home_Team']
    ateam = record['Away_Team']
    
    home_team_goal_diff = findLeagueSeasonTeamRecentPreviousRounds(data, league, season, hteam, round)
    away_team_goal_diff = findLeagueSeasonTeamRecentPreviousRounds(data, league, season, ateam, round)

    return [home_team_goal_diff, away_team_goal_diff]
```

Similarly, I make use apply() to loop all the record in dataframe:

```python
# get recent performance
home_away_recent_perf = current_league_season_pd.apply(fillWithRecentPerformance, data=current_league_season_pd, axis=1)
perf_list = np.array(home_away_recent_perf.values.tolist())
home_away_perf_pd = pd.DataFrame(perf_list, columns=["HOME_LAST_6_GOAL_DIFF", "AWAY_LAST_6_GOAL_DIFF"])
current_league_season_pd.insert(loc=7, column="HOME_LAST_6_GOAL_DIFF", value=home_away_perf_pd["HOME_LAST_6_GOAL_DIFF"]) 
current_league_season_pd.insert(loc=8, column="AWAY_LAST_6_GOAL_DIFF", value=home_away_perf_pd["AWAY_LAST_6_GOAL_DIFF"]) 
```

After creating the new dataframe with new features, the data is as follows:

![New features data](https://raw.githubusercontent.com/frankie-2nfro-com/football_match_outcome_prediction/main/Screens/Milestone2_new_features.png)

The whole logic to handle all csv files is as follows:

```python
# load all directory as league name list
dir = "./Results"
leagues = [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]

# loop to open csv
result_with_goal_sofar_pd = pd.DataFrame()
for league in leagues:
    print("process league: " + league + "...")
    league_folder = os.path.join(dir, league)
    csv_file_for_league = [os.path.join(league_folder, name) for name in os.listdir(league_folder) if name.endswith('.csv')]
    
    for csv_filename in csv_file_for_league:
        current_league_season_pd = pd.read_csv(csv_filename, skiprows=[0], names=["Home_Team", "Away_Team", "Result", "Link", "Season", "Round", "League"])

        # Divide result into home_score and away_score
        df_score =  current_league_season_pd['Result'].str.extract(r'(\d)-(\d)')
        current_league_season_pd.insert(loc=3, column="Home_Score", value=df_score[0].astype('Int64'))     # use Int64 as it support NaN
        current_league_season_pd.insert(loc=4, column="Away_Score", value=df_score[1].astype('Int64')) 

        if len(current_league_season_pd)>0:
            # get home team and away team total goal so far
            home_away_total_goal_sofar = current_league_season_pd.apply(fillWithTotalGoalSoFar, data=current_league_season_pd, axis=1)
            goal_so_far_list = np.array(home_away_total_goal_sofar.values.tolist())         # convert to list
            home_away_total_goal_sofar_pd = pd.DataFrame(goal_so_far_list, columns=["HOME_GOAL_SO_FAR", "AWAY_GOAL_SO_FAR"])    # convert to dataframe
            current_league_season_pd.insert(loc=5, column="HOME_TOTAL_GOAL_SO_FAR", value=home_away_total_goal_sofar_pd["HOME_GOAL_SO_FAR"]) 
            current_league_season_pd.insert(loc=6, column="AWAY_TOTAL_GOAL_SO_FAR", value=home_away_total_goal_sofar_pd["AWAY_GOAL_SO_FAR"])     

            # merge with ELO
            result_elo_pd = current_league_season_pd['Link'].apply(fillWithELO)   
            elo_list = np.array(result_elo_pd.values.tolist())
            elo_df = pd.DataFrame(elo_list, columns=["ELO_HOME", "ELO_AWAY"])
            current_league_season_pd.insert(loc=8, column="ELO_HOME", value=elo_df["ELO_HOME"]) 
            current_league_season_pd.insert(loc=9, column="ELO_AWAY", value=elo_df["ELO_AWAY"]) 

            # get recent performance
            home_away_recent_perf = current_league_season_pd.apply(fillWithRecentPerformance, data=current_league_season_pd, axis=1)
            perf_list = np.array(home_away_recent_perf.values.tolist())
            home_away_perf_pd = pd.DataFrame(perf_list, columns=["HOME_LAST_6_GOAL_DIFF", "AWAY_LAST_6_GOAL_DIFF"])
            current_league_season_pd.insert(loc=7, column="HOME_LAST_6_GOAL_DIFF", value=home_away_perf_pd["HOME_LAST_6_GOAL_DIFF"]) 
            current_league_season_pd.insert(loc=8, column="AWAY_LAST_6_GOAL_DIFF", value=home_away_perf_pd["AWAY_LAST_6_GOAL_DIFF"]) 

            result_with_goal_sofar_pd = pd.concat([result_with_goal_sofar_pd, current_league_season_pd])

# export to csv
result_with_goal_sofar_pd.to_csv('cleaned_dataset.csv', index=False)
```

<br />    
<br />  





  
## Milestone 4 - Upload the data to the database

### Setup AWS postgreSQL RDS 
I created a postgre instance in AWS for storing the result data.  

### Pipeline 
I created a pipeline.py script with a class to handle all tasks of data I did in the previous milestones. 

```python
import re
import pandas as pd
import os
import numpy as np
from csv import reader
import plotly.express as px
import missingno as msno
import pickle
from database import Database

class DataInterface:
	def __init__(self, directory):
		self.dir_error = False
		self.data = pd.DataFrame()            # store the dataframe

		if not os.path.exists(directory):
			print("Error: Database directory not found! You have to check if the database folder is exist")
			self.dir_error = True
		else:
			self.loadData(directory)

	def loadData(self, directory):
		# load data files
		pass


class FootballResultInterface(DataInterface):
	def loadData(self, dir):
		# load pickle and read content
		self.elo_data = pickle.load(open('./ELO/elo_dict.pkl', 'rb'))

		# load all directory as league name list
		leagues = [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]

		# loop to open csv
		result_with_goal_sofar_pd = pd.DataFrame()
		for league in leagues:
			print("process league: " + league + "...")
			league_folder = os.path.join(dir, league)
			csv_file_for_league = [os.path.join(league_folder, name) for name in os.listdir(league_folder) if name.endswith('.csv')]
			
			for csv_filename in csv_file_for_league:
				current_league_season_pd = pd.read_csv(csv_filename, skiprows=[0], names=["Home_Team", "Away_Team", "Result", "Link", "Season", "Round", "League"])

				# Divide result into home_score and away_score
				df_score =  current_league_season_pd['Result'].str.extract(r'(\d)-(\d)')
				current_league_season_pd.insert(loc=3, column="Home_Score", value=df_score[0].astype('Int64'))     # use Int64 as it support NaN
				current_league_season_pd.insert(loc=4, column="Away_Score", value=df_score[1].astype('Int64')) 

				if len(current_league_season_pd)>0:
					# get home team and away team total goal so far
					home_away_total_goal_sofar = current_league_season_pd.apply(self.fillWithTotalGoalSoFar, data=current_league_season_pd, axis=1)
					goal_so_far_list = np.array(home_away_total_goal_sofar.values.tolist())         # convert to list
					home_away_total_goal_sofar_pd = pd.DataFrame(goal_so_far_list, columns=["HOME_GOAL_SO_FAR", "AWAY_GOAL_SO_FAR"])    # convert to dataframe
					current_league_season_pd.insert(loc=5, column="HOME_TOTAL_GOAL_SO_FAR", value=home_away_total_goal_sofar_pd["HOME_GOAL_SO_FAR"].astype('Int64')) 
					current_league_season_pd.insert(loc=6, column="AWAY_TOTAL_GOAL_SO_FAR", value=home_away_total_goal_sofar_pd["AWAY_GOAL_SO_FAR"].astype('Int64'))     

					# merge with ELO
					result_elo_pd = current_league_season_pd['Link'].apply(self.fillWithELO)   
					elo_list = np.array(result_elo_pd.values.tolist())
					elo_df = pd.DataFrame(elo_list, columns=["ELO_HOME", "ELO_AWAY"])
					current_league_season_pd.insert(loc=8, column="ELO_HOME", value=elo_df["ELO_HOME"].astype('Int64')) 
					current_league_season_pd.insert(loc=9, column="ELO_AWAY", value=elo_df["ELO_AWAY"].astype('Int64')) 

					# get recent performance
					home_away_recent_perf = current_league_season_pd.apply(self.fillWithRecentPerformance, data=current_league_season_pd, axis=1)
					perf_list = np.array(home_away_recent_perf.values.tolist())
					home_away_perf_pd = pd.DataFrame(perf_list, columns=["HOME_LAST_6_GOAL_DIFF", "AWAY_LAST_6_GOAL_DIFF"])
					current_league_season_pd.insert(loc=7, column="HOME_LAST_6_GOAL_DIFF", value=home_away_perf_pd["HOME_LAST_6_GOAL_DIFF"].astype('Int64')) 
					current_league_season_pd.insert(loc=8, column="AWAY_LAST_6_GOAL_DIFF", value=home_away_perf_pd["AWAY_LAST_6_GOAL_DIFF"].astype('Int64')) 

					result_with_goal_sofar_pd = pd.concat([result_with_goal_sofar_pd, current_league_season_pd])

		# delete no value column
		result_with_goal_sofar_pd.drop('Result', inplace=True, axis=1)
		result_with_goal_sofar_pd.drop('Link', inplace=True, axis=1)

		# reorder dataframe column
		result_with_goal_sofar_pd.insert(0, 'League', result_with_goal_sofar_pd.pop('League'))
		result_with_goal_sofar_pd.insert(1, 'Season', result_with_goal_sofar_pd.pop('Season'))
		result_with_goal_sofar_pd.insert(2, 'Round', result_with_goal_sofar_pd.pop('Round'))
		result_with_goal_sofar_pd.insert(5, 'ELO_HOME', result_with_goal_sofar_pd.pop('ELO_HOME'))
		result_with_goal_sofar_pd.insert(6, 'ELO_AWAY', result_with_goal_sofar_pd.pop('ELO_AWAY'))

		self.data = result_with_goal_sofar_pd

		# export to csv
		self.data.to_csv('cleaned_dataset.csv', index=False)

		# save to RDS
		db = Database()
		self.data.to_sql('football_result', db.engine, if_exists='replace')
		db.close()


	def getLeagueSeasonTeamBeforeRoundTotalGoal(self, data, league, season, team, round):
		# determine home or away and get the score 
		# get home game of the team
		home_pd = data[(data["League"]==league) & (data["Home_Team"]==team) & (data["Season"]==season) & (data["Round"]<round)]
		df_home_score_sofar =  home_pd['Result'].str.extract(r'(\d)-\d')
		home_total_score = df_home_score_sofar[0].astype('Int64').sum()

		# get away game of the team
		away_pd = data[(data["League"]==league) & (data["Away_Team"]==team) & (data["Season"]==season) & (data["Round"]<round)]
		df_away_score_sofar =  away_pd['Result'].str.extract(r'\d-(\d)')
		away_total_score = df_away_score_sofar[0].astype('Int64').sum()

		# calculate total goals
		return (home_total_score + away_total_score)


	def fillWithTotalGoalSoFar(self, record, data):
		# get home team and away team and round
		league = record['League']
		season = record['Season']
		round = record['Round']
		hteam = record['Home_Team']
		ateam = record['Away_Team']
		
		home_goal_so_far = self.getLeagueSeasonTeamBeforeRoundTotalGoal(data, league, season, hteam, round)
		away_goal_so_far = self.getLeagueSeasonTeamBeforeRoundTotalGoal(data, league, season, ateam, round)

		return [home_goal_so_far, away_goal_so_far]


	def fillWithELO(self, link):
		if link not in self.elo_data:
			return [pd.NA, pd.NA]
		else:
			return [self.elo_data[link]['Elo_home'], self.elo_data[link]['Elo_away']]


	def findRecentPreviousRounds(self, currentRound, limit):
		if currentRound<=limit:
			return None
		else:
			r = []
			for l in range(limit):
				r.append(currentRound - (limit-l))
			return r


	def findLeagueSeasonTeamRecentPreviousRounds(self, data, league, season, team, round):
		rounds = self.findRecentPreviousRounds(round, 6)         # by definition is 6, can change for optimization
		if rounds is None:
			return None

		previous_matches_pd =  data[(data["League"]==league) & ((data["Home_Team"]==team) | (data["Away_Team"]==team)) & (data["Season"]==season) & (data["Round"].isin(rounds))]
		recent_perf = 0
		for index, row in previous_matches_pd.iterrows():
			hteam = row['Home_Team']
			ateam = row['Away_Team']
			if hteam==team:
				recent_perf = recent_perf + (row['Home_Score']-row['Away_Score'])
			else:
				recent_perf = recent_perf + (row['Away_Score']-row['Home_Score'])

		return recent_perf


	def fillWithRecentPerformance(self, record, data):
		# get home team and away team and round
		league = record['League']
		season = record['Season']
		round = record['Round']
		hteam = record['Home_Team']
		ateam = record['Away_Team']
		
		home_team_goal_diff = self.findLeagueSeasonTeamRecentPreviousRounds(data, league, season, hteam, round)
		away_team_goal_diff = self.findLeagueSeasonTeamRecentPreviousRounds(data, league, season, ateam, round)

		return [home_team_goal_diff, away_team_goal_diff]



if __name__ == '__main__':
	result_data = FootballResultInterface('./Results')

	print(result_data.data)
```

For exporting the data to a local csv and remote RDS:

```python
# export to csv
self.data.to_csv('cleaned_dataset.csv', index=False)

# save to RDS
db = Database()
self.data.to_sql('football_result', db.engine, if_exists='replace')
db.close()
```

<br />    
<br />  
  
  
  
  
  
## Milestone 5 - Model Training

### Train a simple model to obtain a baseline score
The full dataframe has been saved in a local csv file. So for training a simple model, I read the full set of data to a dataframe:

```python
full_pd = pd.read_csv("cleaned_dataset.csv")
full_pd
```

And here is the function to filter data for the baseline model:
```python
# functions to filter different league
def getLeagueData(data, league, season=None):
    if season is None:
        league_pd =  data[(data["League"]==league)]
    else:
        league_pd =  data[(data["League"]==league) & (data["Season"]==season)]
    return league_pd
    
model_pd = getLeagueData(full_pd, "serie_b", 2011)
```

Some fields in the dataset is not useful for the model, and so I drop those columns from the dataframe:

```python
# delete no value column
model_pd.drop('League', inplace=True, axis=1)
model_pd.drop('Season', inplace=True, axis=1)
model_pd.drop('Round', inplace=True, axis=1)
model_pd.drop('Home_Team', inplace=True, axis=1)
model_pd.drop('Away_Team', inplace=True, axis=1)
model_pd.drop('Home_Score', inplace=True, axis=1)
model_pd.drop('Away_Score', inplace=True, axis=1)
```

![Baseline Model Dataframe](https://github.com/frankie-2nfro-com/football_match_outcome_prediction/blob/main/Screens/M5T1_data.png?raw=true)

After finalize the data, It's time to prepare features and result for training my supervised machine learning model:

```python
array = model_pd.values

array([[53., 58.,  1., ..., -1.,  1.,  1.],
       [57., 62.,  4., ...,  2., -1.,  0.],
       [64., 52.,  2., ...,  1., -3.,  1.],
       ...,
       [54., 58., 24., ..., -1.,  1.,  1.],
       [55., 64., 31., ..., -1.,  4.,  1.],
       [59., 54., 25., ..., -1., -2.,  1.]])
```
       
```python
X = array[:,0:8].astype('int')
y = array[:,8].astype('int')
```

Before start fitting data to the model, it is better to rescale the data for model input:

```python
# Scaler
from sklearn.preprocessing import MinMaxScaler
from numpy import set_printoptions

scaler = MinMaxScaler(feature_range=(0, 8))
rescaledX = scaler.fit_transform(X)

# summarize transformed data
set_printoptions(precision=3)
```

I will only simple use a logistic regression for the baseline model:
```python
test_size = 0.3
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(rescaledX, y, test_size=test_size, random_state=seed)

model = LogisticRegression() 
model.fit(X_train, Y_train)
```

And the accuracy result is as follows:

```python
result = model.score(X_train, Y_train) 
print("Accuracy for train: %.3f%%" % (result*100.0))

result = model.score(X_test, Y_test) 
print("Accuracy for test: %.3f%%" % (result*100.0))
```

Accuracy for train: 47.333%

Accuracy for test: 42.636%

And then I saved the baseline model:

```python
from joblib import dump, load
dump(model, 'baseline.joblib')
```

And the complete code for this task can be found in [model_m5_t1.ipynb](https://github.com/frankie-2nfro-com/football_match_outcome_prediction/blob/main/model_m5_t1.ipynb)

<br /> 

### Perform feature selection

In previous task, I tried to use all features of my data namely Elo_home, Elo_away, HOMETEAM_HOME_GOAL_SO_FAR, HOMETEAM_AWAY_GOAL_SO_FAR, AWAYTEAM_HOME_GOAL_SO_FAR, AWAYTEAM_AWAY_GOAL_SO_FAR, HOME_LASTEST_GOAL_DIFF, AWAY_LASTEST_GOAL_DIFF and Result. And the baseline score is:

```python
Accuracy for train: 47.333%
Accuracy for test: 42.636%
```

In this task, I will try to take a look at the weights of the features and see which ones are important. Remove those that have low weights and check again the performance.

Elo_home and Elo_away are the estimated team strength for both teams. I will try to use the different of them as a new feature called ELO_DIFF. The function is as follows:

```python
def get_ELO_diff(record):
    hscore = record['Elo_home']
    ascore = record['Elo_away']
    return (hscore - ascore)
```

HOME_LASTEST_GOAL_DIFF and AWAY_LASTEST_GOAL_DIFF are the goal different of both teams for their latest 3 games. I will try to use the different of them as a feature called RECENT_PERF_DIFF. The function is as follows:

```python
def get_recent_goal_diff_diff(record):
    hscore = record['HOME_LASTEST_GOAL_DIFF']
    ascore = record['AWAY_LASTEST_GOAL_DIFF']
    return hscore - ascore
```

HOMETEAM_HOME_GOAL_SO_FAR, HOMETEAM_AWAY_GOAL_SO_FAR, AWAYTEAM_HOME_GOAL_SO_FAR and AWAYTEAM_AWAY_GOAL_SO_FAR are the so far home and away goals for both teams. So HOMETEAM_HOME_GOAL_SO_FAR is more valid for home team; and AWAYTEAM_AWAY_GOAL_SO_FAR is more valid for away team. So I will drop HOMETEAM_AWAY_GOAL_SO_FAR and  AWAYTEAM_HOME_GOAL_SO_FAR featues. And I will try to use the different of them as a feature called HOME_AWAY_GOAL_DIFF. The function is as follows:

```python
def get_home_away_total_goal_diff(record):
    hgoal = record['HOMETEAM_HOME_GOAL_SO_FAR']
    agoal = record['AWAYTEAM_AWAY_GOAL_SO_FAR']
    return hgoal - agoal
```

To make those changes to the dataframe:

```python
model_pd = model_pd.dropna()

elo_diff_pd = model_pd.apply(get_ELO_diff, axis=1)
model_pd.drop('Elo_home', inplace=True, axis=1)
model_pd.drop('Elo_away', inplace=True, axis=1)
model_pd.insert(loc=5, column="ELO_DIFF", value=elo_diff_pd.astype('Int64')) 

recent_perf_diff_pd = model_pd.apply(get_recent_goal_diff_diff, axis=1)
model_pd.drop('HOME_LASTEST_GOAL_DIFF', inplace=True, axis=1)
model_pd.drop('AWAY_LASTEST_GOAL_DIFF', inplace=True, axis=1)
model_pd.insert(loc=6, column="RECENT_PERF_DIFF", value=recent_perf_diff_pd.astype('Int64')) 

goal_diff_pd = model_pd.apply(get_home_away_total_goal_diff, axis=1)
model_pd.drop('HOMETEAM_HOME_GOAL_SO_FAR', inplace=True, axis=1)
model_pd.drop('HOMETEAM_AWAY_GOAL_SO_FAR', inplace=True, axis=1)
model_pd.drop('AWAYTEAM_HOME_GOAL_SO_FAR', inplace=True, axis=1)
model_pd.drop('AWAYTEAM_AWAY_GOAL_SO_FAR', inplace=True, axis=1)
model_pd.insert(loc=7, column="HOME_AWAY_GOAL_DIFF", value=recent_perf_diff_pd.astype('Int64')) 

# delete no value column
model_pd.drop('League', inplace=True, axis=1)
model_pd.drop('Season', inplace=True, axis=1)
model_pd.drop('Round', inplace=True, axis=1)
model_pd.drop('Home_Team', inplace=True, axis=1)
model_pd.drop('Away_Team', inplace=True, axis=1)
```

And the dataframe becomes:

![Task 2 Model Dataframe](https://github.com/frankie-2nfro-com/football_match_outcome_prediction/blob/main/Screens/M5T2_data.png?raw=true)

And to rebuild the model as follows:

```python

array = model_pd.values
X = array[:,0:(array.shape[1]-1)].astype('int')
y = array[:,(array.shape[1]-1)].astype('int')

# Scaler
scaler = MinMaxScaler(feature_range=(0, 8))
rescaledX = scaler.fit_transform(X)

# summarize transformed data
set_printoptions(precision=3)

# Or Standardize
#scaler = StandardScaler().fit(X)
#rescaledX = scaler.transform(X)

test_size = 0.3
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(rescaledX, y, test_size=test_size,
random_state=seed)

model = LogisticRegression() 
model.fit(X_train, Y_train)

result = model.score(X_train, Y_train) 
print("Accuracy for train: %.3f%%" % (result*100.0))

result = model.score(X_test, Y_test) 
print("Accuracy for test: %.3f%%" % (result*100.0))
``` 

And the result is as follows:

```python
Accuracy for train: 58.000%
Accuracy for test: 48.837%
```

And finally I save the modal as follows:

```python
from joblib import dump, load
dump(model, 'baseline_t2.joblib')
```

And the complete code for this task can be found in [model_m5_t2.ipynb](https://github.com/frankie-2nfro-com/football_match_outcome_prediction/blob/main/model_m5_t2.ipynb)

<br /> 

### Train and tune other models

For trying to know performance of other models, I create this function to test different models one by one:

```python
def tryModels(X, y):
    test_size = 0.3
    seed = 42
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    # Model creation
    print("LogisticRegression")
    model = LogisticRegression() 
    model.fit(X_train, Y_train)
    result = model.score(X_train, Y_train) 
    print("Accuracy for train: %.3f%%" % (result*100.0))
    result = model.score(X_test, Y_test) 
    print("Accuracy for test: %.3f%%" % (result*100.0))
    print()

    # KNN 
    print("KNN")
    knn = KNeighborsClassifier(n_neighbors=15)
    knn.fit(X_train, Y_train)
    result = knn.score(X_train, Y_train) 
    print("Accuracy for train: %.3f%%" % (result*100.0))
    result = knn.score(X_test, Y_test) 
    print("Accuracy for test: %.3f%%" % (result*100.0))
    print()

    # decision trees  
    print("decision trees")
    clf = DecisionTreeClassifier()
    clf.fit(X_train, Y_train)
    result = clf.score(X_train, Y_train) 
    print("Accuracy for train: %.3f%%" % (result*100.0))
    result = clf.score(X_test, Y_test) 
    print("Accuracy for test: %.3f%%" % (result*100.0))
    print()

    # random forests
    print("random forests")
    model = RandomForestClassifier()
    model.fit(X_train, Y_train)
    result = model.score(X_train, Y_train) 
    print("Accuracy for train: %.3f%%" % (result*100.0))
    result = model.score(X_test, Y_test) 
    print("Accuracy for test: %.3f%%" % (result*100.0))
    print()
```

And also I would loop data by diffent league to cross check which data set would have better performance for different models:

```python
# load all directory as league name list
dir = "./Results"
leagues = [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]

# loop to open csv
result_with_goal_sofar_pd = pd.DataFrame()
for league in leagues:
    model_pd = getLeagueData(full_pd, league)
    model_pd = model_pd.dropna()

    if (model_pd.shape[0]==0):
        continue

    elo_diff_pd = model_pd.apply(get_ELO_diff, axis=1)
    model_pd.drop('Elo_home', inplace=True, axis=1)
    model_pd.drop('Elo_away', inplace=True, axis=1)
    model_pd.insert(loc=5, column="ELO_DIFF", value=elo_diff_pd.astype('Int64')) 
    
    recent_perf_diff_pd = model_pd.apply(get_recent_goal_diff_diff, axis=1)
    model_pd.drop('HOME_LASTEST_GOAL_DIFF', inplace=True, axis=1)
    model_pd.drop('AWAY_LASTEST_GOAL_DIFF', inplace=True, axis=1)
    model_pd.insert(loc=6, column="RECENT_PERF_DIFF", value=recent_perf_diff_pd.astype('Int64')) 

    goal_diff_pd = model_pd.apply(get_home_away_total_goal_diff, axis=1)
    model_pd.drop('HOMETEAM_HOME_GOAL_SO_FAR', inplace=True, axis=1)
    model_pd.drop('HOMETEAM_AWAY_GOAL_SO_FAR', inplace=True, axis=1)
    model_pd.drop('AWAYTEAM_HOME_GOAL_SO_FAR', inplace=True, axis=1)
    model_pd.drop('AWAYTEAM_AWAY_GOAL_SO_FAR', inplace=True, axis=1)
    model_pd.insert(loc=7, column="HOME_AWAY_GOAL_DIFF", value=recent_perf_diff_pd.astype('Int64')) 

    # delete no value column
    model_pd.drop('League', inplace=True, axis=1)
    model_pd.drop('Season', inplace=True, axis=1)
    model_pd.drop('Round', inplace=True, axis=1)
    model_pd.drop('Home_Team', inplace=True, axis=1)
    model_pd.drop('Away_Team', inplace=True, axis=1)

    array = model_pd.values
    X = array[:,0:(array.shape[1]-1)].astype('int')
    y = array[:,(array.shape[1]-1)].astype('int')

    # Scaler
    scaler = MinMaxScaler(feature_range=(0, 8))
    rescaledX = scaler.fit_transform(X)

    # summarize transformed data
    set_printoptions(precision=3)

    # Or Standardize
    #scaler = StandardScaler().fit(X)
    #rescaledX = scaler.transform(X)

    print()
    print()
    print(league)
    print("-------------------------------------")
    tryModels(rescaledX, y)
```

And the result is as follows:

```python


championship
-------------------------------------
LogisticRegression
Accuracy for train: 55.962%
Accuracy for test: 57.025%

KNN
Accuracy for train: 60.897%
Accuracy for test: 54.111%

decision trees
Accuracy for train: 71.186%
Accuracy for test: 54.036%

random forests
Accuracy for train: 71.186%
Accuracy for test: 54.858%



primeira_liga
-------------------------------------
LogisticRegression
Accuracy for train: 64.117%
Accuracy for test: 62.669%

KNN
Accuracy for train: 67.468%
Accuracy for test: 62.137%

decision trees
Accuracy for train: 70.398%
Accuracy for test: 60.213%

random forests
Accuracy for train: 70.398%
Accuracy for test: 60.008%



ligue_1
-------------------------------------
LogisticRegression
Accuracy for train: 60.556%
Accuracy for test: 58.078%

KNN
Accuracy for train: 63.293%
Accuracy for test: 57.530%

decision trees
Accuracy for train: 66.528%
Accuracy for test: 56.337%

random forests
Accuracy for train: 66.528%
Accuracy for test: 55.563%



segunda_division
-------------------------------------
LogisticRegression
Accuracy for train: 56.805%
Accuracy for test: 56.740%

KNN
Accuracy for train: 59.673%
Accuracy for test: 53.793%

decision trees
Accuracy for train: 62.573%
Accuracy for test: 52.281%

random forests
Accuracy for train: 62.562%
Accuracy for test: 51.896%



2_liga
-------------------------------------
LogisticRegression
Accuracy for train: 56.986%
Accuracy for test: 57.154%

KNN
Accuracy for train: 62.009%
Accuracy for test: 53.924%

decision trees
Accuracy for train: 66.983%
Accuracy for test: 54.713%

random forests
Accuracy for train: 66.983%
Accuracy for test: 54.337%



serie_a
-------------------------------------
LogisticRegression
Accuracy for train: 64.120%
Accuracy for test: 63.587%

KNN
Accuracy for train: 67.083%
Accuracy for test: 60.908%

decision trees
Accuracy for train: 69.945%
Accuracy for test: 59.084%

random forests
Accuracy for train: 69.945%
Accuracy for test: 58.075%



bundesliga
-------------------------------------
LogisticRegression
Accuracy for train: 60.661%
Accuracy for test: 60.805%

KNN
Accuracy for train: 63.824%
Accuracy for test: 57.787%

decision trees
Accuracy for train: 68.120%
Accuracy for test: 55.551%

random forests
Accuracy for train: 68.120%
Accuracy for test: 55.775%



primera_division
-------------------------------------
LogisticRegression
Accuracy for train: 61.193%
Accuracy for test: 60.372%

KNN
Accuracy for train: 64.462%
Accuracy for test: 59.722%

decision trees
Accuracy for train: 67.338%
Accuracy for test: 57.329%

random forests
Accuracy for train: 67.325%
Accuracy for test: 57.417%



ligue_2
-------------------------------------
LogisticRegression
Accuracy for train: 57.143%
Accuracy for test: 55.854%

KNN
Accuracy for train: 60.959%
Accuracy for test: 53.448%

decision trees
Accuracy for train: 64.673%
Accuracy for test: 53.288%

random forests
Accuracy for train: 64.673%
Accuracy for test: 53.208%



premier_league
-------------------------------------
LogisticRegression
Accuracy for train: 63.322%
Accuracy for test: 61.471%

KNN
Accuracy for train: 65.374%
Accuracy for test: 59.370%

decision trees
Accuracy for train: 68.126%
Accuracy for test: 57.151%

random forests
Accuracy for train: 68.114%
Accuracy for test: 57.764%



eredivisie
-------------------------------------
LogisticRegression
Accuracy for train: 66.248%
Accuracy for test: 64.947%

KNN
Accuracy for train: 68.442%
Accuracy for test: 63.267%

decision trees
Accuracy for train: 71.709%
Accuracy for test: 60.805%

random forests
Accuracy for train: 71.709%
Accuracy for test: 60.844%



segunda_liga
-------------------------------------
LogisticRegression
Accuracy for train: 92.715%
Accuracy for test: 93.846%

KNN
Accuracy for train: 90.728%
Accuracy for test: 96.923%

decision trees
Accuracy for train: 100.000%
Accuracy for test: 93.846%

random forests
Accuracy for train: 100.000%
Accuracy for test: 93.846%



serie_b
-------------------------------------
LogisticRegression
Accuracy for train: 56.848%
Accuracy for test: 57.238%

KNN
Accuracy for train: 60.138%
Accuracy for test: 54.552%

decision trees
Accuracy for train: 63.636%
Accuracy for test: 53.052%

random forests
Accuracy for train: 63.621%
Accuracy for test: 53.471%

```

Random Forests seems to have a quite good result. But it should be overfitting. So I think I need to add some regularisation to fix and improve the model. Firstly I will try to visualize the decision tree of the forest by the follows code:

```python 
from sklearn.tree import export_graphviz
import pydot

# Extract the small tree
tree_small = model.estimators_[3]
    
# Save the tree as a png image
export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = ["ELO_DIFF","RECENT_PERF_DIFF","HOME_AWAY_GOAL_DIFF"], rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
graph.write_png('small_tree.png');
```

And here is the picture:

![Visualized tree for random forests model](https://github.com/frankie-2nfro-com/football_match_outcome_prediction/blob/main/small_tree.png?raw=true)

Actually the tree is too complicated. I will try to set n_estimators to 10 and max_depth to 4:

```python
model = RandomForestClassifier(n_estimators=10, max_depth = 4)
model.fit(X_train, Y_train)
result = model.score(X_train, Y_train) 
print("Accuracy for train: %.3f%%" % (result*100.0))
result = model.score(X_test, Y_test) 
print("Accuracy for test: %.3f%%" % (result*100.0))
print()
```

![Visualized tree for random forests model](https://github.com/frankie-2nfro-com/football_match_outcome_prediction/blob/main/small_tree_re.png?raw=true)


And the complete code for this task can be found in [model_m5_t3.ipynb](https://github.com/frankie-2nfro-com/football_match_outcome_prediction/blob/main/model_m5_t3.ipynb)

And the score is:

```python
Accuracy for train: 65.667%
Accuracy for test: 52.713%
```

The overfitting issue seems reduced.

And the complete code for this task can be found in [model_m5_t3.ipynb](https://github.com/frankie-2nfro-com/football_match_outcome_prediction/blob/main/model_m5_t3.ipynb)

<br /> 

### Iteratively train the model with different subsets of the data

In the dataset, some data is quite old and may influence the representative of the current football data. But if just removing too much old data will also leave us with fewer data points. So I will try to find a balance by testing how much data we could remove by observing the scores. 

For removing old data, I make this function:

```python
def getLeagueData(data, league, seasonFrom=None):
    if seasonFrom is None:
        league_pd =  data[(data["League"]==league)]
    else:
        league_pd =  data[(data["League"]==league) & (data["Season"]>=seasonFrom)]
    return league_pd
```

So I try datasets from 1990, 1995, 2000, 2005, 2010, 2015 to the latest record. And if take the highest score league for random forests, the result is as follows:

```python
segunda_liga (from 1990)
-------------------------------------
Accuracy for train: 98.013%
Accuracy for test: 95.385%

segunda_liga (from 1995)
-------------------------------------
Accuracy for train: 98.675%
Accuracy for test: 93.846%

segunda_liga (from 2000)
-------------------------------------
Accuracy for train: 98.675%
Accuracy for test: 93.846%

segunda_liga (from 2005)
-------------------------------------
Accuracy for train: 98.675%
Accuracy for test: 93.846%

segunda_liga (from 2010)
-------------------------------------
Accuracy for train: 97.351%
Accuracy for test: 95.385%

segunda_liga (from 2015)
-------------------------------------
Accuracy for train: 100.000%
Accuracy for test: 100.000%
```

And the complete code for this task can be found in [model_explained.ipynb](https://github.com/frankie-2nfro-com/football_match_outcome_prediction/blob/main/model_explained.ipynb)


<br />    
<br />  





## Milestone 6 - Infference

### Scrape data of matches that haven't taken place for making predictions

All data for predition is in the folder called "Predict". And here is the code for loading the data and combine with the pkl files:

```python
# load all directory as league name list
dir = "./Predict/Results/"
leagues = [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]

# loop to open csv
result_with_goal_sofar_pd = pd.DataFrame()
for league in leagues:
    league_folder = os.path.join(dir, league)

    csv_file_for_league = [os.path.join(league_folder, name) for name in os.listdir(league_folder) if name.endswith('.csv')]
    pkl_file_for_league = [os.path.join(league_folder, name) for name in os.listdir(league_folder) if name.endswith('.pkl')]

    if len(csv_file_for_league)==1 and len(pkl_file_for_league)==1:
        csv_filename = csv_file_for_league[0]
        pkl_filename = pkl_file_for_league[0]

        current_league_season_pd = pd.read_csv(csv_filename, skiprows=[0], names=["Home_Team", "Away_Team", "Result", "Link", "Season", "Round", "League"])

        # Divide result into home_score and away_score
        df_score =  current_league_season_pd['Result'].str.extract(r'(\d)-(\d)')
        current_league_season_pd.insert(loc=3, column="Home_Score", value=df_score[0].astype('Int64'))     # use Int64 as it support NaN
        current_league_season_pd.insert(loc=4, column="Away_Score", value=df_score[1].astype('Int64')) 

        if len(current_league_season_pd)>0:
            # load pickle and read content
            d = pickle.load(open(pkl_filename, 'rb'))
            elo_key_df = pd.DataFrame(d.keys(), columns=["link"])
            elo_val_df = pd.DataFrame.from_dict(d.values())
            elo_df = elo_key_df.join(elo_val_df)

            current_league_season_pd = current_league_season_pd.merge(elo_df, left_on='Link', right_on='link')

            result_with_goal_sofar_pd = pd.concat([result_with_goal_sofar_pd, current_league_season_pd])  
```

After that, trying to drop all NaN and remove columns:

```python
full_pd = result_with_goal_sofar_pd.dropna()

# delete no value column
full_pd.drop('Result', inplace=True, axis=1)
full_pd.drop('Link', inplace=True, axis=1)
full_pd.drop('link', inplace=True, axis=1)
```

And creating the result column:

```python
# find who win H:Home A:Away D:Draw
def get_result(record):
    hscore = record['Home_Score']
    ascore = record['Away_Score']
    if hscore is pd.NA or ascore is pd.NA:
        return pd.NA
    if hscore>ascore:
        return 1
    else:
        return 0

result_pd = full_pd.apply(get_result, axis=1)

full_pd.insert(loc=len(full_pd.columns), column="Result", value=result_pd.astype('Int64')) 

# reorder dataframe column
full_pd.insert(0, 'League', full_pd.pop('League'))
full_pd.insert(1, 'Season', full_pd.pop('Season'))
full_pd.insert(2, 'Round', full_pd.pop('Round'))
full_pd.insert(3, 'Home_Team', full_pd.pop('Home_Team'))
full_pd.insert(4, 'Away_Team', full_pd.pop('Away_Team'))
```

![Current season leage Dataframe](https://github.com/frankie-2nfro-com/football_match_outcome_prediction/blob/main/Screens/M6T1_data.png?raw=true)

And finally saving the dataframe to csv:

```python
full_pd.to_csv('results_for_prediction.csv', index=False)
```

And the complete code for this task can be found in [predict_m6_t1.ipynb](https://github.com/frankie-2nfro-com/football_match_outcome_prediction/blob/main/predict_m6_t1.ipynb)


<br /> 

### Use the pipeline you created to clean the scraped data

Also, firstly trying to loop the league folder to load the predict dataframe and combine with the pkl files:

```python
# functions to filter different league
def getLeagueData(data, league, season=None):
    if season is None:
        league_pd =  data[(data["League"]==league)]
    else:
        league_pd =  data[(data["League"]==league) & (data["Season"]==season)]
    return league_pd
    
def getLeagueSeasonTeamBeforeRoundTotalGoal(data, team, round):
    # determine home or away and get the score 
    # get home game of the team
    home_pd = data[(data["Home_Team"]==team) & (data["Round"]<round)]
    home_total_score = home_pd['Home_Score'].astype('Int64').sum()

    # get away game of the team
    away_pd = data[(data["Away_Team"]==team) & (data["Round"]<round)]
    away_total_score = home_pd['Away_Score'].astype('Int64').sum()

    # calculate total goals
    return home_total_score, away_total_score


def fillWithTotalGoalSoFar(record, data):
    # get home team and away team and round
    round = record['Round']
    hteam = record['Home_Team']
    ateam = record['Away_Team']
    
    hometeam_home_goal_so_far, hometeam_away_goal_so_far = getLeagueSeasonTeamBeforeRoundTotalGoal(data, hteam, round)
    awayteam_home_goal_so_far, awayteam_away_goal_so_far = getLeagueSeasonTeamBeforeRoundTotalGoal(data, ateam, round)

    return [hometeam_home_goal_so_far, hometeam_away_goal_so_far, awayteam_home_goal_so_far, awayteam_away_goal_so_far]
    

def findRecentPreviousRounds(currentRound, limit):
    if currentRound<=limit:
        return None
    else:
        r = []
        for l in range(limit):
            r.append(currentRound - (limit-l))
        return r


def findLeagueSeasonTeamRecentPreviousRounds(data, team, round):
    rounds = findRecentPreviousRounds(round, RECENT_PREFORMANCE_MATCH_COUNT)         # can change for optimization
    if rounds is None:
        return None

    previous_matches_pd =  data[((data["Home_Team"]==team) | (data["Away_Team"]==team)) & (data["Round"].isin(rounds))]
    recent_perf = 0
    for index, row in previous_matches_pd.iterrows():
        hteam = row['Home_Team']
        ateam = row['Away_Team']
        if hteam==team:
            recent_perf = recent_perf + (row['Home_Score']-row['Away_Score'])
        else:
            recent_perf = recent_perf + (row['Away_Score']-row['Home_Score'])

    return recent_perf


def fillWithRecentPerformance(record, data):
    # get home team and away team and round
    round = record['Round']
    hteam = record['Home_Team']
    ateam = record['Away_Team']
    
    home_team_goal_diff = findLeagueSeasonTeamRecentPreviousRounds(data, hteam, round)
    away_team_goal_diff = findLeagueSeasonTeamRecentPreviousRounds(data, ateam, round)

    return [home_team_goal_diff, away_team_goal_diff]
    
    
def get_ELO_diff(record):
    hscore = record['Elo_home']
    ascore = record['Elo_away']
    return (hscore - ascore)
    
    
def get_recent_goal_diff_diff(record):
    hscore = record['HOME_LASTEST_GOAL_DIFF']
    ascore = record['AWAY_LASTEST_GOAL_DIFF']
    return hscore - ascore
    
    
def get_home_away_total_goal_diff(record):
    hgoal = record['HOMETEAM_HOME_GOAL_SO_FAR']
    agoal = record['AWAYTEAM_AWAY_GOAL_SO_FAR']
    return hgoal - agoal
    
    
# load all directory as league name list
dir = "./Predict/To_Predict/"
leagues = [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]

prediction_result_pd = pd.read_csv("results_for_prediction.csv")

# loop to open csv
predict_pd = pd.DataFrame()
for league in leagues:
    league_folder = os.path.join(dir, league)

    csv_file_for_league = [os.path.join(league_folder, name) for name in os.listdir(league_folder) if name.endswith('.csv')]
    pkl_file_for_league = [os.path.join(league_folder, name) for name in os.listdir(league_folder) if name.endswith('.pkl')]


    if len(csv_file_for_league)==1 and len(pkl_file_for_league)==1:
        csv_filename = csv_file_for_league[0]
        pkl_filename = pkl_file_for_league[0]

        # ,Home_Team,Away_Team,Link,Season,Round,League
        current_league_predict_pd = pd.read_csv(csv_filename, skiprows=[0], names=["Home_Team", "Away_Team", "Link", "Season", "Round", "League"])

        if len(current_league_predict_pd)>0:
            # load pickle and read content
            d = pickle.load(open(pkl_filename, 'rb'))
            elo_key_df = pd.DataFrame(d.keys(), columns=["link"])
            elo_val_df = pd.DataFrame.from_dict(d.values())
            elo_df = elo_key_df.join(elo_val_df)

            current_league_predict_pd = current_league_predict_pd.merge(elo_df, left_on='Link', right_on='link')

            # get this season data
            current_season_result = getLeagueData(prediction_result_pd, league)

            goal_so_far = current_league_predict_pd.apply(fillWithTotalGoalSoFar, data=current_season_result, axis=1)
            goal_so_far_list = np.array(goal_so_far.values.tolist()) 
            goal_so_far_pd = pd.DataFrame(goal_so_far_list, columns=["HOMETEAM_HOME_GOAL_SO_FAR", "HOMETEAM_AWAY_GOAL_SO_FAR", "AWAYTEAM_HOME_GOAL_SO_FAR", "AWAYTEAM_AWAY_GOAL_SO_FAR"])    # convert to dataframe
            current_league_predict_pd.insert(loc=7, column="HOMETEAM_HOME_GOAL_SO_FAR", value=goal_so_far_pd["HOMETEAM_HOME_GOAL_SO_FAR"].astype('Int64')) 
            current_league_predict_pd.insert(loc=8, column="HOMETEAM_AWAY_GOAL_SO_FAR", value=goal_so_far_pd["HOMETEAM_AWAY_GOAL_SO_FAR"].astype('Int64')) 
            current_league_predict_pd.insert(loc=9, column="AWAYTEAM_HOME_GOAL_SO_FAR", value=goal_so_far_pd["AWAYTEAM_HOME_GOAL_SO_FAR"].astype('Int64'))     
            current_league_predict_pd.insert(loc=10, column="AWAYTEAM_AWAY_GOAL_SO_FAR", value=goal_so_far_pd["AWAYTEAM_AWAY_GOAL_SO_FAR"].astype('Int64'))            

            recent_perform = current_league_predict_pd.apply(fillWithRecentPerformance, data=current_season_result, axis=1)
            perf_list = np.array(recent_perform.values.tolist())
            home_away_perf_pd = pd.DataFrame(perf_list, columns=["HOME_LASTEST_GOAL_DIFF", "AWAY_LASTEST_GOAL_DIFF"])
            current_league_predict_pd.insert(loc=11, column="HOME_LASTEST_GOAL_DIFF", value=home_away_perf_pd["HOME_LASTEST_GOAL_DIFF"].astype('Int64')) 
            current_league_predict_pd.insert(loc=12, column="AWAY_LASTEST_GOAL_DIFF", value=home_away_perf_pd["AWAY_LASTEST_GOAL_DIFF"].astype('Int64'))   

            predict_pd = pd.concat([predict_pd, current_league_predict_pd])
	    
	    
predict_pd.drop('link', inplace=True, axis=1)

# reorder dataframe column
predict_pd.insert(0, 'League', predict_pd.pop('League'))
predict_pd.insert(1, 'Season', predict_pd.pop('Season'))
predict_pd.insert(2, 'Round', predict_pd.pop('Round'))
predict_pd.insert(3, 'Home_Team', predict_pd.pop('Home_Team'))
predict_pd.insert(4, 'Away_Team', predict_pd.pop('Away_Team'))
predict_pd.insert(5, 'Elo_home', predict_pd.pop('Elo_home').astype('int'))
predict_pd.insert(6, 'Elo_away', predict_pd.pop('Elo_away').astype('int'))
predict_pd.insert(13, 'Link', predict_pd.pop('Link'))


elo_diff_pd = predict_pd.apply(get_ELO_diff, axis=1)
predict_pd.drop('Elo_home', inplace=True, axis=1)
predict_pd.drop('Elo_away', inplace=True, axis=1)
predict_pd.insert(loc=5, column="ELO_DIFF", value=elo_diff_pd.astype('Int64')) 
        
recent_perf_diff_pd = predict_pd.apply(get_recent_goal_diff_diff, axis=1)
predict_pd.drop('HOME_LASTEST_GOAL_DIFF', inplace=True, axis=1)
predict_pd.drop('AWAY_LASTEST_GOAL_DIFF', inplace=True, axis=1)
predict_pd.insert(loc=6, column="RECENT_PERF_DIFF", value=recent_perf_diff_pd.astype('Int64')) 

goal_diff_pd = predict_pd.apply(get_home_away_total_goal_diff, axis=1)
predict_pd.drop('HOMETEAM_HOME_GOAL_SO_FAR', inplace=True, axis=1)
predict_pd.drop('HOMETEAM_AWAY_GOAL_SO_FAR', inplace=True, axis=1)
predict_pd.drop('AWAYTEAM_HOME_GOAL_SO_FAR', inplace=True, axis=1)
predict_pd.drop('AWAYTEAM_AWAY_GOAL_SO_FAR', inplace=True, axis=1)
predict_pd.insert(loc=7, column="HOME_AWAY_GOAL_DIFF", value=recent_perf_diff_pd.astype('Int64')) 

# delete no value column
predict_pd.drop('Season', inplace=True, axis=1)
predict_pd.drop('Round', inplace=True, axis=1)
predict_pd.drop('Home_Team', inplace=True, axis=1)
predict_pd.drop('Away_Team', inplace=True, axis=1)
predict_pd.insert(4, 'League', predict_pd.pop('League'))        # need this field for filter

```

![Predict record Dataframe](https://github.com/frankie-2nfro-com/football_match_outcome_prediction/blob/main/Screens/M6T2_data_u.png?raw=true)

And finally, saving the predict data to csv:

```python
predict_pd.to_csv('to_predict.csv', index=False)
```

And the complete code for this task can be found in [predict_m6_t2.ipynb](https://github.com/frankie-2nfro-com/football_match_outcome_prediction/blob/main/predict_m6_t2.ipynb)



<br /> 

### Use the model to predict the results of the next matches

Trying to predict for segunda_liga because of getting high score by previous milestone. Take the data from 2015 to train the model as randomforest_segunda_liga_from2015.joblib. 

```python
model = joblib.load('./models/randomforest_segunda_liga_from2015.joblib')
```

The score for the trained model is as follows:

```python
Accuracy for train: 79.585%
Accuracy for test: 79.200%
```

And after predict, I got the result as follows:

```python
prediction = model.predict(rescaledX)
prediction_pd = pd.DataFrame(prediction, columns=["PREDICTION"])

team_predict_pd.insert(loc=4, column="PREDICTION", value=prediction_pd["PREDICTION"].astype('Int64')) 
```

![Predict result Dataframe](https://github.com/frankie-2nfro-com/football_match_outcome_prediction/blob/main/Screens/T6T3_result.png?raw=true)

By checking the actual result, I need to simple scraper to get the match result by the link:

```python
def getMatchHomeWin(record):
    url = record["Link"]
    r = requests.get(url)
    x = re.findall(r'<span class="r1">(.*?)</span> - <span class="r2">(.*?)</span>', r.text)
    if (int(x[0][0]) > int(x[0][1])):
        return 1
    else:
        return 0
	
homewin_pd = predict_pd.apply(getMatchHomeWin, axis=1)

homewin_list = np.array(homewin_pd.values.tolist())
home_win_flag_pd = pd.DataFrame(homewin_list, columns=["HOME_WIN"])
predict_pd.insert(loc=4, column="HOME_WIN", value=home_win_flag_pd["HOME_WIN"].astype('Int64')) 
```

The correct predict is only 5/9 (56%). 

```python
ELO_DIFF,RECENT_PERF_DIFF,HOME_AWAY_GOAL_DIFF,Link,PREDICTION,HOME_WIN
2,-5,-5,https://www.besoccer.com/match/penafiel/varzim/202235614,1,1
-1,-6,-6,https://www.besoccer.com/match/casa-pia/farense/202235608,0,0
-21,-5,-5,https://www.besoccer.com/match/mafra/chaves/202235611,0,1
6,-1,-1,https://www.besoccer.com/match/benfica-ii/cf-estrela-de-amadora/202235615,1,0
-6,4,4,https://www.besoccer.com/match/vilafranquense/academico-viseu/202235613,1,1
-6,8,8,https://www.besoccer.com/match/sporting-covilha/academica/202235609,1,1
-3,-5,-5,https://www.besoccer.com/match/nacional/rio-ave/202235607,0,0
-10,-5,-5,https://www.besoccer.com/match/porto-ii/feirense/202235610,0,0
-8,-3,-3,https://www.besoccer.com/match/trofense/leixoes/202235612,0,1
```

And the complete code for this task can be found in [model_results.ipynb](https://github.com/frankie-2nfro-com/football_match_outcome_prediction/blob/main/model_results.ipynb)


<br /> 

### Project conclusion

If I want to draw the conclusion of the model performance, I think I should show all the matrix against models and dataset. And here is the results for primeira_liga which performs quite good in prediction:

![1990_primeira_liga_result](https://github.com/frankie-2nfro-com/football_match_outcome_prediction/blob/main/reports/primeira_liga_1990.png?raw=true)
![1995_primeira_liga_result](https://github.com/frankie-2nfro-com/football_match_outcome_prediction/blob/main/reports/primeira_liga_1995.png?raw=true)
![2000_primeira_liga_result](https://github.com/frankie-2nfro-com/football_match_outcome_prediction/blob/main/reports/primeira_liga_2000.png?raw=true)
![2005_primeira_liga_result](https://github.com/frankie-2nfro-com/football_match_outcome_prediction/blob/main/reports/primeira_liga_2005.png?raw=true)
![2010_primeira_liga_result](https://github.com/frankie-2nfro-com/football_match_outcome_prediction/blob/main/reports/primeira_liga_2010.png?raw=true)
![2015_primeira_liga_result](https://github.com/frankie-2nfro-com/football_match_outcome_prediction/blob/main/reports/primeira_liga_2015.png?raw=true)

However, some leagues perform badly like premier league. The complete result for each league and season dataset be found in [predict_result.txt](https://github.com/frankie-2nfro-com/football_match_outcome_prediction/blob/main/reports/predict_result.txt)

I think the features of the data is not rich enough to determine the winning of the football game. To improve the models, I think I need to spend more time to get more useful features as the new inputs. After that, it would be worth to tune the hyperparameters and trying different subset of data. 

In this project, I go through the process to implement a data science pipeline that predicts the outcome of a football match. Although the outcome is not good enough as the final product quality, I think it is a good experience for me to work as a data scientist. 

The complete code for this task can be found in [model_results_conclusion.ipynb](https://github.com/frankie-2nfro-com/football_match_outcome_prediction/blob/main/model_results_conclusion.ipynb)

(Go back over your entire documentation to make sure everything is clear, concise and reads well)

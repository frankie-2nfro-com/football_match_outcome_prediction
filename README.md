# AiCore Project #3 - Football Match Outcome Prediction

## Milestone 1 - EDA and Data Cleaning
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
  

## Milestone 2 - Feature Engineering
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
  
## Milestone 3 - Upload the data to the database

...

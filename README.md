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
self.leagues = [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]

# loop to open csv
whole_list_df = pd.DataFrame()
for league in self.leagues:
    league_folder = os.path.join(dir, league)
    csv_file_for_league = [os.path.join(league_folder, name) for name in os.listdir(league_folder) if name.endswith('.csv')]
    league_pds = [pd.read_csv(csv_filename, skiprows=[0], names=["Home_Team", "Away_Team", "Result", "Link", "Season", "Round", "League"]) for csv_filename in csv_file_for_league]
    whole_list_df = pd.concat(league_pds)
    self.data = pd.concat([self.data, whole_list_df])
```

#### 2) Match_Info.csv
In this csv file, it contains fields as follows: 
[Link], [Date_New], [Referee], [Home_Yellow], [Home_Red], [Away_Yellow], and [Away_Red]

#### 3) Team_Info.csv
In this csv file, it contains fields as follows: 
[Team], [City], [Country], [Stadium], [Capacity], [Pitch]

### Peek at the data


### Problem found at the data

### Data cleaning and refine

### Prelimilary direction

### Hypothesis Testing




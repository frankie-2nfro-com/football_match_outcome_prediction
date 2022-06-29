import os
import pandas as pd
import numpy as np
from csv import reader
import plotly.express as px
import missingno as msno

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
        # load all directory as league name list
        self.leagues = [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]

        # loop to open csv
        for league in self.leagues:
            league_folder = os.path.join(dir, league)
            csv_file_for_league = [os.path.join(league_folder, name) for name in os.listdir(league_folder) if name.endswith('.csv')]
            league_pds = [pd.read_csv(csv_filename, skiprows=[0], names=["Home_Team", "Away_Team", "Result", "Link", "Season", "Round", "League"]) for csv_filename in csv_file_for_league]
            whole_list_df = pd.concat(league_pds)
            self.data = pd.concat([self.data, whole_list_df])

        # Divide result into home_score and away_score
        df_score =  self.data['Result'].str.extract(r'(\d)-(\d)')
        self.data.insert(loc=3, column="Home_Score", value=df_score[0].astype('Int64'))
        self.data.insert(loc=4, column="Away_Score", value=df_score[1].astype('Int64')) 

        # calculate score difference
        df_h_a = self.data["Home_Score"] - self.data["Away_Score"]
        df_h_a.isna().sum()

        # add home_win, away_win and draw column
        hw_array = []
        aw_array = []
        d_array = []
        for minus_score in df_h_a:
            if pd.isna(minus_score):
                #print(minus_score, type(minus_score))
                hw_array.append(pd.NA)
                aw_array.append(pd.NA)
                d_array.append(pd.NA)
            else:
                if minus_score>0:
                    #print("HOME WIN")
                    hw_array.append(True)
                    aw_array.append(False)
                    d_array.append(False)
                elif minus_score<0:
                    #print("AWAY WIN")
                    hw_array.append(False)
                    aw_array.append(True)
                    d_array.append(False)
                else:
                    #print("DRAW")
                    hw_array.append(False)
                    aw_array.append(False)
                    d_array.append(True)

        self.data.insert(loc=5, column="Home_Win", value=hw_array) 
        self.data.insert(loc=6, column="Away_Win", value=aw_array) 
        self.data.insert(loc=7, column="Draw", value=d_array)

        # print statistic
        print(self.data.info())


    def getLeagueSeasonDataFrame(self, league, season):
        return self.data[(self.data["League"]==league) & (self.data["Season"]==season)]

    
    def getLeagueSeasonList(self, league):
        df = self.data[self.data["League"]==league]
        return df["Season"].sort_values().unique()


    def getLeagueSeasonTeamDataFrame(self, league, season, team):
        return self.data[(self.data["League"]==league) & ((self.data["Home_Team"]==team) | (self.data["Away_Team"]==team)) & (self.data["Season"]==season)]


    def getLeagueTeamCount(self, league, season):
        league_season_df = self.getLeagueSeasonDataFrame(league, season)
        return len(league_season_df["Home_Team"].sort_values().unique())


    def getLeagueTeamResult(self, league, team, season):
        league_season_df = self.getLeagueSeasonDataFrame(league, season)
        return len(league_season_df["Home_Team"].sort_values().unique())


    def getTeamCount(self, season):
        league_season_df = self.data[(self.data["Season"]==season)]
        return len(league_season_df["Home_Team"].sort_values().unique())


    def getLeagueSeasonHomeWinCount(self, league, season):
        return len(self.data[(self.data["Home_Win"]==True) & (self.data["League"]==league) & (self.data["Season"]==season)])

    
    def getLeagueSeasonHomeWinCount(self, league, season):
        return len(self.data[(self.data["Home_Win"]==True) & (self.data["League"]==league) & (self.data["Season"]==season)])


    def getResult(league, season):
        print("Try to get result records of [" + league + "] for season " + str(season))


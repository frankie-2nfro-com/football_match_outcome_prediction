import re
import pandas as pd
import os
import numpy as np
from csv import reader
import plotly.express as px
import missingno as msno
import pickle

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
					current_league_season_pd.insert(loc=5, column="HOME_TOTAL_GOAL_SO_FAR", value=home_away_total_goal_sofar_pd["HOME_GOAL_SO_FAR"]) 
					current_league_season_pd.insert(loc=6, column="AWAY_TOTAL_GOAL_SO_FAR", value=home_away_total_goal_sofar_pd["AWAY_GOAL_SO_FAR"])     

					# merge with ELO
					result_elo_pd = current_league_season_pd['Link'].apply(self.fillWithELO)   
					elo_list = np.array(result_elo_pd.values.tolist())
					elo_df = pd.DataFrame(elo_list, columns=["ELO_HOME", "ELO_AWAY"])
					current_league_season_pd.insert(loc=8, column="ELO_HOME", value=elo_df["ELO_HOME"]) 
					current_league_season_pd.insert(loc=9, column="ELO_AWAY", value=elo_df["ELO_AWAY"]) 

					# get recent performance
					home_away_recent_perf = current_league_season_pd.apply(self.fillWithRecentPerformance, data=current_league_season_pd, axis=1)
					perf_list = np.array(home_away_recent_perf.values.tolist())
					home_away_perf_pd = pd.DataFrame(perf_list, columns=["HOME_LAST_6_GOAL_DIFF", "AWAY_LAST_6_GOAL_DIFF"])
					current_league_season_pd.insert(loc=7, column="HOME_LAST_6_GOAL_DIFF", value=home_away_perf_pd["HOME_LAST_6_GOAL_DIFF"]) 
					current_league_season_pd.insert(loc=8, column="AWAY_LAST_6_GOAL_DIFF", value=home_away_perf_pd["AWAY_LAST_6_GOAL_DIFF"]) 

					result_with_goal_sofar_pd = pd.concat([result_with_goal_sofar_pd, current_league_season_pd])

		# export to csv
		result_with_goal_sofar_pd.to_csv('cleaned_dataset.csv', index=False)

		self.data = result_with_goal_sofar_pd


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
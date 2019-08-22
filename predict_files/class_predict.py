import pandas as pd
import numpy as np

from logreg_files.constants import FEATURES, HOUSES


class Predict:

	def __init__(self, df, data):
		self.df = df
		self.theta_norm = data['theta_norm']
		self.mean_serie = data['mean_serie']
		self.std_serie = data['std_serie']
		self.df_testing = self.ft_formatting()
		self.df_testing_norm = pd.concat([self.df_testing[['Intercept'] + HOUSES], (self.df_testing[FEATURES] - self.mean_serie) / self.std_serie], axis=1)

	def ft_formatting(self):
		new_df = self.df.fillna(self.df.mean())
		new_df = pd.concat([new_df[new_df.columns.difference(FEATURES)],
							pd.DataFrame({'Intercept': np.ones(new_df.shape[0])}),
							new_df.loc[:,FEATURES]], axis=1)
		new_df.drop(labels=new_df.columns.difference(FEATURES + ['Intercept']), axis=1, inplace=True)
		for h in HOUSES:
			new_df[h] = 0
		return new_df

	@staticmethod
	def sigmoid(z):
		return 1 / (1 + np.exp(-z))

	def predict(self):
		for h in HOUSES:
			theta = self.theta_norm[h]
			self.df_testing_norm[h] = self.sigmoid(self.df_testing_norm[['Intercept'] + FEATURES].dot(theta))

	def final_output(self):
		final_df = pd.DataFrame(index=self.df_testing_norm.index)
		final_df['Hogwarts House'] = self.df_testing_norm[HOUSES].idxmax(axis=1)
		final_df.to_csv('predict_files/houses.csv', index_label='Index')
		return final_df


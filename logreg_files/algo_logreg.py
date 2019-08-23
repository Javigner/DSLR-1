import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyautogui

from logreg_files.constants import FEATURES, HOUSES, RANGE_LBDA, CV_BATCH_SIZE, CV_ITER, CV_NUMBER_LBDA
from logreg_files.class_plot import GraphLive


class Algo:

	flag_plot = False
	reg_method = None

	def __init__(self, df, lbda):
		self.df = df
		self.lbda = lbda
		self.df_training = self.ft_formatting()
		self.theta_norm = self.ft_init_thetas()
		self.df_training_norm, self.mean_serie, self.std_serie = self.feature_normalization()
		self.J_history = pd.Series(data=np.full(len(HOUSES), 0, dtype=object), index=HOUSES)
		self.training_errors, self.testing_errors = None, None

	def ft_init_thetas(self):
		theta_array = np.zeros(self.df_training.loc[:,FEATURES].shape[1] + 1, dtype=float)
		theta_dict = {}
		for h in HOUSES:
			theta_dict[h] = theta_array
		theta_df = pd.DataFrame(data=theta_dict, index=['Intercept']+FEATURES)
		return theta_df

	def feature_normalization(self):
		mean_serie = self.df_training.loc[:,FEATURES].sum() / self.df_training.shape[0]
		std_serie = np.sqrt(np.sum((self.df_training.loc[:,FEATURES] - mean_serie)**2) / (self.df_training.shape[0] - 1))
		df_norm = self.df_training.copy(deep=True)
		df_norm.loc[:,FEATURES] = (df_norm.loc[:,FEATURES] - mean_serie) / std_serie
		return df_norm, mean_serie, std_serie

	def ft_formatting(self):

		def ft_fill_nan(data):
			list_df = []
			for h in HOUSES:
				list_df.append(data[data['Hogwarts House'] == h].fillna(data[data['Hogwarts House'] == h].mean()))
			new_df = pd.concat(list_df)
			return new_df

		def ft_output_formatting(df):
			for h in HOUSES:
				df[h] = 0
				df.loc[df['Hogwarts House'] == h, h] = 1
				df.loc[df['Hogwarts House'] != h, h] = 0
			return df

		new_df = ft_fill_nan(self.df)

		new_df = pd.concat([new_df[new_df.columns.difference(FEATURES)],
							pd.DataFrame({'Intercept': np.ones(new_df.shape[0])}),
							new_df.loc[:,FEATURES]], axis=1)
		new_df = ft_output_formatting(new_df)

		new_df.drop(labels=new_df.columns.difference(FEATURES + HOUSES + ['Intercept']), axis=1, inplace=True)

		return new_df

	@staticmethod
	def sigmoid(z):
		return 1 / (1 + np.exp(-z))

	def predict(self, X, theta):
		return self.sigmoid(X.dot(theta))

	def cost(self, X=None, y=None, theta=None, reg=None, lbda=1):
		m = X.shape[0]
		diff = np.sum(np.dot(y, np.log(self.predict(X, theta)))+
					np.dot(1 - y, np.log(1 - self.predict(X, theta))))
		cost_init = (- 1 / m) * diff
		if reg == "ridge":
			regul = (lbda / (2 * m)) * np.sum(np.square(theta[1:]))
		elif reg == "lasso":
			regul = (lbda / m) * np.sum(np.abs(theta[1:]))
		else:
			regul = 0
		return cost_init + regul

	def gradient_descent(self, X, y, theta, alpha=1, iter=150, lbda=1):
		run = 0
		m = X.shape[0]
		J_history = [self.cost(X, y, theta, lbda, Algo.reg_method)]
		regul = pd.Series(data=np.zeros(X.loc[:, FEATURES].shape[1] + 1, dtype=float), index=['Intercept'] + FEATURES)
		for i in range(iter):
			if Algo.reg_method == "ridge":
				regul[1:] = lbda * theta[1:]
			elif Algo.reg_method == "lasso":
				run += 1
				regul[1:][theta[1:] == 0] = 0
				regul[1:][theta[1:] > 0] = 1
				regul[1:][theta[1:] < 0] = -1
				regul[1:] = lbda * regul[1:]
			diff = np.dot(self.predict(X, theta) - y, X) + regul
			if Algo.reg_method == "lasso" and run > 1:
				theta[theta != 0] = theta - (alpha / m) * diff
			else:
				theta = theta - (alpha / m) * diff
			J_history.append(self.cost(X, y, theta, lbda, Algo.reg_method))
		return theta, tuple(J_history)

	def fit_logreg(self, alpha=1, iter=150):
		X = self.df_training_norm[['Intercept'] + FEATURES]
		for h in HOUSES:
			y = self.df_training_norm[h]
			theta = self.theta_norm[h]
			self.theta_norm[h], self.J_history[h] = self.gradient_descent(X, y, theta, alpha=alpha, iter=iter, lbda=self.lbda)
		return None

	def ft_graph_cverrors(self):
		data_errors = self.ft_read_cv_pickle()
		fig, ax = plt.subplots(2, 2, sharex='col', figsize=(15,12))
		for h,i in zip(HOUSES, range(4)):
			l1 = ax[i // 2, i % 2].scatter(data_errors['training_errors'].index, data_errors['training_errors'][h])
			l2 = ax[i // 2, i % 2].scatter(data_errors['testing_errors'].index, data_errors['testing_errors'][h], c='blue')
			ax[i // 2, i % 2].set_title(h)
		ax[1,0].set_xlabel('Regularization parameter - lambda')
		ax[1, 0].set_ylabel('Errors')
		fig.suptitle('Plot of testing and training errors according to lambda, per houses', fontsize=16)
		fig.legend((l1,l2), ('training errors', 'testing errors'), loc='upper left')
		plt.show()
		return None

	def ft_cross_validation(self, alpha=1, iter=150):

		def ft_init_costs(lbda_array):
			cost_array = np.zeros(lbda_array.size, dtype=float)
			cost_dict = {}
			for h in HOUSES:
				cost_dict[h] = cost_array
			cost_df = pd.DataFrame(data=cost_dict, index=lbda_array)
			return cost_df

		def get_test_sample(df):
			m = df.shape[0]
			start = 0
			step = int(m * CV_BATCH_SIZE)
			end = step - 1
			while end < m:
				yield df.iloc[start:end,:]
				start += step
				end += step

		lbda_array = (RANGE_LBDA[1] - RANGE_LBDA[0]) * np.random.rand(CV_NUMBER_LBDA) + RANGE_LBDA[0]

		training_errors = ft_init_costs(lbda_array)
		testing_errors = ft_init_costs(lbda_array)
		numero_lbda = 1
		for l in lbda_array :
			print(f"Numero de lambda : {numero_lbda} sur un total de {CV_NUMBER_LBDA}")
			numero_lbda += 1
			dic_train_errors = {}
			dic_test_errors = {}
			for h in HOUSES:
				dic_train_errors[h] = []
				dic_test_errors[h] = []
			for epoch in range(CV_ITER):
				df_cv = self.df_training_norm.sample(frac=1)
				for test_sample in get_test_sample(df_cv):
					train_sample = df_cv[~df_cv.index.isin(test_sample.index)]
					X_train = train_sample[['Intercept'] + FEATURES]
					X_test = test_sample[['Intercept'] + FEATURES]
					thetas = self.theta_norm.copy()
					J = self.J_history.copy()
					for h in HOUSES:
						y_train = train_sample[h]
						y_test = test_sample[h]
						theta = thetas[h]
						thetas[h], J[h] = self.gradient_descent(X_train, y_train, theta, alpha=alpha, iter=iter, lbda=l)
						dic_train_errors[h].append(self.cost(X_train, y_train, theta, l))
						dic_test_errors[h].append(self.cost(X_test, y_test, theta, l))

			for h in HOUSES:
				training_errors.loc[l,h] = sum(dic_train_errors[h]) / len(dic_train_errors[h])
				testing_errors.loc[l,h] = sum(dic_test_errors[h]) / len(dic_test_errors[h])

		self.training_errors = training_errors
		self.testing_errors = testing_errors
		print(f"Training errors : \n{training_errors}\n")
		print(f"Testing errors : \n{testing_errors}\n")

		self.ft_dump_cv_pickle()

		return None

	def ft_find_best(self, alpha, iter):
		data_errors = self.ft_read_cv_pickle()
		mse = data_errors['training_errors'] + data_errors['testing_errors']
		best_lambda = mse.min(axis=1).idxmin()
		self.lbda = best_lambda
		self.fit_logreg(alpha, iter)
		return None

	def ft_graph_interactive(self, alpha, iter):

		def init_graphs():
			fig = plt.figure(figsize=(pyautogui.size()[0] / 96, pyautogui.size()[1] / 96))
			graphs_cost = {}
			for i, h in enumerate(HOUSES):
				graphs_cost[h] = GraphLive(x_vec=np.arange(0, iter), y_vec=np.full(iter, np.nan),
										   ax=fig.add_subplot(2, 2, i + 1),
										   title=f"Cost evolution for {h} classifier",
										   x_label="Iterations", y_label="J_history")
			return graphs_cost

		graphs_cost = init_graphs()
		for i in range(iter):
			for h in HOUSES:
				graphs_cost[h].y_vec[i] = self.J_history[h][i]
				graphs_cost[h].live_line_evolution(y_limit=(0, self.J_history[h][0]), x_limit=(0, iter))

		GraphLive.close(alpha, iter)

	def ft_dump_data_pickle(self):
		data = {
			'theta_norm': self.theta_norm,
			'mean_serie': self.mean_serie,
			'std_serie': self.std_serie,
			'lambda': self.lbda,
		}
		with open('logreg_files/data.pkl', 'wb') as f:
			pickle.dump(data, f)

	def ft_dump_cv_pickle(self):
		data_errors = {'training_errors': self.training_errors, 'testing_errors': self.testing_errors}
		with open('logreg_files/cv_errors.pkl', 'wb') as f:
			pickle.dump(data_errors, f)

	def ft_read_cv_pickle(self):
		with open('logreg_files/cv_errors.pkl', 'rb') as f:
			return pickle.load(f)




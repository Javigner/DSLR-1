import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyautogui

from constants import FEATURES, HOUSES, RANGE_LBDA, CV_BATCH_SIZE, CV_ITER, CV_NUMBER_LBDA
from class_plot import GraphLive


class Algo:

	flag_plot = False
	reg_method = None
	gd_algo = None

	def __init__(self, df):
		self.df = df
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
		return(theta_df)

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

	def cost(self, X=None, y=None, theta=None, lbda=0.01, reg=None):
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

	def gradient_descent(self, X, y, theta, alpha=1, iter=150, lbda=0.01):
		m = X.shape[0]
		J_history = [self.cost(X, y, theta, lbda, Algo.reg_method)]
		regul = pd.Series(data=np.zeros(X.loc[:, FEATURES].shape[1] + 1, dtype=float),
						  index=['Intercept'] + FEATURES)
		for i in range(iter):
			if Algo.reg_method == "ridge":
				regul[1:] = lbda * theta[1:]
			elif Algo.reg_method == "lasso":
				regul[1:] = lbda * regul[1:].where(theta[1:] != 0, other=0)
				regul[1:] = lbda * regul[1:].where(theta[1:] > 0, other=-1)
				regul[1:] = lbda * regul[1:].where(theta[1:] < 0, other=1)
			diff = np.dot(self.predict(X, theta) - y, X) + regul
			theta = theta - (alpha / m) * diff
			J_history.append(self.cost(X, y, theta, lbda, Algo.reg_method))
		return theta, tuple(J_history)

	def fit_logreg(self, alpha=1, iter=150, lbda=0.1):
		X = self.df_training_norm[['Intercept'] + FEATURES]
		for h in HOUSES:
			y = self.df_training_norm[h]
			theta = self.theta_norm[h]
			self.theta_norm[h], self.J_history[h] = self.gradient_descent(X, y, theta, alpha=alpha, iter=iter, lbda=lbda)
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

		# dataframe qui contiendra les trainign errors pour chaque maison (col) pour chaque lambda (index)
		training_errors = ft_init_costs(lbda_array)

		# dataframe qui contiendra les testing errors pour chaque maison (col) pour chaque lambda (index)
		testing_errors = ft_init_costs(lbda_array)

		#boucle sur chaque lambda (10) genere randomly
		numero_lbda = 1
		for l in lbda_array :
			print(f"Numero de lambda : {numero_lbda} sur un total de {CV_NUMBER_LBDA}")
			numero_lbda += 1
			# Initialisation d'un dictionnaire qui contiendra pour chaque house une liste
			# avec le dernier cost pour chaque epoch (training and test : on fera ensuite l'average)
			dic_train_errors = {}
			dic_test_errors = {}
			for h in HOUSES:
				dic_train_errors[h] = []
				dic_test_errors[h] = []
			#On va faire 10 training sur 10 df aleatoires par lambda
			for epoch in range(CV_ITER):
				# random shuffle du dataframe
				df_cv = self.df_training_norm.sample(frac=1)
				# Tant que le generateur renvoie des elements, on train
				for test_sample in get_test_sample(df_cv):
					# Le train sample et le df_cv auquel on exclut les index du test df
					train_sample = df_cv[~df_cv.index.isin(test_sample.index)]
					#on separe nos features de nos outputs (y)
					X_train = train_sample[['Intercept'] + FEATURES]
					X_test = test_sample[['Intercept'] + FEATURES]
					#On initialise les thetas a 0
					thetas = self.theta_norm.copy()
					#idem pour le cout on chope la serie des couts avec pour indix les maisons
					J = self.J_history.copy()
					# On fait tourner nos 4 classifiers sur nos 4 maisons
					for h in HOUSES:
						y_train = train_sample[h]
						y_test = test_sample[h]
						theta = thetas[h]
						thetas[h], J[h] = self.gradient_descent(X_train, y_train, theta, alpha=alpha, iter=iter, lbda=l)
						dic_train_errors[h].append(self.cost(X_train, y_train, theta, l))
						#Ici on teste de suite notre model (one-vs-all) pour enregistrer le cout associe
						dic_test_errors[h].append(self.cost(X_test, y_test, theta, l))

			for h in HOUSES:
				training_errors.loc[l,h] = sum(dic_train_errors[h]) / len(dic_train_errors[h])
				testing_errors.loc[l,h] = sum(dic_test_errors[h]) / len(dic_test_errors[h])

		self.training_errors = training_errors
		self.testing_errors = testing_errors

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
		data = {'theta_norm': self.theta_norm, 'mean_serie': self.mean_serie, 'std_serie': self.std_serie}
		with open('data.pkl', 'wb') as f:
			pickle.dump(data, f)

	def ft_dump_cv_pickle(self):
		data_errors = {'taining_errors': self.training_errors, 'testing_errors': self.testing_errors}
		with open('cv_errors.pkl', 'wb') as f:
			pickle.dump(data_errors, f)

	#
	# def cost_sgd(self, theta, e=0):
	# 	return np.square(self.predict(self.X_norm[e], theta) - self.y[e]) / 2
	#
	# def cost_mbgd(self, X, y, theta):
	# 	return np.sum(np.square(self.predict(X, theta) - y)) / (2 * X.size)
	#
	# def dynamic_plots(self, iter, i, **kwargs):
	# 	self.theta[0] = self.theta_norm[0] - self.theta_norm[1] * self.mean_ / self.range_
	# 	self.theta[1] = self.theta_norm[1] / self.range_
	#
	# 	if Algo.gd_algo == "SGD":
	# 		kwargs['g1'].y_vec[i] = np.sum(np.array([self.J_history[-a] for a in range(1, Algo.mv_avg + 1)])) / Algo.mv_avg
	# 	else:
	# 		kwargs['g1'].y_vec[i] = self.J_history[-1]
	# 	kwargs['g1'].live_line_evolution(y_limit=(0, self.J_history[0]), x_limit=(0, iter))
	# 	kwargs['g2'].live_regression(y_limit=(0, 1.2 * np.max(self.y, axis=0)),
	# 								 x_limit=(0, 1.2 * np.max(self.X[:, 1], axis=0)),
	# 								 theta=self.theta, true_theta=self.true_theta)
	# 	kwargs['g3'].draw_contour(self.cost_bgd, theta=self.theta_norm)
	#
	# 	return None
	#
	# def batch_gradient(self, alpha, iter, m, **kwargs):
	# 	for i in range(iter):
	# 		diff = np.dot(self.predict(self.X_norm, self.theta_norm) - self.y, self.X_norm)
	# 		self.theta_norm = self.theta_norm - (alpha / m) * diff
	# 		self.J_history.append(self.cost_bgd(self.theta_norm))
	#
	# 		if Algo.flag_plot:
	# 			self.dynamic_plots(iter, i, **kwargs)
	#
	# 	return None
	#
	# def stochastic_gradient(self, alpha, iter, m, **kwargs):
	#
	# 	def learning_rate_decay(epoch):
	# 		return 1/ (3 + epoch)
	#
	# 	index_array = np.arange(m)
	# 	run = 0
	# 	for i in range(iter):
	# 		np.random.shuffle(index_array)
	# 		for e in index_array:
	# 			run += 1
	# 			diff = np.dot(self.predict(self.X_norm[e], self.theta_norm) - self.y[e], self.X_norm[e])
	# 			self.J_history.append(self.cost_sgd(self.theta_norm, e))
	# 			self.theta_norm = self.theta_norm - alpha * diff
	# 		alpha = learning_rate_decay(i)
	#
	# 		if Algo.flag_plot :
	# 			self.dynamic_plots(iter, i, **kwargs)
	# 	return None
	#
	# def mb_gradient(self, alpha, iter, m, **kwargs):
	#
	# 	def ft_shuffle_data():
	# 		concat_array = np.c_[self.X_norm, self.y]
	# 		np.random.shuffle(concat_array)
	# 		self.X_norm = concat_array[:, :-1]
	# 		self.y = concat_array[:, -1]
	#
	# 	def ft_get_batch(X, y, b=Algo.batch_size):
	# 		if b > m:
	# 			linear_regressioin.ft_errors("Batch size cannot exceed number of training examples.")
	# 		start = 0
	# 		batch_i = b
	# 		X_batch = []
	# 		y_batch = []
	# 		while batch_i <= m:
	# 			X_batch.append(X[start:batch_i,:])
	# 			y_batch.append(y[start:batch_i])
	# 			batch_i += b
	# 			start += b
	# 		if start < m:
	# 			X_batch.append(X[start:,:])
	# 			y_batch.append(y[start:])
	# 		for x,y in zip(X_batch, y_batch):
	# 			yield x, y
	#
	# 	for i in range(iter):
	# 		ft_shuffle_data()
	# 		for X_batch, y_batch in ft_get_batch(self.X_norm, self.y):
	# 			diff = np.dot(self.predict(X_batch, self.theta_norm) - y_batch, X_batch)
	# 			self.J_history.append(self.cost_mbgd(X_batch, y_batch, self.theta_norm))
	# 			self.theta_norm = self.theta_norm - (alpha / X_batch.size) * diff
	#
	# 		if Algo.flag_plot:
	# 			self.dynamic_plots(iter, i, **kwargs)
	#
	# 	return None
	#
	# def fit_linear(self, alpha=1, iter=150):
	# 	m = self.X.shape[0]
	# 	g1, g2, g3 = None, None, None
	#
	# 	if Algo.flag_plot:
	# 		fig = plt.figure(figsize=(pyautogui.size()[0] / 96, pyautogui.size()[1] / 96))
	# 		g1 = GraphLive(x_vec=np.arange(0, iter), y_vec=np.full(iter, np.nan),
	# 					   ax=fig.add_subplot(221), title="Real time cost evolution",
	# 					   x_label="Iterations", y_label="J_history")
	# 		g2 = GraphLive(x_vec=self.X[:,1], y_vec=self.y, ax=fig.add_subplot(222),
	# 					   title="Regression line", x_label="Mileage", y_label="Price")
	# 		g3 = GraphLive(x_vec=np.arange(3000, 10000, 50),
	# 					   y_vec=np.arange(-15000, 10000, 50), ax=fig.add_subplot(212),
	# 					   title="Gradient descent", x_label="theta0", y_label="theta1")
	#
	# 	if Algo.gd_algo == "BGD":
	# 		self.batch_gradient(alpha, iter, m, g1=g1, g2=g2, g3=g3)
	# 	elif Algo.gd_algo == "SGD":
	# 		self.stochastic_gradient(alpha, iter, m, g1=g1, g2=g2, g3=g3)
	# 	elif Algo.gd_algo == "MBGD":
	# 		self.mb_gradient(alpha, iter, m, g1=g1, g2=g2, g3=g3)
	#
	# 	self.theta[0] = self.theta_norm[0] - self.theta_norm[1] * self.mean_ / self.range_
	# 	self.theta[1] = self.theta_norm[1] / self.range_
	# 	if Algo.flag_plot:
	# 		GraphLive.close()
	#
	# 	return None



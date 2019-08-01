import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyautogui

from constants import FEATURES, HOUSES
# from class_plot import GraphLive


class Algo:

	flag_plot = False
	lbda = None
	reg_method = None
	gd_algo = ''
	mv_avg = 10
	batch_size = 10

	def __init__(self, df):
		self.df = df
		self.df_training = self.ft_formatting()
		self.theta_norm = np.zeros(self.df_training.loc[:,FEATURES].shape[1] + 1, dtype=float)
		self.theta = np.zeros(self.df_training.loc[:,FEATURES].shape[1] + 1, dtype=float)
		self.df_training_norm, self.mean_serie, self.std_serie = self.feature_normalization()
		# self.J_history = []

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

		return new_df

	@staticmethod
	def sigmoid(z):
		return 1 / (1 + np.exp(-z))

	def predict(self, X, theta):
		return self.sigmoid(np.dot(X, theta))

	def cost(self, X=None, y=None, theta=None):
		m = X.shape[0]
		diff = np.sum(np.dot(y, np.log(self.predict(X, theta)))+
					np.dot(1 - y, np.log(1 - self.predict(X, theta))))
		cost_init = (- 1 / m) * diff
		if Algo.reg_method == "ridge":
			regul = (Algo.lbda / (2 * m)) * np.sum(np.square(theta))
		elif Algo.reg_method == "lasso":
			regul = (Algo.lbda / m) * np.sum(np.abs(theta))
		else:
			regul = None
		return cost_init + regul
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



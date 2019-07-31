
import math
import pandas as pd
import numpy as np

class Describe:

	def __init__(self, df):
		self.df_in = df.set_index(['Index'])
		self.df_out_index = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
		self.df_out = pd.DataFrame(self.df_out_index, columns=['metrics'])


	def ft_describe(self):


		def ft_count(array):
			c = 0
			for e in np.nditer(array):
				if not np.isnan(e):
					c += 1
			return c

		def ft_mean(array, c):
			sum = 0
			for e in np.nditer(array):
				if not np.isnan(e):
					sum += e
			return sum / c

		def ft_std(array, c, mu):
			sum_diff = 0
			for e in array:
				if not np.isnan(e):
					sum_diff += (e - mu) ** 2
			return math.sqrt(sum_diff / (c - 1))

		def ft_min(array):
			min  = array[0]
			for e in array:
				if e < min:
					min = e
			return min

		def percentile(array, percent):
			if not array.any():
				return None
			k = (array.size - 1) * percent
			f = math.floor(k)
			c = math.ceil(k)
			if f == c:
				return array[int(k)]
			d0 = array[int(f)] * (c - k)
			d1 = array[int(c)] * (k - f)
			return d0 + d1

		def ft_first_quartile(array):
			return percentile(np.sort(array[~np.isnan(array)]), 0.25)

		def ft_median(array):
			return percentile(np.sort(array[~np.isnan(array)]), 0.5)

		def ft_third_quartile(array):
			return percentile(np.sort(array[~np.isnan(array)]), 0.75)

		def ft_max(array):
			max = array[0]
			for e in array:
				if e > max:
					max = e
			return max


		dic_functions = {
			'min': ft_min,
			'25%': ft_first_quartile,
			'50%': ft_median,
			'75%': ft_third_quartile,
			'max': ft_max,
		}

		for col in self.df_in.columns:
			dic_results = {}
			if self.df_in[col].dtype == np.float64:
				dic_results['count'] = ft_count(self.df_in[col].values)
				dic_results['mean'] = ft_mean(self.df_in[col].values, dic_results['count'])
				dic_results['std'] = ft_std(self.df_in[col].values, dic_results['count'], dic_results['mean'])
				for k, func in dic_functions.items():
					dic_results[k] = func(self.df_in[col].values)
				self.df_out[col] = self.df_out['metrics'].map(dic_results)
		self.df_out = self.df_out.set_index('metrics')
		del self.df_out.index.name


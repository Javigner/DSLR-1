#!/usr/bin/env python3

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def ft_argparser():
	parser = argparse.ArgumentParser()
	parser.add_argument("data_file", type=str, help="csv file containing the data to analyze")
	parser.add_argument("-k", "--kde",  action="store_true", help="Plot a kernel density function around histograms for better analysis")
	args = parser.parse_args()
	return args


def draw_histogram(df, kde):
	num_ln = 4
	num_col = 4
	plt.style.use('ggplot')
	plt.rc('legend', fontsize=12)
	fig, axs = plt.subplots(num_ln, num_col, tight_layout=True, figsize=(25,13))
	i = 0
	j = 1
	colors = ['#00006d', '#00613e', "#ae0001", '#f0c75e']
	for col in df.columns:
		if df[col].dtype == np.float64:
			j = 0 if j == num_col else j
			for house, c in zip(df['Hogwarts House'].unique().tolist(), colors):
				serie = df[df['Hogwarts House'] == house][col].dropna()
				g = sns.distplot(serie, kde=kde, hist=True, ax=axs[i,j], color=c, axlabel=False, label=house)
			axs[i, j].set_title(col)
			if col == "Care of Magical Creatures":
				for side in ["right", "left", "top", "bottom"]:
					axs[i,j].spines[side].set_color('red')
					axs[i,j].spines[side].set_linewidth(1.2)
			i = i + 1 if j == num_col - 1 else i
			j += 1
	while j < num_col:
		axs[i, j].axis('off')
		j += 1
	axs[0, 0].axis('off')
	axs[0,1].set_ylabel("Frequency of students")
	axs[0,1].set_xlabel("Grades")
	handles, labels = axs[3,0].get_legend_handles_labels()
	axs[0, 0].legend(handles, labels, loc="lower center", borderpad=1.5, labelspacing=1.25)
	axs[0,0].set_title('Students repartition by subject', fontsize=20, fontweight='bold')
	if not os.path.exists('./plots'):
		os.makedirs('./plots')
	plt.savefig('plots/histogram.png')
	plt.show()
	return None


def main(args):

	try:
		df = pd.read_csv(args.data_file)
	except Exception as e:
		print(f"Please specify a correct file.\n{e}")
		sys.exit(0)
	df = df.drop(labels='Index', axis=1)
	draw_histogram(df, args.kde)
	return None


if __name__ == "__main__":
	args = ft_argparser()
	main(args)


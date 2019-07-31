#!/usr/bin/env python3

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ft_argparser():

	parser = argparse.ArgumentParser()
	parser.add_argument("data_file", type=str, help="csv file containing the data to analyze")
	args = parser.parse_args()
	return args


def draw_histogram(df):
	plt.style.use('ggplot')
	fig, axs = plt.subplots(4, 4, tight_layout=True, figsize=(25,13))
	i = 0
	j = 0
	colors = ['#00006d', '#00613e', "#ae0001", '#f0c75e']
	for col in df.columns:
		if df[col].dtype == np.float64:
			j = 0 if j == 4 else j
			list_df = []
			labels = []
			for house in df['Hogwarts House'].unique().tolist():
				list_df.append(df[df['Hogwarts House'] == house][col])
				labels.append(house)
			axs[i, j].hist(list_df, label=labels, color=colors)
			i = i + 1 if j == 3 else i
			j += 1
	for col_graph in range(1,4):
		axs[3, col_graph].axis('off')
	handles, labels = axs[3,0].get_legend_handles_labels()
	fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.15), bbox_transform=fig.transFigure, prop={'size': 12}, borderpad=2)
	plt.show()


def main(args):

	try:
		df = pd.read_csv(args.data_file)
	except Exception as e:
		print(f"Please specify a correct file.\n{e}")
		sys.exit(0)
	df = df.drop(labels='Index', axis=1)
	draw_histogram(df)
	print(df.head())
	return None


if __name__ == "__main__":
	args = ft_argparser()
	main(args)


#!/usr/bin/env python3

import sys
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def ft_argparser():

	parser = argparse.ArgumentParser()
	parser.add_argument("data_file", type=str, help="csv file containing the data to analyze")
	args = parser.parse_args()
	return args


def draw_scatterplot(df):
	plt.style.use('ggplot')
	fig = plt.figure(num='Similar features', figsize=(13,13))
	ax = plt.subplot(111)
	colors = {'Ravenclaw': '#00006d', 'Slytherin': '#00613e', 'Gryffindor': "#ae0001", 'Hufflepuff': '#f0c75e'}
	sns.scatterplot(x="Care of Magical Creatures", y="Arithmancy", hue='Hogwarts House', palette=colors, data=df, ax=ax)
	ax.legend(loc="upper right")
	if not os.path.exists('./plots'):
		os.makedirs('./plots')
	plt.savefig('plots/scatter_plot.png')
	plt.show()
	return None


def ft_fill_nan(df):
	houses = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
	list_df = []
	for h in houses:
		list_df.append(df[df['Hogwarts House'] == h].fillna(df[df['Hogwarts House'] == h].mean()))
	return pd.concat(list_df)


def main(args):

	try:
		df = pd.read_csv(args.data_file)
	except Exception as e:
		print(f"Please specify a correct file.\n{e}")
		sys.exit(0)
	df = df.drop(labels='Index', axis=1)
	df = ft_fill_nan(df)
	draw_scatterplot(df)
	return None


if __name__ == "__main__":
	args = ft_argparser()
	main(args)
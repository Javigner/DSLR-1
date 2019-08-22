#!/usr/bin/env python3

import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def ft_argparser():
	parser = argparse.ArgumentParser()
	parser.add_argument("data_file", type=str, help="csv file containing the data to analyze")
	args = parser.parse_args()
	return args


def draw_pairplot(df):
	plt.style.use('ggplot')
	colors = {'Ravenclaw': '#00006d', 'Slytherin': '#00613e', 'Gryffindor': "#ae0001", 'Hufflepuff': '#f0c75e'}
	g = sns.pairplot(df, hue='Hogwarts House', palette=colors)
	g._legend.remove()
	plt.tight_layout()
	plt.savefig('plots/pair_plot.png')
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
	draw_pairplot(df)
	return None


if __name__ == "__main__":
	args = ft_argparser()
	main(args)
#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import argparse

from algo_logreg import Algo
from constants import FEATURES

def ft_argparser():

	parser = argparse.ArgumentParser()
	parser.add_argument("data_file", type=str, help="csv file containing the data to analyze")
	parser.add_argument("-reg", "--regularization", type=str, default="ridge", choices=["ridge", "lasso", None], help="Regularization method")
	parser.add_argument("-l", "--lambda_", type=float, default=1, help="Fix regularization parameter.")
	args = parser.parse_args()
	return args


def main(args):
	try:
		df = pd.read_csv(args.data_file)
	except Exception as e:
		print(f"Please specify a correct file.\n{e}")
		sys.exit(0)
	df = df.drop(labels='Index', axis=1)
	Algo.lbda = args.lambda_
	Algo.reg_method = args.regularization
	algo = Algo(df)
	print(algo.df.head())
	print(algo.df_training.head())
	print(algo.df_training_norm.head())
	print(algo.cost(X=algo.df_training_norm[['Intercept'] + FEATURES], y=algo.df_training_norm['Ravenclaw'], theta=algo.theta_norm))



if __name__ == "__main__":
	args = ft_argparser()
	main(args)
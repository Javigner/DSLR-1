#!/usr/bin/env python3

import sys
import time
import numpy as np
import pandas as pd
import argparse

from algo_logreg import Algo
from constants import FEATURES

def ft_argparser():

	parser = argparse.ArgumentParser()
	parser.add_argument("data_file", type=str, help="csv file containing the data to analyze")
	parser.add_argument("-reg", "--regularization", type=str, default="ridge", choices=["ridge", "lasso", None], help="Regularization method")
	parser.add_argument("-l", "--lambda_", type=float, default=0.1, help="Fix regularization parameter.")
	parser.add_argument("-m", "--method", type=str, default="BGD", choices=["BGD", "MBGD", "SGD"],
						help="Type of gradient descent algorithm: Batch GD, Mini-Batch GD or Stochastic GD")
	parser.add_argument("-i", "--iterations", type=int, default=150, help="fix number of iterations")
	parser.add_argument("-a", "--alpha", type=float, default=0.6, help="fix size of Gradient Descent step")
	parser.add_argument("-p", "--plot", action="store_true", help="Draw a plot of cost function as GD advances")
	args = parser.parse_args()
	return args


def main(args):
	try:
		df = pd.read_csv(args.data_file)
	except Exception as e:
		print(f"Please specify a correct file.\n{e}")
		sys.exit(0)
	df = df.drop(labels='Index', axis=1)
	Algo.reg_method = args.regularization
	Algo.gd_algo = args.method
	Algo.flag_plot = args.plot
	model = Algo(df)
	model.fit_logreg(alpha=args.alpha, iter=args.iterations, lbda=args.lambda_)
	if args.plot:
		model.ft_graph_interactive(args.alpha, args.iterations)
	start = time.time()
	model.ft_cross_validation(alpha=args.alpha, iter=args.iterations)
	end = time.time()
	print(f"time elapsed : {(end - start) / 60}")
	print(model.training_errors)
	print(model.testing_errors)
	model.ft_dump_cv_pickle()
	# print("theta avant enreigstrement\n", model.theta_norm)
	# model.ft_dump_data_pickle()


if __name__ == "__main__":
	args = ft_argparser()
	main(args)
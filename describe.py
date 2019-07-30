#!/usr/bin/env python3

import sys
import argparse
import pandas as pd

from df_analysis import Describe

def ft_argparser():

	parser = argparse.ArgumentParser()
	parser.add_argument("data_file", type=str, help="csv file containing the data to analyze")
	args = parser.parse_args()
	return args


def main(args):

	describe = Describe(pd.read_csv(args.data_file))
	describe.ft_describe()
	# print(describe.df_in.describe())
	print(describe.df_out)




if __name__ == "__main__":
	args = ft_argparser()
	main(args)
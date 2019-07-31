#!/usr/bin/env python3

import sys
import argparse
import pandas as pd

from describe_files.df_analysis import Describe


def ft_argparser():

	parser = argparse.ArgumentParser()
	parser.add_argument("data_file", type=str, help="csv file containing the data to analyze")
	args = parser.parse_args()
	return args


def main(args):

	try:
		describe = Describe(pd.read_csv(args.data_file))
	except Exception as e:
		print(f"Please specify a correct file.\n{e}")
		sys.exit(0)

	describe.ft_describe()
	print(describe.df_out)
	return None


if __name__ == "__main__":
	args = ft_argparser()
	main(args)
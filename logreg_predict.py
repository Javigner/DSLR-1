#!/usr/bin/env python3

import sys
import pandas as pd
import argparse
import pickle

from predict_files.class_predict import Predict


def ft_argparser():

	parser = argparse.ArgumentParser()
	parser.add_argument("prediction_file", type=str, help="csv file containing the data to work a prediction on")
	args = parser.parse_args()
	return args


def main(args):

	try:
		with open('logreg_files/data.pkl', 'rb') as f:
			data = pickle.load(f)
	except Exception as e:
		print(f"Please provide a valid data file where the parameters of your trained model have been recorded.\n{e}")
		sys.exit(0)
	try:
		df = pd.read_csv(args.prediction_file)
	except Exception as e:
		print(f"Please specify a correct test file (csv format).\n{e}")
		sys.exit(0)

	pred = Predict(df, data)
	pred.predict()
	pred.final_output()

	return None


if __name__ == "__main__":
	args = ft_argparser()
	main(args)

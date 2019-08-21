#!/usr/bin/env python3

import sys
import pandas as pd
import argparse
import pickle

from constants import FEATURES, HOUSES
from class_predict import Predict

def ft_argparser():

	parser = argparse.ArgumentParser()
	parser.add_argument("prediction_file", type=str, help="csv file containing the data to analyze")
	args = parser.parse_args()
	return args


# def print_output(prediction, confidence=None, conf_level=None):
# 	print(f"Mean estimation of the price of your car : {prediction:.0f} euros")
# 	if confidence:
# 		print(f"Prediction interval at {conf_level * 100:.0f} % : [{prediction - confidence:.0f}, {prediction + confidence:.0f}]")
# 	return None


def main(args):

	try:
		with open('data.pkl', 'rb') as f:
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
	print(pred.df_testing.head())
	print(pred.df_testing_norm.head())
	pred.predict()
	print(pred.df_testing_norm.head())
	print(pred.final_output())

	return None



if __name__ == "__main__":
	args = ft_argparser()
	main(args)
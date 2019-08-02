#!/usr/bin/env python3

import sys
import pandas as pd
import argparse
import pickle

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


def	main(args):

	try:
		with open('data.pkl', 'rb') as f:
			data = pickle.load(f)
	except Exception as e:
		print(f"Please provide a valid data file.\n{e}")
		sys.exit(0)

	try:
		df = pd.read_csv(args.prediction_file)
	except Exception as e:
		print(f"Please specify a correct file.\n{e}")
		sys.exit(0)



if __name__ == "__main__":
	args = ft_argparser()
	main(args)
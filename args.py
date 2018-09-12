import argparse
from logging import log


def get_args():
	parser = argparse.ArgumentParser(description='Text Classification')
	parser.add_argument('-c', type=str, help='Path to config files folder')
	args = parser.parse_args()
	if not args.c:
		log('ERROR: The input folder is required')
		parser.exit(1)
	return args.c

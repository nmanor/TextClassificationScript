import datetime


def init():
	global FEATURES, NORMALIZATION, OUTPUT_DIR, METHODS, TRAIN_DIR, TEST_DIR, NORM_PATH, RESULTS_PATH
	FEATURES = []
	NORMALIZATION = ''
	OUTPUT_DIR = ''
	METHODS = ''
	TRAIN_DIR = ''
	TEST_DIR = ''
	NORM_PATH = ''
	RESULTS_PATH = ''
	TRAIN_DATA=''
	TEST_DATA=''

def print_message(msg, num_tabs=0):
	if num_tabs > 0:
		print('\t' * num_tabs, end='')
	print("{} >> {}".format(datetime.datetime.now(), msg))

import json
import pickle


import global_parameters
from global_parameters import print_message
from write_xlsx_file import write_xlsx_file

global method_names


def print_results(results):
	for config in results.keys():
		current_key = json.loads(config)
		print("-----------------")
		print(json.dumps(current_key, indent=4, sort_keys=True))
		print(json.dumps(results[config], indent=4, sort_keys=True))
		print("-----------------")


def write_results(results):
	name = ''
	for f in global_parameters.FEATURES:
		name += f
		name += '#'
	name += global_parameters.NORMALIZATION
	for m in global_parameters.METHODS:
		name += '#'
		name += m
	print_message("Writing results...")
	with open(global_parameters.RESULTS_PATH + "\\" + name + ".pickle", 'wb+') as file:
		pickle.dump(results, file)
	write_xlsx_file(global_parameters.RESULTS_PATH + "\\" + name + ".pickle")



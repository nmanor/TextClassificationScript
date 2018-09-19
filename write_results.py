import json
import pickle


from global_parameters import print_message, GlobalParameters
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
	glbs = GlobalParameters()
	name = ''
	for f in glbs.FEATURES:
		name += f
		name += '#'
	name += glbs.NORMALIZATION
	for m in glbs.METHODS:
		name += '#'
		name += m
	print_message("Writing results...")
	with open(glbs.RESULTS_PATH + "\\" + name + ".pickle", 'wb+') as file:
		pickle.dump(results, file)
	write_xlsx_file(glbs.RESULTS_PATH + "\\" + name + ".pickle")



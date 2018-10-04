import json

import pickle
import xlsxwriter as writer
from statistical_analysis import significance_testing_information


def write_xlsx_file(pickle_file_dir, baseline_index=0):
	with open(pickle_file_dir, 'rb') as file:
		pickle_file_content = pickle.load(file)
	dataset_name = json.loads(list(pickle_file_content.keys())[0])['train'].split("\\")[-3]
	results_path = json.loads(list(pickle_file_content.keys())[0])['results']
	pickle_file_content = reformat_data(pickle_file_content, baseline_index)
	write_file_content(pickle_file_content, dataset_name, results_path, baseline_index)


def write_file_content(pickle_file_content, dataset_name, results_path, baseline_index):
	dataset_name = dataset_name
	file_path = results_path + "\\" + dataset_name + '.xlsx'
	print("writing " + file_path)
	workbook = writer.Workbook(file_path)
	worksheet = workbook.add_worksheet()
	rows = 0
	cols = 0
	best_sig, best_non_sig, bigger_sig, bigger_non_sig, title_format = init_formats(workbook)
	rows = write_data_information(best_non_sig, best_sig, bigger_non_sig, bigger_sig, cols, dataset_name, rows,
								  title_format, worksheet)
	global method_names
	cols = 1
	for method in method_names:
		worksheet.write(rows, cols, method)
		cols += 1
	content_keys = sort_dict_by_normalization(list(pickle_file_content.keys()), baseline_index)
	for run in content_keys:
		rows += 1
		cols = 0
		worksheet.write(rows, cols, run)
		for method in pickle_file_content[run].keys():
			index = method_names.index(method)
			cols = index + 1
			value, info = pickle_file_content[run][method]
			if info == 'max_sig':
				worksheet.write(rows, cols, value, best_sig)
			elif info == 'max_non_sig':
				worksheet.write(rows, cols, value, best_non_sig)
			elif info == 'bigger_sig':
				worksheet.write(rows, cols, value, bigger_sig)
			elif info == 'bigger_non_sig':
				worksheet.write(rows, cols, value, bigger_non_sig)
			else:
				worksheet.write(rows, cols, value)
	workbook.close()


def write_data_information(best_non_sig, best_sig, bigger_non_sig, bigger_sig, cols, dataset_name, rows, title_format,
						   worksheet):
	worksheet.write(0, 0, dataset_name, title_format)
	rows += 1
	worksheet.write(rows, cols, "Best and Significant than baseline", best_sig)
	rows += 1
	worksheet.write(rows, cols, "Best and NOT Significant than baseline", best_non_sig)
	rows += 1
	worksheet.write(rows, cols, "Better than baseline and Significant than baseline", bigger_sig)
	rows += 1
	worksheet.write(rows, cols, "Better than baseline and NOT Significant than baseline", bigger_non_sig)
	rows += 1
	rows, cols = write_information(worksheet, rows, cols)
	return rows


def reformat_data(data, baseline_index=0):
	reformed_data = {}
	for cfg in data.keys():
		name = ''
		config = json.loads(cfg)
		for feature in config['features']:
			name += str(feature)
			name += '|'
		name = name.rstrip('|')
		name += '@'
		if config['nargs'] == '':
			name += 'None'
		else:
			name += config['nargs']
		reformed_data[name] = data[cfg]

	methods = {}
	for run in reformed_data.values():
		for method in run.keys():
			if method in methods.keys():
				methods[method].append(run[method])
			else:
				methods[method] = [run[method]]
	global method_names
	method_names = list(methods.keys())

	for method in methods.keys():
		results, threshold = significance_testing_information(methods[method][baseline_index], methods[method])
		for run in reformed_data.keys():
			reformed_data[run][method] = results[0]
			results.pop(0)

	# sorted_keys = sorted(reformed_data, key=lambda x:sort_by_length(x,reformed_data,baseline_index), reverse=True)
	return reformed_data


def sort_key(key, topkey):
	if key == topkey:
		return -1
	return len(key)


def sort_dict_by_normalization(keys, baseline_index=0):
	top_key = keys[baseline_index]
	keys = sorted(keys, key=lambda x: x.split('@')[1])
	keys = sorted(keys, key=lambda x: sort_key(x, top_key))
	return keys


def write_information(worksheet, rows, cols):
	rows += 3
	worksheet.write(rows, cols, "C - Spelling Correction")
	rows += 1
	worksheet.write(rows, cols, "L - Lowercase")
	rows += 1
	worksheet.write(rows, cols, "H - HTML tags")
	rows += 1
	worksheet.write(rows, cols, "P - Punctuations")
	rows += 1
	worksheet.write(rows, cols, "S - Stop words")
	rows += 1
	worksheet.write(rows, cols, "R - Repeated chars")
	rows += 1
	worksheet.write(rows, cols, "T - Stemming")
	rows += 1
	worksheet.write(rows, cols, "M - Lemmatizer")
	rows += 2
	return rows, cols


def init_formats(workbook):
	title_format = workbook.add_format({'bold': True,
										'font_color': '#00338f',
										'size': 15,
										'underline': True})
	best_sig = workbook.add_format({'bold': True,
									'font_color': '#e60000',
									'size': 11})
	best_non_sig = workbook.add_format({'bold': False,
										'font_color': '#e60000',
										'size': 11})
	bigger_sig = workbook.add_format({'bold': True,
									  'font_color': 'blue',
									  'size': 11})
	bigger_non_sig = workbook.add_format({'bold': False,
										  'font_color': 'blue',
										  'size': 11})
	return best_sig, best_non_sig, bigger_sig, bigger_non_sig, title_format


if __name__ == '__main__':
	pickle_dir = input("enter pickle file path:\n")
	with open(pickle_dir, 'rb') as file:
		pickle_file_content = pickle.load(file)
	print("enter baseline index: \n")
	for i, item in enumerate(pickle_file_content.keys()):
		print("{} - \n{}".format(i, item))
	print('\n')
	baseline_index = input()
	write_xlsx_file(pickle_dir, int(baseline_index))

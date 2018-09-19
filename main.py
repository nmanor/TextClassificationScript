import json
import os
import shutil
import traceback

from args import get_args
from classification import classify
from features import extract_features
from normalization import normalize
from notification_handler import send_work_done
from write_results import write_results, print_results
import global_parameters


def set_global_parameters(config):
	global_parameters.FEATURES = config['features']
	global_parameters.NORMALIZATION = ''.join(sorted(config['nargs'].upper()))
	global_parameters.OUTPUT_DIR = config['output_csv']
	global_parameters.METHODS = config['methods']
	global_parameters.TRAIN_DIR = config['train']
	global_parameters.TEST_DIR = config['test']
	global_parameters.RESULTS_PATH = config['results']


def print_run_details():
	print("""
	---------------------------------------
	Training path: {}
	Testing Path: {}
	Features: {}
	Normalization: {}
	Methods: {}
	Output Path: {}
	Results Path: {}
	---------------------------------------
	""".format(global_parameters.TRAIN_DIR, global_parameters.TEST_DIR, global_parameters.FEATURES,
			   global_parameters.NORMALIZATION, global_parameters.METHODS, global_parameters.OUTPUT_DIR,
			   global_parameters.RESULTS_PATH))


def main(cfg):
	try:
		configs = get_cfg_files(cfg)
		results = {}
		n_test_dir = ''
		total_files = len(configs)
		for i, config in enumerate(configs):
			global_parameters.print_message("Running config {}/{}".format(i + 1, total_files))
			set_global_parameters(config)
			print_run_details()
			n_train_dir = normalize()
			if global_parameters.TEST_DIR != '':
				n_test_dir = normalize(test=True)
			train, tr_labels, test, ts_labels = extract_features(n_train_dir, n_test_dir)
			results[json.dumps(config)] = classify(train, tr_labels, test, ts_labels)
		print_results(results)
		write_results(results)
		send_work_done(global_parameters.TRAIN_DIR)
		clean_backup_files()
	except Exception as e:
		traceback.print_exc()
		send_work_done(global_parameters.TRAIN_DIR, "", error=str(e), traceback=str(traceback.format_exc()))


def get_cfg_files(dir):
	items = []
	for file in os.listdir(dir):
		if file.endswith('.json'):
			with open(dir + "\\" + file, "r", encoding="utf8", errors='replace') as f:
				# keys : train,test='',output_csv,nargs,features.
				items.append(json.load(f))
	return items


def clean_backup_files():
	global_parameters.print_message("removing temp files...")
	folder_path = '\\'.join(global_parameters.RESULTS_PATH.split("\\")[:-1]) + "\\temp_backups"
	shutil.rmtree(folder_path, ignore_errors=True)


if __name__ == '__main__':
	global_parameters.init()
	cfg_dir = get_args()
	main(cfg_dir)

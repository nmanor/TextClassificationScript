import json
import os
import shutil
import traceback

from args import get_args
from classification import classify
from features import extract_features
from global_parameters import GlobalParameters, print_message
from normalization import normalize
from notification_handler import send_work_done
from write_results import write_results, print_results


def set_global_parameters(config):
	glbls = GlobalParameters()
	glbls.FEATURES = config['features']
	glbls.NORMALIZATION = ''.join(sorted(config['nargs'].upper()))
	glbls.OUTPUT_DIR = config['output_csv']
	glbls.METHODS = config['methods']
	glbls.TRAIN_DIR = config['train']
	glbls.TEST_DIR = config['test']
	glbls.RESULTS_PATH = config['results']


def print_run_details():
	glbs = GlobalParameters()
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
	""".format(glbs.TRAIN_DIR, glbs.TEST_DIR, glbs.FEATURES,
			   glbs.NORMALIZATION, glbs.METHODS, glbs.OUTPUT_DIR,
			   glbs.RESULTS_PATH))


def main(cfg):
	try:
		glbs = GlobalParameters()
		configs = get_cfg_files(cfg)
		results = {}
		n_test_dir = ''
		total_files = len(configs)
		for i, config in enumerate(configs):
			print_message("Running config {}/{}".format(i + 1, total_files))
			set_global_parameters(config)
			print_run_details()
			n_train_dir = normalize()
			if glbs.TEST_DIR != '':
				n_test_dir = normalize(test=True)
			train, tr_labels, test, ts_labels = extract_features(n_train_dir, n_test_dir)
			results[json.dumps(config)] = classify(train, tr_labels, test, ts_labels)
		print_results(results)
		write_results(results)
		send_work_done(glbs.TRAIN_DIR)
		clean_backup_files()
	except Exception as e:
		traceback.print_exc()
		send_work_done(glbs.TRAIN_DIR, "", error=str(e), traceback=str(traceback.format_exc()))


def get_cfg_files(dir):
	items = []
	for file in os.listdir(dir):
		if file.endswith('.json'):
			with open(dir + "\\" + file, "r", encoding="utf8", errors='replace') as f:
				# keys : train,test='',output_csv,nargs,features.
				items.append(json.load(f))
	return items


def clean_backup_files():
	glbs = GlobalParameters()
	print_message("removing temp files...")
	folder_path = '\\'.join(glbs.RESULTS_PATH.split("\\")[:-1]) + "\\temp_backups"
	shutil.rmtree(folder_path, ignore_errors=True)


if __name__ == '__main__':
	cfg_dir = get_args()
	main(cfg_dir)

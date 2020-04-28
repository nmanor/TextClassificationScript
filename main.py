import datetime
import json
import os
import shutil
import traceback

import numpy as np

from classification import classify
from features import extract_features
from global_parameters import GlobalParameters, print_message
from normalization import normalize
from notification_handler import send_work_done
from stylistic_features import text_language
from words_cloud import generate_word_clouds
from write_results import write_results


def set_global_parameters(configs):
    glbls = GlobalParameters()
    config = configs[1]
    glbls.FILE_NAME = configs[0]
    glbls.FEATURES = config["features"]
    glbls.NORMALIZATION = "".join(sorted(config["nargs"].upper()))
    glbls.METHODS = config["methods"]
    glbls.DATASET_DIR = config["dataset"]
    glbls.RESULTS_PATH = config["results"]
    glbls.MEASURE = config["measure"]
    glbls.STYLISTIC_FEATURES = config["stylistic_features"]
    glbls.SELECTION = list(config["selection"].items())
    glbls.K_FOLDS = config["k_folds_cv"]
    glbls.ITERATIONS = config["iterations"]
    glbls.BASELINE_PATH = config["baseline_path"]
    glbls.EXPORT_AS_BASELINE = config["export_as_baseline"]
    try:
        if "language" in config:
            glbls.LANGUAGE = config["language"]
        else:
            path = config["dataset"] + "\\" + os.listdir(config["dataset"])[0]
            glbls.LANGUAGE = text_language(
                open(path, "r", encoding="utf8", errors="replace").read()
            )
    except:
        glbls.LANGUAGE = "english"


def print_run_details():
    glbs = GlobalParameters()
    print(
        """
	---------------------------------------
	Dataset path: {}
	Features: {}
	Stylistic Features: {}
	Normalization: {}
	Methods: {}
	Measure: {}
	Export as baseline: {}
	Results Path: {}
	---------------------------------------
	""".format(
            glbs.DATASET_DIR,
            glbs.FEATURES,
            glbs.STYLISTIC_FEATURES,
            glbs.NORMALIZATION,
            glbs.METHODS,
            glbs.MEASURE,
            glbs.EXPORT_AS_BASELINE,
            glbs.RESULTS_PATH,
        )
    )


def add_results(old_results, glbs):
    temp = {}
    temp["results"] = old_results[glbs.FILE_NAME]
    temp["featurs"] = glbs.FEATURES
    temp["normalization"] = glbs.NORMALIZATION
    temp["stylistic_features"] = glbs.STYLISTIC_FEATURES
    temp["k_folds"] = glbs.K_FOLDS
    temp["iterations"] = glbs.ITERATIONS
    temp["baseline_path"] = glbs.BASELINE_PATH
    temp["num_of_features"] = glbs.NUM_OF_FEATURE
    old_results[glbs.FILE_NAME] = temp
    return old_results


def add_results_glbs(results, glbs):
    glbs.RESULTS.update(results)


def divide_results(result):
    new_result = {}

    for config_name, dic in result.items():
        for method, score in dic["results"].items():
            for measure, value in score.items():
                if measure not in new_result.keys():
                    new_result[measure] = {}
                new_result[measure][config_name] = {}
                new_result[measure][config_name]["results"] = {}

    for config_name, dic in result.items():
        for method, score in dic["results"].items():
            for measure, value in score.items():
                new_result[measure][config_name]["results"][method] = value
                new_result[measure][config_name]["featurs"] = dic["featurs"]
                new_result[measure][config_name]["normalization"] = dic["normalization"]
                new_result[measure][config_name]["stylistic_features"] = dic["stylistic_features"]
                new_result[measure][config_name]["k_folds"] = dic["k_folds"]
                new_result[measure][config_name]["iterations"] = dic["iterations"]
                new_result[measure][config_name]["baseline_path"] = dic["baseline_path"]
                new_result[measure][config_name]["num_of_features"] = dic["num_of_features"]

    return_results = {}
    for measure, data in new_result.items():
        if new_result[measure] != {}:
            return_results[measure] = data

    return return_results


def export_as_baseline(result, config):
    name = "Baseline " + datetime.datetime.now().strftime("%d.%m.%Y") + " -"
    for feature in config["features"]:
        name += " " + feature
    for feature in config["stylistic_features"]:
        name += " " + feature

    main_measure = config["measure"][0]
    best_method = list(result.keys())[0]
    best_score = result[best_method]

    for method in result:
        mean = np.mean(result[method][main_measure])
        if np.mean(best_score[main_measure]) < mean:
            best_score = result[method]
            best_method = method

    baseline = {
        "run_date": datetime.datetime.now().strftime("%d.%m.%Y %H:%M"),
        "best_method": best_method,
        "best_score": best_score,
        "all_scores": result,
        "original_config": config,
    }

    if config["baseline_path"].endswith(".json"):
        config["baseline_path"] = "\\".join(config["baseline_path"].split("\\")[:-1])

    with open(config["baseline_path"] + "\\" + name + ".json", "w") as file:
        json.dump(baseline, file, indent=6)


def main(cfg):
    try:
        # nltk.download("vader_lexicon")
        glbs = GlobalParameters()
        configs = get_cfg_files(cfg)
        total_files = len(configs)
        results = {}
        for i, config in enumerate(configs):
            print_message("Running config {}/{}".format(i + 1, total_files))
            set_global_parameters(config)
            print_run_details()
            dataset_dir = normalize()
            X, y = extract_features(dataset_dir)
            config_result = classify(X, y, glbs.K_FOLDS, glbs.ITERATIONS)
            glbs.RESULTS[glbs.FILE_NAME] = config_result
            glbs.RESULTS = add_results(glbs.RESULTS, glbs)
            if glbs.EXPORT_AS_BASELINE:
                export_as_baseline(config_result, config[1])
        if glbs.WORDCLOUD:
            print_message("Generating word clouds (long processes)")
            generate_word_clouds()
        add_results_glbs(results, glbs)
        write_results(divide_results(glbs.RESULTS))
        send_work_done(glbs.DATASET_DIR)
        print_message("Done!")
    except Exception as e:
        traceback.print_exc()
        send_work_done(
            glbs.DATASET_DIR, "", error=str(e), traceback=str(traceback.format_exc())
        )


def get_cfg_files(dir):
    items = []
    for file in os.listdir(dir):
        if file.endswith(".json"):
            with open(dir + "\\" + file, "r", encoding="utf8", errors="replace") as f:
                items.append((file.replace(".json", ""), json.load(f)))
    return items


def clean_backup_files():
    glbs = GlobalParameters()
    print_message("removing temp files...")
    folder_path = "\\".join(glbs.RESULTS_PATH.split("\\")[:-1]) + "\\temp_backups"
    shutil.rmtree(folder_path, ignore_errors=True)


if __name__ == "__main__":
    cfg_dir = r"C:\Users\Mickey\Documents\kerner\data\cfgs"
    if not os.path.exists(cfg_dir):
        cfg_dir = r"C:\Users\natan\OneDrive\מסמכים\test\cfgs"
    main(cfg_dir)

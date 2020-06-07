import itertools
import json
import os
import re
from random import shuffle

import xlsxwriter
from sklearn.feature_extraction.text import CountVectorizer


def write_result(prediction, classifier, path):
    dic = {1: "normal", 0: "anorexia"}
    prediction = list(prediction)
    with open(path + "\\result.txt", "r", encoding="utf8", errors='replace') as file:
        text = file.read()
    for i, predict in enumerate(prediction):
        text = text.replace('!!' + classifier + str(hex(i)) + '!!', classifier + ': ' + dic[predict])
    with open(path + "\\result.txt", "w", encoding="utf8", errors='replace') as file:
        file.write(text)


def regex():
    text = open(r"C:\Users\user\Documents\test\dataset\all.txt", 'r', encoding="utf8", errors='ignore').read()

    result = re.findall(r'\w*מילה\w*', text)

    tup = [(word, result.count(word)) for word in set(result)]
    for word in tup:
        print(word[0] + ': ' + str(word[1]))


def adapt_other_json(path_from, path_to):
    """
    Take JSON cfgs files from one computer and adapt them to different computer
    :param path_from: the path fo the original files
    :param path_to: the path do drop the new files
    :return:
    """
    all_dics = []
    i = 1
    total_length = str(len([file for file in os.listdir(path_from) if file.endswith('.json')]))
    for file in os.listdir(path_from):
        if file.endswith('.json'):
            with open(path_from + "\\" + file, "r", encoding="utf8", errors='replace') as f:
                old_dic = json.load(f)

                # For Hill Climbing only!
                best = ["wc", "cc", "sc", "alw", "als", "aws", "awl", "caf", "lof", "fdf", "thf", "nw"]

                new_dic = {"train": r"C:\Users\user\Documents\test\dataset\training",
                           "test": r"C:\Users\user\Documents\test\dataset\testing",
                           "output_csv": r"C:\Users\user\Documents\test\output",
                           "nargs": "e" if old_dic["nargs"] != "" else "", "features": old_dic["features"],
                           "results": r"C:\Users\user\Documents\test\results",
                           "methods": ["lr", "svc", "mlp", "rf", "mnb"],
                           "measure": ["accuracy_score"],
                           "stylistic_features": list(set(old_dic["stylistic_features"] + best)), "selection": {}}
                if new_dic not in all_dics:
                    all_dics += [new_dic]
                    with open(path_to + "\\" + file, 'w') as fp:
                        json.dump(new_dic, fp, indent=4)
                        print('File ' + str(i) + '/' + total_length)
                i += 1


def gen_cfgs_in_range(output_path):
    i = 1
    # HE
    lst = ["aof", "fdf", "ftf", "anf", "huf", "mef", "vof", "pnf", "agf", "te", "xte", "slf", "spf", "thf", "caf",
           "vuf", "def", "inf", "sif", "lof", "frc", "ref", "acf", "wef", "pw", "nw", "e50th", "e50tth"]
    # EN
    lst = ["aof", "fdf", "ftf", "anf", "huf", "fpe", "spe", "tpe", "nof", "vof", "pnf", "agf", "te", "xte", "slf",
           "spf", "thf", "caf", "sxf", "cuf", "alf", "skf", "def", "inf", "sif", "lof", "frc", "ref", "acf", "wef",
           "pw", "nw", "e50te", "e50tte"]

    for fe in lst:
        cfgs = {
            "dataset": "C:\\Users\\natan\\OneDrive\\מסמכים\\test\\dataset",
            "baseline_path": "C:\\Users\\natan\\OneDrive\\\u05de\u05e1\u05de\u05db\u05d9\u05dd\\test\\baseline",
            "export_as_baseline": False,
            "k_folds_cv": 5,
            "iterations": 20,
            "nargs": ["l"],
            "features": [],
            "results": "C:\\Users\\natan\\OneDrive\\\u05de\u05e1\u05de\u05db\u05d9\u05dd\\test\\results",
            "methods": ["lr", "svc", "mlp", "rf", "mnb"],
            "measure": ["accuracy"],
            "stylistic_features": [fe],
            "selection": {}
        }

        with open(output_path + "\\" + fe + ".json", 'w') as fp:
            json.dump(cfgs, fp, indent=4)

        print(i)
        i += 1


def random_groups(output_path):
    """result = []
    for i in range(1, len(lst) + 1):
        result += list(itertools.combinations(lst, i))"""

    lst = ["ftf", "huf", "fpe", "nof", "te", "xte", "fdf", "cuf", "alf", "def", "sif", "wef", "e50tte"]

    bests = ['aof', 'caf', 'inf', 'thf', 'vof', 'nw', 'pw', 'sxf', 'spe', 'lof', 'anf', 'frc', 'slf', 'e50te', 'acf',
             'agf', "pnf", "skf", "ref", "spf", "tpe"]

    for i, family in enumerate(lst):
        cfgs = {
            "dataset": "C:\\Users\\natan\\OneDrive\\מסמכים\\test\\dataset",
            "baseline_path": "C:\\Users\\natan\\OneDrive\\\u05de\u05e1\u05de\u05db\u05d9\u05dd\\test\\baseline",
            "export_as_baseline": False,
            "k_folds_cv": 5,
            "iterations": 20,
            "nargs": ["l"],
            "features": [],
            "results": "C:\\Users\\natan\\OneDrive\\\u05de\u05e1\u05de\u05db\u05d9\u05dd\\test\\results",
            "methods": ["lr", "svc", "mlp", "rf", "mnb"],
            "measure": ["accuracy"],
            "stylistic_features": [family] + bests,
            "selection": {}
        }

        with open(output_path + "\\" + family + ".json", 'w') as fp:
            json.dump(cfgs, fp, indent=4)

        print(i + 1)


def random_groups2(output_path):
    lst = ['caf', 'inf', 'e50tte', 'tpe', 'acf', 'ftf', 'fdf', 'fpe', 'e50te', 'vof', 'thf', 'nw', 'wef', 'def', 'cuf',
           'frc', 'nof', 'xte', 'te', 'sif', 'anf', 'pnf', 'huf', 'sxf', 'pw', 'lof', 'aof', 'agf', 'spe', 'spf']

    result = []
    for i in range(len(lst) - 3, len(lst)):
        result += list(itertools.combinations(lst, i))

    shuffle(result)
    result = result[:50]

    for i, family in enumerate(result):
        cfgs = {
            "dataset": "C:\\Users\\natan\\OneDrive\\מסמכים\\test\\dataset",
            "baseline_path": "",
            "export_as_baseline": False,
            "k_folds_cv": 5,
            "iterations": 3,
            "nargs": ["l"],
            "features": [],
            "results": "C:\\Users\\natan\\OneDrive\\\u05de\u05e1\u05de\u05db\u05d9\u05dd\\test\\results",
            "methods": ["lr", "svc", "mlp", "rf", "mnb"],
            "measure": ["accuracy"],
            "stylistic_features": list(family),
            "selection": {}
        }

        with open(output_path + "\\" + str(i) + ".json", 'w') as fp:
            json.dump(cfgs, fp, indent=4)

        print(i + 1)

def get_fetuer_by_DF(corpus):
    vec = CountVectorizer(min_df=3, lowercase=False).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [[word, sum_words[0, idx], 0] for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    for row in range(len(words_freq)):
        for post in corpus:
            post = re.findall(r"(?u)\b\w\w+\b", post)
            if words_freq[row][0] in post:
                words_freq[row][2] += 1
    file_path = r"C:\Users\natan\OneDrive\מסמכים\test\words.xlsx"
    workbook = xlsxwriter.Workbook(file_path)
    worksheet = workbook.add_worksheet()
    row = 2
    for word in words_freq:
        worksheet.write("A" + str(row + 1), word[0])
        worksheet.write_number("B" + str(row + 1), word[1])
        worksheet.write_number("C" + str(row + 1), word[2])
        row += 1

    worksheet.add_table(
        "A2:C" + str(row),
        {
            "columns": [
                {"header": "Word"},
                {"header": "F"},
                {"header": "DF"}
            ],
            "style": "Table Style Light 8",
        },
    )

    workbook.close()



if __name__ == "__main__":
    random_groups2(r"C:\Users\natan\OneDrive\מסמכים\test\cfgs")

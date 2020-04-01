import json
import os
import re
from random import shuffle


def create_dataset(dir, output, split=0.33):
    """
    :param dir: the directory with all the files (txt & json of anorexia and normal)
    :param output: where to put the merged files
    :param split: the ratio of the split (default: 1/3)
    """
    # get all the data and put it in the right list
    print('Reading files...')
    anorexia = []
    normal = []
    for file in os.listdir(dir):
        if file.endswith('.json'):
            with open(dir + "\\" + file, "r", encoding="utf8", errors='replace') as f:
                if json.load(f)["classification"] == "anorexia":
                    anorexia = anorexia + [file]
                else:
                    normal = normal + [file]

    # shuffle the lists
    shuffle(anorexia)
    shuffle(normal)

    # split the anorexia files
    print('Split the anorexia files...')
    test = anorexia[:int(split*len(anorexia))]
    train = anorexia[int(split*len(anorexia)):]

    # save the content of the anorexia test
    text = ''
    for file in test:
        with open(dir + "\\" + file, "r+", encoding="utf8", errors='replace') as f:
            dic = json.load(f)
            text += '\n' + open(dir + "\\" + dic["file_id"], "r", errors='replace', encoding='utf8').read().replace(
                '\n', ' ')
    with open(output + "\\testing\\anorexia.txt", "w", encoding="utf8", errors='replace') as file:
        file.write(text[1:])

    # save the content of the anorexia train
    text = ''
    for file in train:
        with open(dir + "\\" + file, "r+", encoding="utf8", errors='replace') as f:
            dic = json.load(f)
            text += '\n' + open(dir + "\\" + dic["file_id"], "r", errors='replace', encoding='utf8').read().replace(
                '\n', ' ')
    with open(output + "\\training\\anorexia.txt", "w", encoding="utf8", errors='replace') as file:
        file.write(text[1:])

    print('Done saving the anorexia files')

    # split the normal files
    print('Split the normal files...')
    test = normal[:int(split*len(normal))]
    train = normal[int(split*len(normal)):]

    # save the content of the normal test
    text = ''
    for file in test:
        with open(dir + "\\" + file, "r+", encoding="utf8", errors='replace') as f:
            dic = json.load(f)
            text += '\n' + open(dir + "\\" + dic["file_id"], "r", errors='replace', encoding='utf8').read().replace(
                '\n', ' ')
    with open(output + "\\testing\\normal.txt", "w", encoding="utf8", errors='replace') as file:
        file.write(text[1:])

    # save the content of the normal train
    text = ''
    for file in train:
        with open(dir + "\\" + file, "r+", encoding="utf8", errors='replace') as f:
            dic = json.load(f)
            text += '\n' + open(dir + "\\" + dic["file_id"], "r", errors='replace', encoding='utf8').read().replace(
                '\n', ' ')
    with open(output + "\\training\\normal.txt", "w", encoding="utf8", errors='replace') as file:
        file.write(text[1:])

    print('Done saving the normal files')
    print('Process done')


def count_stopwords(path):
    text = ''
    for file in os.listdir(path):
        with open(path + "\\" + file, "r", encoding="utf8", errors='replace') as f:
            text += f.read().replace('\n', ' ')

    for char in r".,;()[]{}:-–?!’'\"“”/&*@\\†‡°#~_|¦¶ˆ^•№§%‰‱¤$™®©":
        text = text.replace(char, '')

    text = [word for word in text.split(' ') if word != ' ']
    stopword = {}
    for word in text:
        if word in stopword:
            stopword[word] += 1
        else:
            stopword[word] = 1

    i = 0
    for w in sorted(stopword, key=stopword.get, reverse=True):
        print(w)


def pre_results(test_data, path):
    text = ""
    for i, post in enumerate(test_data):
        text += post[0]
        text += '\n----------------------'
        text += '\noriginal: ' + post[1][:-4] + '\n'
        for method in ['svc', 'rf', 'mlp', 'lr', 'mnb']:
            text += '!!' + method + str(hex(i)) + '!!\n'
        text += '\n'
    with open(path + "\\result.txt", "w", encoding="utf8", errors='replace') as file:
        file.write(text)


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
    # REF = ["dw", "tw", "dh", "dx", "tx"]
    # ACF = ["wc", "cc", "sc", "alw", "als", "aws", "awl"]
    # WEF = ["ww", "owc", "twc", "ttc"]

    # HE
    """lst = ["aof", "fdf", "ftf", "anf", "huf", "mef", "vof", "pnf", "agf", "te", "xte", "slf", "spf", "thf", "caf",
           "vuf", "def", "inf", "sif", "lof", "frc", ["dw", "tw", "dh", "dx", "tx"], ["wc", "cc", "sc", "alw", "als",
            "aws", "awl"], ["ww", "owc", "twc", "ttc"], "pw", "nw", "e50th", "e50tth"]"""
    # EN
    """lst = ["aof", "fdf", "ftf", "anf", "huf", "fpe", "spe", "tpe", "nof", "vof", "pnf", "agf", "te", "xte", "slf",
           "spf", "thf", "caf", "sxf", "cuf", "alf", "skf", "def", "inf", "sif", "lof", "frc", ["dw", "tw", "dh", "dx",
            "tx"], ["wc", "cc", "sc", "alw", "als", "aws", "awl"], ["ww", "owc", "twc", "ttc"], "pw", "nw", "e50te",
             "e50tte"]"""

    # EN top 10
    """lst = ["FDF", "INF", ["wc", "cc", "sc", "alw", "als", "aws", "awl"], "VOF", "CAF", "TPE", "E50TE", "E50TTE", "CUF",
           "DEF"]"""

    # HE Top 10
    # lst = ["FDF", "E50TH", "E50TTH", "HUF", "MEF", "ANF", "AOF", "FTF", "TE", "XTE"]
    lst = ['FDF', 'HUF', 'MEF', 'ANF', 'AOF', 'FTF', 'XTE', 'THF', 'VUF', 'SIF']

    h_lst = []

    for fe in lst:
        h_lst += [fe.lower()] if isinstance(fe, str) else fe
        cfgs = {
            "train": "C:\\Users\\natan\\OneDrive\\מסמכים\\test\\dataset\\training",
            "test": "C:\\Users\\natan\\OneDrive\\מסמכים\\test\\dataset\\testing",
            "output_csv": "C:\\Users\\natan\\OneDrive\\מסמכים\\test\\output",
            "nargs": "",
            "features": [],
            "results": "C:\\Users\\natan\\OneDrive\\מסמכים\\test\\results",
            "methods": ["lr", "svc", "mlp", "rf", "mnb"],
            "measure": ["accuracy_score"],
            # "stylistic_features": fe if isinstance(fe, list) else [fe],
            "stylistic_features": h_lst,
            "selection": {}
        }

        with open(output_path + "\\" + str(i) + ".json", 'w') as fp:
            json.dump(cfgs, fp, indent=4)

        print(i)
        i += 1


def random_groups(output_path):
    # HE
    # lst = ["fdf", "e50th", "e50tth", "huf", "mef", "anf", "aof", "ftf"]

    # EN
    lst = ["fdf", "inf", "vof", "e50te", "caf", "e50tte", "tpe", "ftf", "cuf", "fpe", "te", "xte",
           ["wc", "cc", "sc", "alw", "als", "aws", "awl"], ["ww", "owc", "twc", "ttc"]]

    se = []

    lst = ["pnf", "ftf", "fdf", "frc", "aof", "anf", "xte"]
    lst = ["ftf", "fdf", "mef", "frc", "anf"]
    x = len(lst)
    for i in range(1 << x):
        temp = [lst[j] for j in range(x) if (i & (1 << j))]
        if len(temp) > 1:
            se += [temp]
    print(len(se))

    i = 1
    for fe in se:
        x1 = []
        for x2 in fe:
            x1 += x2 if isinstance(x2, list) else [x2]
        cfgs = {
            "train": "C:\\Users\\natan\\OneDrive\\מסמכים\\test\\dataset\\training",
            "test": "C:\\Users\\natan\\OneDrive\\מסמכים\\test\\dataset\\testing",
            "output_csv": "C:\\Users\\natan\\OneDrive\\מסמכים\\test\\output",
            "nargs": "",
            "features": [],
            "results": "C:\\Users\\natan\\OneDrive\\מסמכים\\test\\results",
            "methods": ["lr", "svc", "mlp", "rf", "mnb"],
            "measure": ["accuracy_score"],
            "stylistic_features": x1,
            "selection": {}
        }

        with open(output_path + "\\" + str(i) + ".json", 'w') as fp:
            json.dump(cfgs, fp, indent=4)

        print(i)
        i += 1


if __name__ == "__main__":
    random_groups(r"C:\Users\natan\OneDrive\מסמכים\test\cfgs")
    # gen_cfgs_in_range(r"C:\Users\natan\OneDrive\מסמכים\test\cfgs")

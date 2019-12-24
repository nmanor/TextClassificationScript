import json
import os
import re
from random import shuffle


def create_dataset(dir, output, split = 0.33):
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
            text += '\n' + open(dir + "\\" + dic["file_id"], "r", errors='replace', encoding='utf8').read().replace('\n', ' ')
    with open(output + "\\testing\\anorexia.txt", "w", encoding="utf8", errors='replace') as file:
        file.write(text[1:])

    # save the content of the anorexia train
    text = ''
    for file in train:
        with open(dir + "\\" + file, "r+", encoding="utf8", errors='replace') as f:
            dic = json.load(f)
            text += '\n' + open(dir + "\\" + dic["file_id"], "r", errors='replace', encoding='utf8').read().replace('\n', ' ')
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
            text += '\n' + open(dir + "\\" + dic["file_id"], "r", errors='replace', encoding='utf8').read().replace('\n', ' ')
    with open(output + "\\testing\\normal.txt", "w", encoding="utf8", errors='replace') as file:
        file.write(text[1:])

    # save the content of the normal train
    text = ''
    for file in train:
        with open(dir + "\\" + file, "r+", encoding="utf8", errors='replace') as f:
            dic = json.load(f)
            text += '\n' + open(dir + "\\" + dic["file_id"], "r", errors='replace', encoding='utf8').read().replace('\n', ' ')
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

    result = re.findall(r'\w*חרא\w*', text)

    tup = [(word, result.count(word)) for word in set(result)]
    for word in tup:
        print(word[0] + ': ' + str(word[1]))


if __name__ == "__main__":
    path = r"C:\Users\user\Documents\מחקר - ד''ר קרנר\מחקר הפרעות נפשיות\אנגלית\all"
    create_dataset(path, output=r"C:\Users\user\Documents\test\100 מול 100 אנגלית")

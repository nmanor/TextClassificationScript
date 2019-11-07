import os
import pickle
from datetime import datetime
import sys
import xlsxwriter

from os import path

from PIL import Image

from confusion_matrix import plot_confusion_matrix
from global_parameters import GlobalParameters
from precision_recall_curve import plot_precision_recall_curve
from roc_curve import plot_roc_curve
from stylistic_features import text_language


def corpus_name():
    """
    return the name of the corpus (for example: Corpus of 1000 female & 600 male in English)
    :param train: the train (or the test) text
    :param train_labels: the labels of the train corpus
    :param test_labels: the labels of the test corpus
    :return: string, the name of the corpus
    """
    glbs = GlobalParameters()
    string = "Corpus of "
    labels = glbs.LABELS
    dic = {}

    for label in labels:
        # usually the labels contain the file format at the end (file.txt)
        label = label.split('.')[0]
        if label in dic:
            dic[label] += 1
        else:
            dic[label] = 1

    for label, number in dic.items():
        string += str(number) + " " + label + ", "

    language = text_language(glbs.TRAIN_DATA[0])
    string = string[:-2] + " in " + language[0].upper() + language[1:]

    # Replace the last , with &
    for i in range(1, len(string)):
        if string[-i] == ',':
            string = string[:-i] + ' &' + string[-i+1:]
            break

    return string


def open_pickle_content(pickle_file_path):
    with open(pickle_file_path, 'rb') as file:
        return pickle.load(file)


def new_write_file_content(pickle_file_path, measure, results_path):
    # Setup the path and the name of the file
    dataset_name = corpus_name()
    pickle_file_content = open_pickle_content(pickle_file_path)
    file_path = results_path + "\\" + measure.upper().replace('_', ' ') + " for " + dataset_name + '.xlsx'

    # Create an new Excel file and add a worksheet.
    workbook = xlsxwriter.Workbook(file_path)
    worksheet = workbook.add_worksheet()

    # Write titels
    extra_big = workbook.add_format({'bold': True, 'font_size': 17, 'underline': True})
    big = workbook.add_format({'bold': True, 'font_size': 12})
    worksheet.write('A2', dataset_name, big)
    worksheet.write('A3', 'The classification results are shown in the table below in percentages')

    # Write general data
    bold_gray = workbook.add_format({'bold': True, 'font_color': 'gray'})
    gray = workbook.add_format({'font_color': 'gray'})
    worksheet.write('A5', 'General information about the classification software', bold_gray)
    now = datetime.now()
    worksheet.write('A6', 'Issue date: ' + now.strftime("%d/%m/%Y %H:%M:%S"), gray)
    version = str(sys.version_info[0]) + '.' + str(sys.version_info[1]) + '.' + str(sys.version_info[2])
    worksheet.write('A7', 'Python version: Python ' + version, gray)
    worksheet.write('A8', 'Python classification libraries: keras, sklearn, tensorflow, VADAR from nltk, WordCloud', gray)

    # Write normalization
    worksheet.write('A10', 'Pre Processing', bold_gray)
    worksheet.write('A11', "C - Spelling Correction", gray)
    worksheet.write('A12', "L - Lowercase", gray)
    worksheet.write('A13', "H - HTML tags", gray)
    worksheet.write('A14', "P - Punctuations", gray)
    worksheet.write('A15', "R - Repeated chars", gray)
    worksheet.write('A16', "T - Stemming", gray)
    worksheet.write('A17', "M - Lemmatizer", gray)

    # Write learning methods
    worksheet.write('H10', 'Learning methods', bold_gray)
    worksheet.write('H11', "svc  - Linear SVC", gray)
    worksheet.write('H12', "rf      - Random Forest", gray)
    worksheet.write('H13', "mlp  - Multilayer Perceptron", gray)
    worksheet.write('H14', "lr      - Logistic Regression", gray)
    worksheet.write('H15', "mnb - Multinomial Naive Bayes", gray)
    worksheet.write('H16', "rnn   - Recurrent Neural Network", gray)

    # Write stop words option
    worksheet.write('D10', 'Stop Words Options', bold_gray)
    worksheet.write('D11', "E - English stop words", gray)
    worksheet.write('D12', "H - Hebrew stop words", gray)
    worksheet.write('D13', "X - Extended Hebrew stop words", gray)

    # Write stylistic features option
    worksheet.write('F19', 'Stylistic Features Options', bold_gray)
    worksheet.write('F20', "CC - chars count", gray)
    worksheet.write('F21', "WC - words count", gray)
    worksheet.write('F22', "SC - sentence count", gray)
    worksheet.write('F23', "EMC - exclamation mark (!) count", gray)
    worksheet.write('F24', "QSMC - question mark (?) count", gray)
    worksheet.write('F25', "SCC - special characters (@, #, $, &, *, %, ^) count", gray)
    worksheet.write('F26', "QTMC - quotation mark (\", ') count", gray)
    worksheet.write('F27', "ALW - average letters in words", gray)
    worksheet.write('F28', "ALS - average letters in sentence", gray)
    worksheet.write('F29', "AWS - average words in sentence", gray)
    worksheet.write('F30', "AWL - average words length", gray)
    worksheet.write('F31', "IE - increasing expressions", gray)
    worksheet.write('F32', "DE - doubt expressions", gray)
    worksheet.write('F33', "NW - negative terms", gray)
    worksheet.write('F34', "PW - positive terms", gray)
    worksheet.write('F35', "TE - time expressions", gray)
    worksheet.write('F36', "EE - emotion expressions", gray)
    worksheet.write('I20', "FPE - first person expressions", gray)
    worksheet.write('I21', "SPE - second person expressions", gray)
    worksheet.write('I22', "TPE - third person expressions", gray)
    worksheet.write('I23', "INE - inclusion expressions", gray)
    worksheet.write('I24', "P1 - expressions form power 1", gray)
    worksheet.write('I25', "P2 - expressions form power 2", gray)
    worksheet.write('I26', "P3 - expressions form power 3", gray)
    worksheet.write('I27', "PM1 - expressions form power -1", gray)
    worksheet.write('I28', "PM2 - expressions form power -2", gray)
    worksheet.write('I29', "PM3 - expressions form power -3", gray)
    worksheet.write('I30', "PM4 - expressions form power -4", gray)
    worksheet.write('I31', "AP - expressions form all the powers", gray)
    worksheet.write('I32', "TOPA1 - Enable all features on the 1'st trimester of the text", gray)
    worksheet.write('I32', "TOPA2 - Enable all features on the 2'nd trimester of the text", gray)
    worksheet.write('I33', "TOPA3 - Enable all features on the 3'rd trimester of the text", gray)
    worksheet.write('I34', "TOPB1 - Enable all features on the first ten words of the text", gray)
    worksheet.write('I35', "TOPB2 - Enable all features on the text without 10 first and last 10 words", gray)
    worksheet.write('I36', "TOPB3 - Enable all features on the last ten words of the text", gray)

    # Write the result
    row = 40
    kind = {'w': 'Words', 'c': 'Chars'}
    ngrams = {'1': 'Unigrams', '2': 'Bigrams', '3': 'Trigrams'}
    tf = {'tf': 'TF', 'tfidf': 'TF-IDF'}
    methods = {'svc': 8, 'rf': 9, 'mlp': 10, 'lr': 11, 'mnb': 12, 'rnn': 13}
    maxes = {'svc': [[0, 0, 0]], 'rf': [[0, 0, 0]], 'mlp': [[0, 0, 0]], 'lr': [[0, 0, 0]], 'mnb': [[0, 0, 0]], 'rnn': [[0, 0, 0]]}
    best = [[0, 0, 0]]
    image_num = 0
    for key in sorted(pickle_file_content):
        value = pickle_file_content[key]

        # N-Grams data
        cell_format = workbook.add_format()
        cell_format.set_text_wrap()
        cell_format.set_align('vcenter')
        cell_format.set_align('center')
        features = value['featurs']
        count = ''
        type = ''
        tfidf = ''
        grams = ''
        skips = ''
        for feature in features:
            feature = feature.split('_')
            count += feature[1] + '\n'
            type += kind[feature[2]] + '\n'
            tfidf += tf[feature[3]] + '\n'
            grams += ngrams[feature[4]] + '\n'
            skips += feature[5] + '\n'
        worksheet.write(row, 0, str(count[:-1]), cell_format)
        worksheet.write(row, 1, type[:-1], cell_format)
        worksheet.write(row, 2, grams[:-1], cell_format)
        worksheet.write(row, 3, tfidf[:-1], cell_format)
        worksheet.write(row, 4, skips[:-1], cell_format)

        # Stylistic Features data
        stylistic_features = ''
        for styl_feature in value['stylistic_features']:
            stylistic_features += styl_feature.upper() + '  '
        worksheet.write(row, 5, stylistic_features[:-2], cell_format)

        # Pre Processing and Stop Words data
        cell_format = workbook.add_format()
        cell_format.set_align('center')
        cell_format.set_align('vcenter')
        normalization = ""
        stopwords = ""
        for char in value['normalization']:
            if char in "sbx":
                stopwords += char.replace('s', 'E').replace('b', 'H').replace('x', 'X')
            else:
                normalization += char.upper()
        if normalization == "":
            normalization = "NONE"
        if stopwords == "":
            stopwords = "NONE"
        worksheet.write(row, 6, normalization, cell_format)
        worksheet.write(row, 7, stopwords, cell_format)

        # ML methods and result data
        for method, result in value['results'].items():
            # confusion matrix
            if not isinstance(result, float):
                title = measure + str(image_num)
                if measure == "confusion_matrix":
                    plot_confusion_matrix(result, results_path, title=title)
                    worksheet.set_column(methods[method], methods[method], 40)
                    worksheet.set_row(row, 140)
                elif measure == "roc_curve":
                    plot_roc_curve(result, results_path, method, title=title)
                    worksheet.set_column(methods[method], methods[method], 50)
                    worksheet.set_row(row, 225)
                elif measure == "precision_recall_curve":
                    plot_precision_recall_curve(result, results_path, title=title)
                    worksheet.set_column(methods[method], methods[method], 47)
                    worksheet.set_row(row, 215)
                worksheet.insert_image(row, methods[method], results_path + "\\" + title + ".jpg")
                image_num += 1
                continue

            val = float('{0:.4g}'.format(result*100))

            worksheet.write(row, methods[method], val, cell_format)
            # Check if val bigger then max
            new = [row, methods[method], val]
            for num in maxes[method]:
                if val > num[2]:
                    num[0] = row
                    num[1] = methods[method]
                    num[2] = val
                if val == num[2] and new not in maxes[method]:
                    maxes[method] += [new]
            for num in best:
                if val > num[2]:
                    num[0] = row
                    num[1] = methods[method]
                    num[2] = val
                if val == num[2] and new not in best:
                    best += [new]
        row += 1

    worksheet.write('A19', 'Colors', bold_gray)
    good = workbook.add_format({'bold': True, 'font_color': 'blue'})
    good.set_align('center')
    good.set_align('vcenter')
    for _, method in maxes.items():
        for val in method:
            worksheet.write(val[0], val[1], val[2], good)
    good = workbook.add_format({'font_color': 'blue'})
    worksheet.write('A20', 'The best result of the learning method', good)
    good = workbook.add_format({'bold': True, 'font_color': 'red'})
    good.set_align('center')
    good.set_align('vcenter')
    for val in best:
        worksheet.write(val[0], val[1], val[2], good)
    good = workbook.add_format({'font_color': 'red'})
    worksheet.write('A21', 'The best result in all classification', good)
    bold = workbook.add_format({'bold': True})
    worksheet.write('A39', 'Results', bold)
    worksheet.add_table("A40:N" + str(row), {'columns': [{'header': 'Number'},
                                                {'header': 'Type'},
                                                {'header': 'N-GRAMS'},
                                                {'header': 'TF'},
                                                {'header': 'Skips'},
                                                {'header': 'Stylistic Features'},
                                                {'header': 'Pre Processing'},
                                                {'header': 'Stop Words'},
                                                {'header': 'svc'},
                                                {'header': 'rf'},
                                                {'header': 'mlp'},
                                                {'header': 'lr'},
                                                {'header': 'mnb'},
                                                {'header': 'rnn'}],
                                              'style': 'Table Style Light 8'})

    worksheet.write('A1', 'Classification results: ' + measure.replace("_", " "), extra_big)

    workbook.close()

    # Delete the images of the non integer measures
    for file in os.listdir(results_path):
        if file.endswith('.jpg'):
            os.remove(results_path + "\\" + file)



if __name__ == '__main__':
    print(open_pickle_content(r"C:\Users\user\Documents\test\results\precision_score.pickle"))
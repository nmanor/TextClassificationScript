import os
import pickle
import random
from array import array
from keras import Sequential
from scipy.sparse import hstack, vstack
from keras.layers import (
    Embedding,
    SpatialDropout1D,
    Conv1D,
    MaxPooling1D,
    LSTM,
    Dense,
    Dropout,
)
from statistics import mean
import matplotlib.pyplot as plt
from keras_preprocessing.sequence import pad_sequences
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from tensorflow import one_hot

from confusion_matrix import accuracy_confusion_matrix
from global_parameters import print_message, GlobalParameters
from help_functions import write_result
from precision_recall_curve import precision_recall
from roc_curve import roc_curve_data
from sklearn.feature_selection import (
    chi2,
    mutual_info_classif,
    f_classif,
    f_regression,
    RFECV,
    mutual_info_regression,
    SelectKBest,
    SelectFromModel,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from new_xlsx_file import write_info_gain, write_sfm
from classification import classify
from main import add_results, add_results_glbs

glbs = GlobalParameters()

methods = {
    "svc": LinearSVC(),
    "rf": RandomForestClassifier(),
    "lr": LogisticRegression(),
    "mnb": MultinomialNB(),
}

selection_type = {
    "chi2": chi2,
    "mir": mutual_info_regression,
    "mic": mutual_info_classif,
    "fc": f_classif,
    "rfecv": RFECV,
    "sfm": SelectFromModel,
    "fr": f_regression,
}


def get_selection_list(selection, X, y):
    if selection == "mir" or selection == "mic":
        return selection_type[selection](X, y, n_neighbors=371)
    return selection_type[selection](X, y)


def select_k_best(selection, k):
    return SelectKBest(selection_type[selection], k=k)


def get_recursive(features, labels):
    ranking = {}
    for key, method in methods.items():
        recursive = RFECV(
            method, step=1, cv=[(range(134), range(134, 200))], scoring="accuracy"
        )
        train_features = recursive.fit(features, labels)
        ranking[key] = train_features.ranking_
    return ranking


def select_rfecv_sfm(selection, features, labels):
    if selection[0] == "rfecv":
        for key, method in methods.items():
            recursive = RFECV(
                method, step=1, cv=[(range(134), range(134, 200))], scoring="accuracy"
            )
            recursive.fit(features, labels)
            # Plot number of features VS. cross-validation scores
            plt.figure()
            plt.xlabel("Number of features selected")
            plt.ylabel("accuracy score" + key + " (nb of correct classifications)")
            plt.plot(range(1, len(recursive.grid_scores_) + 1), recursive.grid_scores_)
            plt.savefig(glbs.RESULTS_PATH + "\\" + key + ".jpg", bbox_inches="tight")
    if selection[0] == "sfm":
        score = {}
        for key, method in methods.items():
            sfm = SelectFromModel(method, max_features=int(selection[1]))
            train_new = sfm.fit_transform(features[0], labels[0])
            test_new = sfm.transform(features[1])
            clf = method
            clf.fit(train_new, labels[0])
            pred = clf.predict(test_new)
            acc = accuracy_score(labels[1], pred)
            score[key] = acc
        write_sfm(score)


def selectionHalfMethod(X, y, all_features):
    glbs = GlobalParameters()
    filename = glbs.FILE_NAME
    results = {}
    # nxt = (glbs.SELECTION[0][0], int(glbs.SELECTION[0][1]))
    nxt = (glbs.SELECTION[0][0], int(glbs.SELECTION[0][1]))
    max_last_result = 0
    bottom = (0, 0)
    top = nxt
    while top != bottom:
        max_nxt_result = 0
        print_message(nxt[0])
        print_message(nxt[1])
        glbs.FILE_NAME = glbs.FILE_NAME + str(nxt[1])
        select = select_k_best(nxt[0], int(nxt[1]))
        glbs.FEATURE_MODEL[1] = select
        results[glbs.FILE_NAME] = classify(X, y, glbs.K_FOLDS, glbs.ITERATIONS)
        for method in results[glbs.FILE_NAME].items():
            if mean(method[1]["accuracy"]) > max_nxt_result:
                max_nxt_result = mean(method[1]["accuracy"])
        results = add_results(results, glbs, nxt)
        if max_nxt_result >= max_last_result:
            top = nxt
            if bottom[1] == 0:
                nxt = (nxt[0], int(int(nxt[1]) / 2))
            if bottom[1] != 0:
                nxt = (nxt[0], int((int(nxt[1]) + bottom[1]) / 2))
            max_last_result = max_nxt_result
        elif max_nxt_result < max_last_result:
            bottom = nxt
            nxt = (nxt[0], int((top[1] + bottom[1]) / 2))
        glbs.SELECTION[0] = nxt
        if bottom[1] - top[1] == -1 and bottom == nxt:
            break
    glbs.FILE_NAME = filename
    add_results_glbs(results, glbs)


def get_selected_features(X, y, all_features):
    for selection in glbs.SELECTION:
        if glbs.PRINT_SELECTION:
            # Shuffle  the inner order of the posts within each fold
            F = X[:160]
            t = y[:160]
            le = LabelEncoder()
            le.fit(t)
            t = le.transform(t)
            F = all_features.fit_transform(F)

            selection_list = get_selection_list(selection[0], F, t)
            ziped = []
            try:
                ziped = zip(
                    all_features.get_feature_names(),
                    selection_list[0],
                    selection_list[1],
                )
            except:
                ziped = zip(all_features.get_feature_names(), selection_list)
            write_info_gain(ziped, str(selection[0]))
        if selection[0] in selection_type.keys():
            select = select_k_best(selection[0], int(selection[1]))

            glbs.FEATURE_MODEL.append(select)
            if glbs.SELECTION_HALF:
                selectionHalfMethod(X, y, all_features)


if __name__ == "__main__":
    pass

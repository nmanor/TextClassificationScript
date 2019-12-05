import os
import pickle
from array import array
from keras import Sequential
from keras.layers import (
    Embedding,
    SpatialDropout1D,
    Conv1D,
    MaxPooling1D,
    LSTM,
    Dense,
    Dropout,
)
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
from features import get_data
from global_parameters import print_message, GlobalParameters
from help_functions import write_result
from precision_recall_curve import precision_recall
from roc_curve import roc_curve_data
from sklearn.feature_selection import (
    chi2,
    mutual_info_classif,
    f_classif,
    RFECV,
    mutual_info_regression,
    SelectKBest,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from new_xlsx_file import write_info_gain

glbs = GlobalParameters()

methods = {
    "svc": LinearSVC(),
    "rf": RandomForestClassifier(),
    "mlp": MLPClassifier(),
    "lr": LogisticRegression(),
    "mnb": MultinomialNB(),
}

selection_type = {
    "chi2": chi2,
    "mir": mutual_info_regression,
    "mic": mutual_info_classif,
    "fc": f_classif,
}


def get_selection(selection, features, labels):
    return selection_type[selection](features, labels)


def select_k_best(selection, k):
    return SelectKBest(selection_type[selection], k=k)


def get_featuregain(features, train_features, train_labels, test_featurs, test_labels):
    results = {}
    result = []
    select = SelectKBest(mutual_info_regression, k=125)
    select.fit(train_features, train_labels)
    train_features = select.transform(train_features)
    test_featurs = select.transform(test_featurs)
    for classifier in glbs.METHODS:
        clf = methods[classifier]
        clf.fit(train_features, train_labels)
        prediction = clf.predict(test_featurs)
        decision = []
        try:
            decision = clf.decision_function(test_featurs)
        except:
            decision = clf.predict_proba(test_featurs)
            decision = decision[:, 1]
        result = get_results(test_labels, prediction, decision)

        del clf

        results[classifier] = result
    # return results
    # feat_labels = features.get_feature_names()

    # chi = chi2(train_features, train_labels)
    # write_info_gain(zip(feat_labels, chi[0]), "chi^2")

    # mir = mutual_info_regression(train_features, train_labels)
    # write_info_gain(zip(feat_labels, mir), "mutual_info_regresson")

    recursive = []
    for key, method in methods.items():
        if key == "mlp":
            continue
        recursive = RFECV(method, step=1, cv=StratifiedKFold(2), scoring="accuracy")
        train_features = recursive.fit(glbs.ALL_DATA, glbs.LABELS)

        print("Optimal number of features : %d" % recursive.n_features_)

        # Plot number of features VS. cross-validation scores
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("accuracy score" + key + " (nb of correct classifications)")
        plt.plot(range(1, len(recursive.grid_scores_) + 1), recursive.grid_scores_)
        plt.savefig(glbs.RESULTS_PATH + "\\" + key + ".jpg", bbox_inches="tight")
    # write_info_gain(zip(feat_labels, recursive.ranking_), "rfevc " + key)

    # mic = mutual_info_classif(train_features, train_labels)
    # write_info_gain(zip(feat_labels, mic), "mutual_info")

    # anova = f_classif(train_features, train_labels)
    # write_info_gain(zip(feat_labels, anova[0]), "ANOVA F-value")


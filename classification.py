# from keras import Sequential
# from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Conv1D, MaxPooling1D
# from keras.preprocessing import sequence
# from keras.preprocessing.text import Tokenizer
import json
import os
import pickle
import random

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
from pandas import np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from confusion_matrix import accuracy_confusion_matrix
from global_parameters import print_message, GlobalParameters
from precision_recall_curve import precision_recall
from roc_curve import roc_curve_data

glbs = GlobalParameters()

methods = {
    "svc": LinearSVC(),
    "rf": RandomForestClassifier(n_jobs=-1 if glbs.MULTIPROCESSING else None),
    "mlp": MLPClassifier(),
    "lr": LogisticRegression(n_jobs=-1 if glbs.MULTIPROCESSING else None),
    "mnb": MultinomialNB(),
}


def get_results(ts_labels, prediction, decision):
    measures = {
        "accuracy_score": accuracy_score(ts_labels, prediction),
        "f1_score": f1_score(ts_labels, prediction),
        "precision_score": precision_score(ts_labels, prediction),
        "recall_score": recall_score(ts_labels, prediction),
        "roc_auc_score": roc_auc_score(ts_labels, decision),
        "confusion_matrix": confusion_matrix(ts_labels, prediction),
        "roc_curve": roc_curve_data(ts_labels, decision),
        "precision_recall_curve": precision_recall(ts_labels, decision),
        "accuracy_&_confusion_matrix": accuracy_confusion_matrix(ts_labels, prediction),
    }

    return dict([(i, measures[i]) for i in glbs.MEASURE])
    # return accuracy_score(ts_labels, prediction)
    # return stats


def classify(X, y, k_fold, num_iteration=1):
    results = {}
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    print_message("Classifying")

    """def cross_validation(X, classifier, clf, i, k_fold, num_iteration, y):
        print_message("iteration " + str(i + 1) + "/" + str(num_iteration), 2)
        scores = cross_validate(clf, X, y, cv=k_fold, scoring=glbs.MEASURE)
        for measure in glbs.MEASURE:
            if measure in results[classifier].keys():
                results[classifier][measure] += list(scores['test_' + measure])
            else:
                results[classifier][measure] = list(scores['test_' + measure])"""

    for classifier in glbs.METHODS:
        print_message("running " + str(classifier), num_tabs=1)
        if classifier not in results.keys():
            results[classifier] = {}

        if classifier == "rnn":
            # clf = get_rnn_model(X)
            continue
        else:
            clf = methods[classifier]
        lst = []
        if glbs.SELECTION:
            lst.append(("feture", glbs.FEATURE_MODEL[0]))
            lst.append(("select", glbs.FEATURE_MODEL[1]))
            lst.append(("classifier", clf))
            clf = Pipeline(lst)

        else:
            clf = make_pipeline(glbs.FEATURE_MODEL[0], clf)

        if glbs.MULTIPROCESSING:
            n_jobs = -1
        else:
            n_jobs = None

        ####################################################################
        # Used for parameters tuning
        from sklearn.model_selection import RandomizedSearchCV
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=1000, stop=2000, num=10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10, 15]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4, 6]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {'classifier__n_estimators': n_estimators,
                       'classifier__max_features': max_features,
                       'classifier__max_depth': max_depth,
                       'classifier__min_samples_split': min_samples_split,
                       'classifier__min_samples_leaf': min_samples_leaf,
                       'classifier__bootstrap': bootstrap}
        print(random_grid)

        # Use the random grid to search for best hyperparameters
        # Random search of parameters, using 3 fold cross validation,
        # search across 100 different combinations, and use all available cores
        rf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, n_iter=100, cv=k_fold, verbose=2,
                                       random_state=42, n_jobs=-1)

        print(clf.get_params().keys())
        # Fit the random search model
        rf_random.fit(X, y)

        print(rf_random.best_params_)
        clf = rf_random.best_estimator_
        with open("best_estimator.json", "w") as file:
            json.dump(rf_random.best_params_, file, indent=6)

        """best_random = rf_random.best_estimator_
        scores = cross_validate(best_random, X, y, cv=k_fold, scoring=glbs.MEASURE, n_jobs=n_jobs)
        random_accuracy = numpy.mean(list(scores["test_accuracy"]))

        base_model = clf
        scores = cross_validate(base_model, X, y, cv=k_fold, scoring=glbs.MEASURE, n_jobs=n_jobs)
        base_accuracy = numpy.mean(list(scores["test_accuracy"]))

        print('Base accuracy: {:0.2f}%.'.format(100 * base_accuracy))
        print('Random accuracy: {:0.2f}%.'.format(100 * random_accuracy))

        print('Improvement of {:0.2f}%.'.format(100 * (random_accuracy - base_accuracy) / base_accuracy))

        stop = "Put breakpoint here"""""
        ####################################################################

        for i in range(num_iteration):
            # Shuffle the inner order of the posts within each fold
            Xy = list(zip(X, y))
            splited = []
            len_l = len(Xy)
            for j in range(k_fold):
                start = int(j * len_l / k_fold)
                end = int((j + 1) * len_l / k_fold)
                splited.append(Xy[start:end])
            Xy = []
            for fold in splited:
                random.Random(num_iteration).shuffle(fold)
                Xy += fold
            X, y = zip(*Xy)

            print_message("iteration " + str(i + 1) + "/" + str(num_iteration), 2)
            scores = cross_validate(
                clf, X, y, cv=k_fold, scoring=glbs.MEASURE, n_jobs=n_jobs
            )
            for measure in glbs.MEASURE:
                if measure in results[classifier].keys():
                    results[classifier][measure] += list(scores["test_" + measure])
                else:
                    results[classifier][measure] = list(scores["test_" + measure])

        del clf
    return results


def get_rnn_model(tr_features):
    top_words = tr_features.shape[1]
    model_lstm = Sequential()
    model_lstm.add(
        Embedding(input_dim=top_words, output_dim=256, input_length=top_words)
    )
    model_lstm.add(SpatialDropout1D(0.3))
    model_lstm.add(LSTM(256, dropout=0.3, recurrent_dropout=0.3))
    model_lstm.add(Dense(256, activation="relu"))
    model_lstm.add(Dropout(0.3))
    model_lstm.add(Dense(5, activation="softmax"))
    model_lstm.compile(
        loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"]
    )
    return model_lstm


def get_rnn_model2(tr_features):
    top_words = tr_features.shape[1]
    # create the model
    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=top_words))
    model.add(SpatialDropout1D(0.2))
    model.add(Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100, recurrent_dropout=0.2, dropout=0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def gen_file_path():
    name = ""
    for feature in glbs.FEATURES:
        name += feature.upper()
        name += "@"
    name += glbs.NORMALIZATION + "@"
    for method in glbs.METHODS:
        name += method.upper()
        name += "@"
    folder_path = "\\".join(glbs.RESULTS_PATH.split("\\")[:-1]) + "\\temp_backups"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path + "\\" + name + ".pickle"


def save_backup_file(data, path):
    with open(path, "wb+") as file:
        pickle.dump(data, file)


def load_backup_file(path):
    with open(path, "rb") as file:
        return pickle.load(file)


if __name__ == "__main__":
    # extract data
    pass

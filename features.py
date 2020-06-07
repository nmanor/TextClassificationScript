import os
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

# region helpful functions
from global_parameters import print_message, GlobalParameters
from skipgrams_vectorizer import SkipGramVectorizer
from stylistic_features import get_stylistic_features_vectorizer

glbs = GlobalParameters()


def read_dataset(path):
    data = []
    for category in sorted(os.listdir(path)):
        with open(
            path + "\\" + category, "r+", encoding="utf8", errors="ignore"
        ) as read:
            for example in read:
                record = example.rstrip("\n")
                data.append((record, category))
    return data


def is_ngrams(feature):
    return feature.startswith("ngrams")


def extract_ngrams_args(feature):
    parts = feature.split("_")
    count = int(parts[1])
    type = parts[2]
    tfidf = parts[3]
    n = int(parts[4])
    k = int(parts[5])
    if type == "w":
        type = "word"
    elif type == "c":
        type = "char"
    return count, tfidf, type, n, k


def split_train(split, tr_labels, train):
    data = list(zip(train, tr_labels))
    random.shuffle(data)
    test = data[: int(split * len(data))]
    train = data[int(split * len(data)) :]
    train, tr_labels = zip(*train)
    test, ts_labels = zip(*test)
    return train, tr_labels, test, ts_labels


def get_vectorizer(feature):
    count, tfidf, type, n, k = extract_ngrams_args(feature)
    if tfidf == "tfidf":
        tfidf = True
    else:
        tfidf = False

    if k <= 0:
        vectorizer = TfidfVectorizer(
            max_features=count,
            analyzer=type,
            ngram_range=(n, n),
            lowercase=False,
            use_idf=tfidf,
            min_df=3,
            stop_words=glbs.STOP_WORDS,
        )

    else:
        vectorizer = SkipGramVectorizer(
            max_features=count,
            analyzer=type,
            n=n,
            k=k,
            lowercase=False,
            min_df=3,
            stop_words=glbs.STOP_WORDS,
        )

    return vectorizer


def add_feature(feature_dict, feature_name, feature):
    feature_dict.append((feature_name, feature))
    return feature_dict


def extract_features(dataset_dir):
    print_message("Extracting Features")

    X, y = get_data(dataset_dir)
    glbs.LABELS = y
    glbs.DATASET_DATA = X

    ########################################
    # X, y = zip(*list(zip(X, y))[:160])
    # from help_functions import get_fetuer_by_DF

    # get_fetuer_by_DF(X)
    ########################################

    feature_lst = []
    # add all the N-Grams feature to the list
    for feature in glbs.FEATURES:
        if is_ngrams(feature):
            vectorizer = get_vectorizer(feature)
            feature_lst = add_feature(feature_lst, feature, vectorizer)
    # add all the stylistic features to the list
    for feature in glbs.STYLISTIC_FEATURES:
        vectorizers = get_stylistic_features_vectorizer(feature)
        for i in range(len(vectorizers)):
            feature_lst = add_feature(feature_lst, feature + str(i), vectorizers[i])

    # convert the list to one vectoriazer using FeatureUnion
    if glbs.MULTIPROCESSING:
        n_jobs = -1
    else:
        n_jobs = None
    all_features = FeatureUnion(feature_lst, n_jobs=n_jobs)

    glbs.FEATURE_MODEL.append(all_features)

    all_features.fit(X, y)
    glbs.NUM_OF_FEATURE = len(all_features.get_feature_names())

    if glbs.SELECTION:
        from feature_selction import get_selected_features

        get_selected_features(X, y, all_features)

    return X, y


def get_data(dataset_dir):
    dataset_data = read_dataset(dataset_dir)
    dataset_data.sort(key=lambda doc: doc[0])
    random.Random(4).shuffle(dataset_data)
    X, y = zip(*dataset_data)
    return X, y

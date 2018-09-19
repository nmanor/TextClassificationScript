import os
import random

import numpy as np

import global_parameters
from pickle import dump, load
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# region helpful functions
import system_config
from global_parameters import print_message
from skipgrams_vectorizer import SkipGramVectorizer


def read_dataset(path):
	data = []
	for category in os.listdir(path):
		with open(path + "\\" + category, 'r+', encoding='utf8', errors='ignore') as read:
			for example in read:
				record = example.rstrip('\n')
				data.append((record, category))
	return data


def is_ngrams(feature):
	return feature.startswith('ngrams')


def extract_ngrams_args(feature):
	parts = feature.split("_")
	count = int(parts[1])
	type = parts[2]
	tfidf = parts[3]
	n = int(parts[4])
	k = int(parts[5])
	if type == 'w':
		type = 'word'
	elif type == 'c':
		type = 'char'
	return count, tfidf, type, n, k


def fuse_features(f1, f2):
	return np.hstack((f1.toarray(), f2.toarray()))


def split_train(split, tr_labels, train):
	data = list(zip(train, tr_labels))
	random.shuffle(data)
	test = data[:int(split * len(data))]
	train = data[int(split * len(data)):]
	train, tr_labels = zip(*train)
	test, ts_labels = zip(*test)
	return train, tr_labels, test, ts_labels


# endregion

def extract_ngrams(train_data, test_data, feature):
	# data = [(content,label)]
	# extract ngrams
	count, tfidf, type, n, k = extract_ngrams_args(feature)
	if k <= 0:
		vectorizer = CountVectorizer(max_features=count, analyzer=type, ngram_range=(n, n), lowercase=False)
	else:
		vectorizer = SkipGramVectorizer(max_features=count, analyzer=type, n=n, k=k, lowercase=False)
	x = vectorizer.fit_transform(train_data)
	y = vectorizer.transform(test_data)
	if tfidf == 'tfidf':
		tfidf = TfidfTransformer()
		x = tfidf.fit_transform(x.toarray())
		y = tfidf.transform(y.toarray())
	return x, y


def extract_features(train_dir, test_dir=''):
	print_message("Extracting Features")

	vals = are_features_saved()
	if vals[0] is not False:
		print_message("Found saved features, Loading...")
		return vals

	train_data, train_labels, test_data, test_labels = get_data(test_dir, train_dir)

	for feature in global_parameters.FEATURES:
		if is_ngrams(feature):
			train_features, test_features = extract_ngrams(train_data, test_data, feature)
	# add more features here, don't forget to fuse them with fuse_features()
	save_features(train_features, train_labels, test=False)
	save_features(test_features, test_labels, test=True)
	return train_features, train_labels, test_features, test_labels


def get_data(test_dir, train_dir):
	train_data = read_dataset(train_dir)
	train_data, train_labels = zip(*train_data)
	if global_parameters.TEST_DIR == '':
		train_data, train_labels, test_data, test_labels = split_train(system_config.TEST_SPLIT, train_labels,
																	   train_data)
	else:
		test_data = read_dataset(test_dir)
		test_data, test_labels = zip(*test_data)
	return train_data, train_labels, test_data, test_labels


def save_features(data, labels, test):
	name = gen_file_name(test)
	with open(global_parameters.OUTPUT_DIR + '\\' + name + ".pickle", 'wb+') as file:
		dump((data, labels), file)


def gen_file_name(test):
	name = ''
	for feature in global_parameters.FEATURES:
		name += feature.upper()
		name += '@'
	name += global_parameters.NORMALIZATION + "@"
	if test:
		name += "TEST"
	else:
		name += 'TRAIN'
	return name


def are_features_saved():
	name = gen_file_name(test=False)
	test_name = gen_file_name(test=True)
	file_path = global_parameters.OUTPUT_DIR + '\\' + name + ".pickle"
	test_file_path = global_parameters.OUTPUT_DIR + '\\' + test_name + ".pickle"
	if os.path.exists(file_path) and os.path.exists(test_file_path):
		with open(file_path, 'rb') as file:
			train_data, train_labels = load(file)
		with open(test_file_path, 'rb') as file:
			test_data, test_labels = load(file)
		return train_data, train_labels, test_data, test_labels
	return False,


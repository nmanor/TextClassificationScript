import random

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
import global_parameters
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Conv1D, MaxPooling1D

from scipy import sparse
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from features import get_data
from global_parameters import print_message

methods = {"svc": LinearSVC(),
		   "rf": RandomForestClassifier(),
		   "mlp": MLPClassifier(),
		   "lr": LogisticRegression(),
		   "mnb": MultinomialNB()
		   }


def classify(train, tr_labels, test, ts_labels, num_iteration=1):
	results = {}
	le = LabelEncoder()
	le.fit(tr_labels)
	ts_labels = le.transform(ts_labels)
	tr_labels = le.transform(tr_labels)
	print_message("Classifying")
	for classifier in global_parameters.METHODS:
		print_message("running " + str(classifier), num_tabs=1)
		average_acc = 0
		for i in range(num_iteration):
			if classifier == 'rnn':
				clf = get_rnn_model(train)
				clf.fit(train, tr_labels, epochs=3, batch_size=64)
				scores = clf.evaluate(test, ts_labels, verbose=0)
				average_acc += scores[1]
			else:
				clf = methods[classifier]
				clf.fit(train, tr_labels)
				prediction = clf.predict(test)
				average_acc += accuracy_score(ts_labels, prediction)
			del clf
		average_acc = average_acc / num_iteration
		results[classifier] = average_acc
	return results


def get_rnn_model(tr_features):
	top_words = tr_features.shape[1]
	# create the model
	embedding_vecor_length = 32
	model = Sequential()
	model.add(Embedding(top_words, embedding_vecor_length, input_length=top_words))
	model.add(SpatialDropout1D(0.2))
	model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
	model.add(LSTM(100, recurrent_dropout=0.2, dropout=0.2))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


if __name__ == '__main__':
	#extract data
	print("Reading Data...")
	#train_dir = r"C:\Users\yairi\Desktop\AGs News\dataset\training"
	train_dir = r"C:\Personal\Studies\Research\Data_Mining\Datasets\Mini20Newsgroups_processed\Mini-20Newsgroups_processed"
	#test_dir = r"C:\Users\yairi\Desktop\AGs News\dataset\testing"
	global_parameters.TEST_DIR = ''
	tr, tr_labels, ts, ts_labels = get_data('', train_dir)
	#tokenize data
	print("Tokenizing...")
	max_words = 1000
	max_len = 150
	tok = Tokenizer(num_words=max_words)
	tok.fit_on_texts(tr)
	sequences = tok.texts_to_sequences(tr)
	sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)
	#process test data
	test_sequences = tok.texts_to_sequences(ts)
	test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen=max_len)
	#encode labels
	print("Encoding Labels...")
	le = LabelEncoder()
	tr_labels = le.fit_transform(tr_labels)
	tr_labels = tr_labels.reshape(-1, 1)
	ts_labels = le.transform(ts_labels)
	ts_labels = ts_labels.reshape(-1, 1)
	#create rnn
	print("Creating Rnn...")
	embedding_vecor_length = 50
	model = Sequential()
	model.add(Embedding(max_words, embedding_vecor_length, input_length=max_len))
	model.add(SpatialDropout1D(0.2))
	model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
	model.add(LSTM(100, recurrent_dropout=0.2, dropout=0.2))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# run rnn
	print("Running...")
	rnn = model
	rnn.fit(sequences_matrix, tr_labels, epochs=100, batch_size=1000)
	# evalute
	print("Evaluating...")
	scores = rnn.evaluate(test_sequences_matrix, ts_labels, verbose=0)
	print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(scores[0], scores[1]))

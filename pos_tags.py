import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder


def features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }


def untag(tagged_sentence):
    return [w for w, t in tagged_sentence]


def transform_to_dataset(tagged_sentences):
    """
    Split tagged sentences to X and y datasets and append some basic features.
    :param tagged_sentences: a list of POS tagged sentences
    :param tagged_sentences: list of list of tuples (term_i, tag_i)
    :return:
    """
    X, y = [], []
    for pos_tags in tagged_sentences:
        for index, (term, class_) in enumerate(pos_tags):
            # Add basic NLP features for each sentence term
            X.append(features(untag(pos_tags), index))
            y.append(class_)
    return X, y


def tag_corpus(corpus):
    tagged_corpus = []
    for doc in corpus:
        tagged_corpus += [nltk.pos_tag(nltk.word_tokenize(doc))]
    return tagged_corpus


class POSTagsTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.init_transformer = None

    def fit(self, X, y=None):
        """All SciKit-Learn compatible transformers and classifiers have the
        same interface. `fit` always returns the same object."""
        self.init_transformer = DictVectorizer()
        return self

    def transform(self, X):
        tagged_corpus = tag_corpus(X)
        X, y = transform_to_dataset(tagged_corpus)

        # Fit our DictVectorizer with our set of features
        dict_vectorizer = DictVectorizer(sparse=False)
        dict_vectorizer.fit(X)
        # Convert dict features to vectors
        X = dict_vectorizer.transform(X)

        # Fit LabelEncoder with our list of classes
        label_encoder = LabelEncoder()
        label_encoder.fit(y)
        # Encode class values as integers
        y = label_encoder.transform(y)
        # Convert integers to dummy variables (one hot encoded)
        from keras.utils import np_utils
        y = np_utils.to_categorical(y)

        return X, y

    def get_feature_names(self):
        """Array mapping from feature integer indices to feature name"""
        return self.init_transformer.get_feature_names()


def get_pos_transformer():
    return POSTagsTransformer();


if __name__ == "__main__":
    text = "All SciKit-Learn compatible transformers and classifiers have the same interface. `fit` always returns the same object."
    vec = POSTagsTransformer()
    print(vec.fit_transform([text]))

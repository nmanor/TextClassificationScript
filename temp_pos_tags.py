# -----------------------------------------------------------------------------------------
# Part Of Speech (POS) tags
import re

import nltk
import requests
from scipy.sparse import csr_matrix
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import FeatureUnion


def text_language(text):
    """
    Determine if the language of the text is Hebrew or English
    :param text: the text that the function need to check
    :return: string, the language of the text - "hebrew" or "english"
    """
    hebrew = 0
    english = 0
    for char in text:
        if char in "אבגדהוזחטיכךלמםנסעפףצץקרשת":
            hebrew += 1
        elif char.lower() in "abcdefghijklmnopqrstuvwxyz":
            english += 1
    return {True: "hebrew", False: "english"}[hebrew > english]


def get_english_pos_tags_dict(post):
    """
    English version
    :param post: text to analyze
    :return: dict. the key is the POS and the value is the num of repeatoin in the post
    """
    tokens = nltk.word_tokenize(post)
    tags = nltk.pos_tag(tokens)
    post_dict = {}
    for _, pos in tags:
        if pos in post_dict:
            post_dict[pos] += 1
        else:
            post_dict[pos] = 1
    return post_dict


def get_hebrew_pos_tags_dict(post):
    """
    English version
    :param post: text to analyze
    :return: dict. the key is the POS and the value is the num of repeatoin in the post
    """
    request = {
        'token': 'J3cPRb90K5W0P53',
        'readable': False,
        'paragraph': post
    }
    # use the internet to get the result
    try:
        result = requests.post('https://hebrew-nlp.co.il/service/Morphology/Analyze', json=request).json()
    except:
        return {}
    post_dict = {}
    for sentence in result:
        for word in sentence:
            pos = word[0]['partOfSpeech']
            if pos in post_dict:
                post_dict[pos] += 1
            else:
                post_dict[pos] = 1
    return post_dict


def get_corpus_pos_tags(corpus):
    # determine the language of the corpus
    pos_function = {"hebrew": get_hebrew_pos_tags_dict, "english": get_english_pos_tags_dict}[text_language(corpus[0])]

    # get the POS tags for every post in the corpus
    all_posts_pos = {}
    for i, post in enumerate(corpus):
        all_posts_pos[i] = pos_function(post)

    # calculate all the kinds of POS in all the posts
    all_pos = []
    for _, pos_per_post in all_posts_pos.items():
        all_pos += list(pos_per_post.keys())

    # return all the result
    return all_posts_pos, all_pos


def calculate_pos_freq(post, post_num, pos, all_posts_pos):
    """
    :param post: the post to calculate
    :param post_num: the index of tha post in all_post_pos
    :param pos: the POS to calculate in the current post
    :param all_posts_pos:
    :return: the percentage of words in the type of the POS in the post
    """
    num_of_words = len(re.findall(r'\b\w[\w-]*\b', post.lower()))
    if pos in all_posts_pos[post_num]:
        return all_posts_pos[post_num][pos]/num_of_words;
    return 0


class InitTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, POS, all_posts_pos):
        self.POS = POS
        self.all_posts_pos = all_posts_pos

    def fit(self, X, y=None):
        """All SciKit-Learn compatible transformers and classifiers have the
        same interface. `fit` always returns the same object."""
        return self

    def transform(self, X):
        X_ = []
        for i, post in enumerate(X):
            X_ += [[calculate_pos_freq(post, i, self.POS, self.all_posts_pos)]]
        return csr_matrix(X_)

    def get_feature_names(self):
        """Array mapping from feature integer indices to feature name"""
        return [self.POS]


class POSTagsTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.init_transformer = None

    def fit(self, X, y=None):
        """All SciKit-Learn compatible transformers and classifiers have the
        same interface. `fit` always returns the same object."""
        feature_lst = []
        all_posts_pos, all_pos = get_corpus_pos_tags(X)
        for i, pos in enumerate(all_pos):
            feature_lst += [("pos" + pos + str(i), InitTransformer(pos, all_posts_pos))]
        self.init_transformer = FeatureUnion(feature_lst)
        return self

    def transform(self, X):
        return self.init_transformer.transform(X)

    def get_feature_names(self):
        """Array mapping from feature integer indices to feature name"""
        return self.init_transformer.get_feature_names()


def get_pos_transformer():
    return POSTagsTransformer();


if __name__ == '__main__':
    text = "Hello their! What a wonderful day!"
    tags = nltk.pos_tag(nltk.word_tokenize(text))
    t2 = nltk.BigramTagger(text)
    print(t2)

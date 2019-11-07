import re

import numpy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.sparse import csr_matrix
from scipy.sparse.compressed import _cs_matrix
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


# General help functions

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


def prevalence_rate(str, lst, length_relation=False):
    """
    :param str: the text that needs to be analyzed
    :param lst: list the words that need to check the prevalence of inside the text
    :param length_relation: do attach importance to the length of each words in the list
    :return: the percentage of repetition of the number of words in the list of all words in the text
    """
    orginal_str = str
    num = 0
    for word in sorted(set(lst), key=len, reverse=True):
        if length_relation:
            length = len(word.split(' '))
        else:
            length = 1
        num += str.lower().count(word) * length
        str = str.replace(word, '')
    return [num / len(re.findall(r'\b\w[\w-]*\b', orginal_str.lower()))]



# -----------------------------------------------------------------------------------------
# Quantitative features
# (Normalized in words and characters)

def chars_count(data):
    """
    1
    :param data: the corpus
    :return: list the number of characters in each post
    """
    return [[len(post)] for post in data]


def words_count(data):
    """
    2
    :param data: the corpus
    :return: list the number of words in each post
    """
    return [[len(re.findall(r'\b\w[\w-]*\b', post.lower()))] for post in data]


def sentence_count(data):
    """
    3
    :param data: the corpus
    :return: list the estimated number of sentences in each post
    """
    for post in data:
        post.replace('...', '.').replace('..', '.')
        post.replace('!!!', '!').replace('!!', '!')
        post.replace('???', '?').replace('??', '?')
    return [[len(re.split(r'[.!?]+', post))] for post in data]


def exclamation_mark_count(data):
    """
    4
    :param data: the corpus
    :return: list of number of repetitions of ! normalized by the number of characters in each post
    """
    return [[post.count('!')/len(post)] for post in data]


def question_mark_count(data):
    """
    5
    :param data: the corpus
    :return: list of number of repetitions of ? normalized by the number of characters in each post
    """
    return [[post.count('?')/len(post)] for post in data]


def special_characters_count(data):
    """
    6
    :param data: the corpus
    :return: list of number of repetitions of special characters normalized by the number of characters in each post
    """
    result = [0] * len(data)
    for char in ["@", "#", "$", "&", "*", "%", "^"]:
        for i in range(len(data)):
            result[i] += data[i].count(char)
    return [[result[i]/len(data[i])] for i in range(len(data))]


def quotation_mark_count(data):
    """
    7
    :param data: the corpus
    :return: list of number of repetitions of " or ' normalized by the number of characters in each post
    """
    result = [0] * len(data)
    for char in ["\"", "\'"]:
        for i in range(len(data)):
            result[i] += data[i].count(char)
    return [[result[i]/len(data[i])] for i in range(len(data))]


# -----------------------------------------------------------------------------------------
# Averages features


def average_letters_word(data):
    """
    8
    :param data: the corpus
    :return: list of the average length of a words per post
    """
    def average_per_post(post):
        post = re.findall(r'\b\w[\w-]*\b', post.lower())
        num = 0
        for word in post:
            num += len(word)
        return num / len(post)
    return [[average_per_post(post)] for post in data]


def average_letters_sentence(data):
    """
    9
    :param data: the corpus
    :return: list the estimated average of the length of each sentence (no spaces)
    """
    def average_per_post(post):
        post = re.split(r'[.!?]+', post.replace(' ', ''))
        num = 0
        for sentence in post:
            num += len(sentence)
        return num / len(post)
    return [[average_per_post(post)] for post in data]


def average_words_sentence(data):
    """
    10
    :param data: the corpus
    :return: list the estimated average of the num of words in each sentence
    """
    def average_per_post(post):
        post = re.split(r'[.!?]+', post)
        num = 0
        for sentence in post:
            num += len(re.findall(r'\b\w[\w-]*\b', sentence.lower()))
        return num / len(post)
    return [[average_per_post(post)] for post in data]


def average_word_length(data):
    """
    :param data: the corpus
    :return: list the average words length in each post
    """
    new_list = []
    for post in data:
        post = re.findall(r'\b\w[\w-]*\b', post.lower())
        sum = 0
        for word in post:
            sum += len(word)
        new_list += [[sum/len(post)]]
    return new_list


# -----------------------------------------------------------------------------------------
# Reduction and increase features
# (Normalized in the number of words)

def increasing_expressions(data):
    """
    11 - 12 - 13
    :param data: the corpus
    :return: list the percentage of increasing words out of the total words in each post
    """
    from stopwords import increasing_expressions_hebrew
    from stopwords import increasing_expressions_english
    lst = {"hebrew": increasing_expressions_hebrew, "english": increasing_expressions_english}[text_language(data[0])]
    return [prevalence_rate(post, lst, True) for post in data]


def decreasing_expressions(data):
    """
    14 - 15 - 16
    :param data: the corpus
    :return: list the percentage of decreasing words out of the total words in each post
    """
    from stopwords import decreasing_expressions_hebrew
    from stopwords import decreasing_expressions_english
    lst = {"hebrew": decreasing_expressions_hebrew, "english": decreasing_expressions_english}[text_language(data[0])]
    return [prevalence_rate(post, lst, True) for post in data]


# -----------------------------------------------------------------------------------------
# Positive and negative features
# (Normalized in the number of words)


def negative_words(data):
    """
    17 - 18 - 19
    Determine the language of the text and enable the appropriate function
    :param data: the corpus
    :return: list the percentage of positive words out of the total words in each post
    """

    # English version: negative words
    def english_negative_words(data):
        def negative_count(post):
            sid = SentimentIntensityAnalyzer()
            neg_word_num = 0
            post = re.findall(r'\b\w[\w-]*\b', post.lower())
            for word in post:
                if (sid.polarity_scores(word)['compound']) <= -0.5:
                    neg_word_num += 1
            return neg_word_num/len(post)
        return [negative_count(post) for post in data]

    # Hebrew version: negative words
    def hebrew_negative_words(data):
        from stopwords import negative_list_hebrew
        return [prevalence_rate(post, negative_list_hebrew) for post in data]

    if text_language(data[0]) == 'hebrew':
        return hebrew_negative_words(data)
    #nltk.download('vader_lexicon')
    return english_negative_words(data)


def positive_words(data):
    """
    20 - 21 - 22
    Determine the language of the text and activate the appropriate function
    :param data: the corpus
    :return: list the percentage of positive words out of the total words in each post
    """

    # English version: positive words
    def english_positive_words(data):
        def positive_count(post):
            sid = SentimentIntensityAnalyzer()
            pos_word_num = 0
            post = re.findall(r'\b\w[\w-]*\b', post.lower())
            for word in post:
                if (sid.polarity_scores(word)['compound']) >= 0.5:
                    pos_word_num += 1
            return pos_word_num/len(post)
        return [positive_count(post) for post in data]

    # Hebrew version: positive words
    def hebrew_positive_words(data):
        from stopwords import positive_list_hebrew
        return [prevalence_rate(post, positive_list_hebrew) for post in data]

    if text_language(data[0]) == 'hebrew':
        return hebrew_positive_words(data)
    # nltk.download('vader_lexicon')
    return english_positive_words(data)


# -----------------------------------------------------------------------------------------
# Time features
# (Normalized in the number of words)
def time_expressions(data):
    """
    23 - 24 - 25
    :param data: the corpus
    :return: list the percentage of time words out of the total words in each post
    """
    from stopwords import time_expressions_hebrew
    from stopwords import time_expressions_english
    lst = {"hebrew": time_expressions_hebrew, "english": time_expressions_english}[text_language(data[0])]
    return [prevalence_rate(post, lst, True) for post in data]


# -----------------------------------------------------------------------------------------
# Features of disapproval and doubt
# (Normalized in the number of words)
def doubt_expressions(data):
    """
    26 - 27 - 28
    :param data: the corpus
    :return: list the percentage of doubt words out of the total words in each post
    """
    from stopwords import doubt_expressions_hebrew
    from stopwords import doubt_expressions_english
    lst = {"hebrew": doubt_expressions_hebrew, "english": doubt_expressions_english}[text_language(data[0])]
    return [prevalence_rate(post, lst, True) for post in data]


# -----------------------------------------------------------------------------------------
# Features of emotions
# (Normalized in the number of words)
def emotion_expressions(data):
    """
    29
    :param data: the corpus
    :return: list the percentage of emotion terms out of the total words in each post
    """
    from stopwords import emotion_expressions_hebrew
    from stopwords import emotion_expressions_english
    lst = {"hebrew": emotion_expressions_hebrew, "english": emotion_expressions_english}[text_language(data[0])]
    return [prevalence_rate(post, lst, True) for post in data]


# -----------------------------------------------------------------------------------------
# Features of persons
# (Normalized in the number of words)

def first_person_expressions(data):
    """
    30
    :param data: the corpus
    :return: list the percentage of first person terms out of the total words in each post
    """
    from stopwords import first_person_expressions_hebrew
    from stopwords import first_person_expressions_english
    lst = {"hebrew": first_person_expressions_hebrew, "english": first_person_expressions_english}[text_language(data[0])]
    return [prevalence_rate(post, lst, False) for post in data]


def second_person_expressions(data):
    """
    31
    :param data: the corpus
    :return: list the percentage of second person terms out of the total words in each post
    """
    from stopwords import second_person_expressions_hebrew
    from stopwords import second_person_expressions_english
    lst = {"hebrew": second_person_expressions_hebrew, "english": second_person_expressions_english}[text_language(data[0])]
    return [prevalence_rate(post, lst, False) for post in data]


def third_person_expressions(data):
    """
    32
    :param data: the corpus
    :return: list the percentage of third person terms out of the total words in each post
    """
    from stopwords import third_person_expressions_hebrew
    from stopwords import third_person_expressions_english
    lst = {"hebrew": third_person_expressions_hebrew, "english": third_person_expressions_english}[text_language(data[0])]
    return [prevalence_rate(post, lst, False) for post in data]


# -----------------------------------------------------------------------------------------
# Features of inclusion
# (Normalized in the number of words)

def inclusion_expressions(data):
    """
    33
    :param data: the corpus
    :return: list the percentage of inclusion terms out of the total words in each post
    """
    from stopwords import inclusion_expressions_hebrew
    from stopwords import inclusion_expressions_english
    lst = {"hebrew": inclusion_expressions_hebrew, "english": inclusion_expressions_english}[text_language(data[0])]
    return [prevalence_rate(post, lst, False) for post in data]



# -----------------------------------------------------------------------------------------
# Features of powers
# (Normalized in the number of words)

def power1(data):
    """
    34
    :param data: the corpus
    :return: list the percentage of terms from power 1 out of the total words in each post
    """
    from stopwords import power1_expressions_hebrew
    from stopwords import power1_expressions_english
    lst = {"hebrew": power1_expressions_hebrew, "english": power1_expressions_english}[text_language(data[0])]
    return [prevalence_rate(post, lst, True) for post in data]


def power2(data):
    """
    35
    :param data: the corpus
    :return: list the percentage of terms from power 2 out of the total words in each post
    """
    from stopwords import power2_expressions_hebrew
    from stopwords import power2_expressions_english
    lst = {"hebrew": power2_expressions_hebrew, "english": power2_expressions_english}[text_language(data[0])]
    return [prevalence_rate(post, lst, True) for post in data]


def power3(data):
    """
    36
    :param data: the corpus
    :return: list the percentage of terms from power 3 out of the total words in each post
    """
    from stopwords import power3_expressions_hebrew
    from stopwords import power3_expressions_english
    lst = {"hebrew": power3_expressions_hebrew, "english": power3_expressions_english}[text_language(data[0])]
    return [prevalence_rate(post, lst, True) for post in data]


def power_minus1(data):
    """
    37
    :param data: the corpus
    :return: list the percentage of terms from power -1 out of the total words in each post
    """
    from stopwords import powerm1_expressions_hebrew
    from stopwords import powerm1_expressions_english
    lst = {"hebrew": powerm1_expressions_hebrew, "english": powerm1_expressions_english}[text_language(data[0])]
    return [prevalence_rate(post, lst, True) for post in data]


def power_minus2(data):
    """
    38
    :param data: the corpus
    :return: list the percentage of terms from power -2 out of the total words in each post
    """
    from stopwords import powerm2_expressions_hebrew
    from stopwords import powerm2_expressions_english
    lst = {"hebrew": powerm2_expressions_hebrew, "english": powerm2_expressions_english}[text_language(data[0])]
    return [prevalence_rate(post, lst, True) for post in data]


def power_minus3(data):
    """
    39
    :param data: the corpus
    :return: list the percentage of terms from power -3 out of the total words in each post
    """
    from stopwords import powerm3_expressions_hebrew
    from stopwords import powerm3_expressions_english
    lst = {"hebrew": powerm3_expressions_hebrew, "english": powerm3_expressions_english}[text_language(data[0])]
    return [prevalence_rate(post, lst, True) for post in data]


def power_minus4(data):
    """
    40
    :param data: the corpus
    :return: list the percentage of terms from power -4 out of the total words in each post
    """
    from stopwords import powerm4_expressions_hebrew
    from stopwords import powerm4_expressions_english
    lst = {"hebrew": powerm4_expressions_hebrew, "english": powerm4_expressions_english}[text_language(data[0])]
    return [prevalence_rate(post, lst, True) for post in data]


def all_powers(data):
    """
    41
    :param data: the corpus
    :return: list the percentage of terms from all powers out of the total words in each post
    """
    lst = []
    if text_language(data[0]) == "hebrew":
        from stopwords import power1_expressions_hebrew
        from stopwords import power2_expressions_hebrew
        from stopwords import power3_expressions_hebrew
        lst += power1_expressions_hebrew + power2_expressions_hebrew + power3_expressions_hebrew
        from stopwords import powerm1_expressions_hebrew
        from stopwords import powerm2_expressions_hebrew
        from stopwords import powerm3_expressions_hebrew
        from stopwords import powerm4_expressions_hebrew
        lst += powerm1_expressions_hebrew + powerm2_expressions_hebrew + powerm3_expressions_hebrew + powerm4_expressions_hebrew
    else:
        from stopwords import power1_expressions_english
        from stopwords import power2_expressions_english
        from stopwords import power3_expressions_english
        lst += power1_expressions_english + power2_expressions_english + power3_expressions_english
        from stopwords import powerm1_expressions_english
        from stopwords import powerm2_expressions_english
        from stopwords import powerm3_expressions_english
        from stopwords import powerm4_expressions_english
        lst += powerm1_expressions_english + powerm2_expressions_english + powerm3_expressions_english + powerm4_expressions_english
    return [prevalence_rate(post, lst, True) for post in data]


# -----------------------------------------------------------------------------------------
# Topographic Features

def topographicA1(data):
    """
    42
    :param data: the corpus
    :return: enable all properties 1-42 on the first trimester of the post
    """
    data = [post[:int(len(post)/3)] for post in data]
    return topographic(data)


def topographicA2(data):
    """
    43
    :param data: the corpus
    :return: enable all properties 1-42 on the second trimester of the post
    """
    data = [post[int(len(post)/3):len(post) - int(len(post)/3)] for post in data]
    return topographic(data)


def topographicA3(data):
    """
    44
    :param data: the corpus
    :return: enable all properties 1-42 on the third trimester of the post
    """
    data = [post[int(len(post)*(2/3)):] for post in data]
    return topographic(data)


def topographicB1(data):
    """
    45
    :param data: the corpus
    :return: enable all properties 1-42 on the first 10 words of the post
    """
    data = [" ".join(re.findall(r'\b\w[\w-]*\b', post.lower())[:10]) for post in data]
    for i in range(len(data)):
        if len(data[i]) < 1:
            data[i] = "NA"
    return [{True: 0.0, False: value}[value == 11.0] for value in topographic(data)]


def topographicB2(data):
    """
    46
    :param data: the corpus
    :return: enable all properties 1-42 on the first 10 words of the post
    """
    data = [" ".join(re.findall(r'\b\w[\w-]*\b', post.lower())[10:-10]) for post in data]
    for i in range(len(data)):
        if len(data[i]) < 1:
            data[i] = "NA"
    return [{True: 0.0, False: value}[value == 11.0] for value in topographic(data)]


def topographicB3(data):
    """
    47
    :param data: the corpus
    :return: enable all properties 1-42 on the first 10 words of the post
    """
    data = [" ".join(re.findall(r'\b\w[\w-]*\b', post.lower())[:-10]) for post in data]
    for i in range(len(data)):
        if len(data[i]) < 1:
            data[i] = "NA"
    return [{True: 0.0, False: value}[value == 11.0] for value in topographic(data)]


def topographic(data):
    """
    General topographic function
    :param data: the corpus after processing
    :return: enable all properties 1-42 on the corpus
    """
    functions = [chars_count, words_count, sentence_count, exclamation_mark_count, question_mark_count, special_characters_count,
                 quotation_mark_count, average_letters_word, average_letters_sentence, average_words_sentence, average_word_length,
                 increasing_expressions, decreasing_expressions, positive_words, negative_words, time_expressions, doubt_expressions,
                 emotion_expressions, first_person_expressions, second_person_expressions, third_person_expressions,
                 inclusion_expressions, all_powers]
    result = [0] * len(data)
    for func in functions:
        func_re = func(data)
        for i in range(len(data)):
            result[i] += func_re[i]
    return result


##################################################################
class StylisticFeaturesTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, featurizers):
        self.featurizers = featurizers

    def fit(self, X, y=None):
        """All SciKit-Learn compatible transformers and classifiers have the
        same interface. `fit` always returns the same object."""
        return self

    def transform(self, X):
        """Given a list of original data, return a list of feature vectors."""
        _X = preprocessing.normalize(self.featurizers(X))
        for i in range(len(_X)):
            if _X[i][0] < 1:
                _X[i][0] *= 100
        return csr_matrix(_X)
#################################################################


# dic of all supported stylistic features
stylistic_features_dict = {'cc': chars_count,
                           'wc': words_count,
                           'sc': sentence_count,
                           'emc': exclamation_mark_count,
                           'qsmc': question_mark_count,
                           'scc': special_characters_count,
                           'qtmc': quotation_mark_count,
                           'alw': average_letters_word,
                           'als': average_letters_sentence,
                           'aws': average_words_sentence,
                           'awl': average_word_length,
                           'ie': increasing_expressions,
                           'de': decreasing_expressions,
                           'nw': negative_words,
                           'pw': positive_words,
                           'te': time_expressions,
                           'de': doubt_expressions,
                           'ee': emotion_expressions,
                           'fpe': first_person_expressions,
                           'spe': second_person_expressions,
                           'tpe': third_person_expressions,
                           'ine': inclusion_expressions,
                           'p1': power1,
                           'p2': power2,
                           'p3': power3,
                           'pm1': power_minus1,
                           'pm2': power_minus2,
                           'pm3': power_minus3,
                           'pm4': power_minus4,
                           'ap': all_powers,
                           'topa1': topographicA1,
                           'topa2': topographicA2,
                           'topa3': topographicA3,
                           'topb1': topographicB1,
                           'topb2': topographicB2,
                           'topb3': topographicB3}


def get_stylistic_features_vectorizer(feature):
    # return the values of the features
    vectorizer = StylisticFeaturesTransformer(stylistic_features_dict[feature])
    return vectorizer



if __name__ == "__main__":
    dic = {'{"train": "C:\\\\Users\\\\user\\\\Documents\\\\test\\\\dataset\\\\training", "test": "C:\\\\Users\\\\user\\\\Documents\\\\test\\\\dataset\\\\testing", "output_csv": "C:\\\\Users\\\\user\\\\Documents\\\\test\\\\output", "nargs": "", "features": ["ngrams_0_w_tfidf_1_0", "ngrams_2000_c_tf_1_0"], "results": "C:\\\\Users\\\\user\\\\Documents\\\\test\\\\results", "methods": ["mlp", "svc", "rf", "lr", "mnb", "rnn"]}': {'mlp': 0.8121212121212121, 'svc': 0.8121212121212121, 'rf': 0.793939393939394, 'lr': 0.8, 'mnb': 0.8, 'rnn': 0.8000000003612403}}
    #print(reformat_data(dic))

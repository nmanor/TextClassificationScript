import re

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from global_parameters import GlobalParameters
from pos_tags import get_pos_transformer
from stopwords_and_lists import (
    food_family,
    fat_family,
    vomiting_family,
    me_family,
    ana_family,
    increasing_family,
    decreasing_family,
    sport_family,
    sleep_family,
    hunger_family,
    pain_family,
    anorexia_family,
    anger_family,
    thinness_family,
    calories_family,
    vulgarity_family,
    sickness_family,
    love_family,
    export_50_terms_he,
    export_50_terms_en,
    export_50_terms_trans_he,
    export_50_terms_trans_en,
    anorexia_family_en,
    food_family_en,
    fat_family_en,
    ana_family_en,
    hunger_family_en,
    me_family_en,
    vomiting_family_en,
    pain_family_en,
    anger_family_en,
    sleep_family_en,
    sport_family_en,
    thinness_family_en,
    calories_family_en,
    vulgarity_family_en,
    decreasing_family_en,
    increasing_family_en,
    sickness_family_en,
    love_family_en,
    noun_family,
    sex_family_en,
    cursing_family_en,
    alcohol_family_en,
    smoke_family_en,
    extended_time_expressions_hebrew,
    extended_time_expressions_english,
    first_person_expressions_hebrew,
    first_person_expressions_english,
    second_person_expressions_hebrew,
    second_person_expressions_english,
    third_person_expressions_hebrew,
    third_person_expressions_english,
    time_expressions_hebrew,
    time_expressions_english,
    positive_list_hebrew,
    negative_list_hebrew,
    custom_list)
from terms_frequency_counts import get_top_n_words

glbs = GlobalParameters()

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
            length = len(word.split(" "))
        else:
            length = 1
        num += str.lower().count(word) * length
        str = str.replace(word, "")
    return [num / len(re.findall(r"\b\w[\w-]*\b", orginal_str.lower()))]


# -----------------------------------------------------------------------------------------
# Quantitative features
# (Normalized in words and characters)


def chars_count(data):
    """
    1
    :param data: the corpus
    :return: list the number of characters in each post
    """
    if data == "get feature names":
        return ["chars count"]

    return [[len(post)] for post in data]


def words_count(data):
    """
    2
    :param data: the corpus
    :return: list the number of words in each post
    """
    if data == "get feature names":
        return ["words count"]

    return [[len(re.findall(r"\b\w[\w-]*\b", post.lower()))] for post in data]


def sentence_count(data):
    """
    3
    :param data: the corpus
    :return: list the estimated number of sentences in each post
    """
    if data == "get feature names":
        return ["sentence count"]

    for post in data:
        post.replace("...", ".").replace("..", ".")
        post.replace("!!!", "!").replace("!!", "!")
        post.replace("???", "?").replace("??", "?")
    return [[len(re.split(r"[.!?]+", post))] for post in data]


def exclamation_mark_count(data):
    """
    4
    :param data: the corpus
    :return: list of number of repetitions of ! normalized by the number of characters in each post
    """
    if data == "get feature names":
        return ["exclamation mark count"]

    return [[post.count("!") / len(post)] for post in data]


def question_mark_count(data):
    """
    5
    :param data: the corpus
    :return: list of number of repetitions of ? normalized by the number of characters in each post
    """
    if data == "get feature names":
        return ["question mark count"]

    return [[post.count("?") / len(post)] for post in data]


def special_characters_count(data):
    """
    6
    :param data: the corpus
    :return: list of number of repetitions of special characters normalized by the number of characters in each post
    """
    if data == "get feature names":
        return ["special characters count"]

    result = [0] * len(data)
    for char in ["@", "#", "$", "&", "*", "%", "^"]:
        for i in range(len(data)):
            result[i] += data[i].count(char)
    return [[result[i] / len(data[i])] for i in range(len(data))]


def quotation_mark_count(data):
    """
    7
    :param data: the corpus
    :return: list of number of repetitions of " or ' normalized by the number of characters in each post
    """
    if data == "get feature names":
        return ["quotation mark count"]

    result = [0] * len(data)
    for char in ['"', "'"]:
        for i in range(len(data)):
            result[i] += data[i].count(char)
    return [[result[i] / len(data[i])] for i in range(len(data))]


# -----------------------------------------------------------------------------------------
# Averages features


def average_letters_word(data):
    """
    8
    :param data: the corpus
    :return: list of the average length of a words per post
    """

    def average_per_post(post):
        post = re.findall(r"\b\w[\w-]*\b", post.lower())
        num = 0
        for word in post:
            num += len(word)
        return num / len(post)

    if data == "get feature names":
        return ["average letters word"]

    return [[average_per_post(post)] for post in data]


def average_letters_sentence(data):
    """
    9
    :param data: the corpus
    :return: list the estimated average of the length of each sentence (no spaces)
    """

    def average_per_post(post):
        post = re.split(r"[.!?]+", post.replace(" ", ""))
        num = 0
        for sentence in post:
            num += len(sentence)
        return num / len(post)

    if data == "get feature names":
        return ["average letters sentence"]

    return [[average_per_post(post)] for post in data]


def average_words_sentence(data):
    """
    10
    :param data: the corpus
    :return: list the estimated average of the num of words in each sentence
    """

    def average_per_post(post):
        post = re.split(r"[.!?]+", post)
        num = 0
        for sentence in post:
            num += len(re.findall(r"\b\w[\w-]*\b", sentence.lower()))
        return num / len(post)

    if data == "get feature names":
        return ["average words sentence"]

    return [[average_per_post(post)] for post in data]


def average_word_length(data):
    """
    :param data: the corpus
    :return: list the average words length in each post
    """
    if data == "get feature names":
        return ["average word length"]

    new_list = []
    for post in data:
        post = re.findall(r"\b\w[\w-]*\b", post.lower())
        sum = 0
        for word in post:
            sum += len(word)
        new_list += [[sum / len(post)]]
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
    from stopwords_and_lists import increasing_expressions_hebrew
    from stopwords_and_lists import increasing_expressions_english

    lst = {
        "hebrew": increasing_expressions_hebrew,
        "english": increasing_expressions_english,
    }[text_language(data[0])]

    if data == "get feature names":
        return ["increasing expressions"]

    return [prevalence_rate(post, lst, True) for post in data]


def decreasing_expressions(data):
    """
    14 - 15 - 16
    :param data: the corpus
    :return: list the percentage of decreasing words out of the total words in each post
    """
    from stopwords_and_lists import decreasing_expressions_hebrew
    from stopwords_and_lists import decreasing_expressions_english

    lst = {
        "hebrew": decreasing_expressions_hebrew,
        "english": decreasing_expressions_english,
    }[text_language(data[0])]

    if data == "get feature names":
        return ["decreasing expressions"]

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
            post = re.findall(r"\b\w[\w-]*\b", post.lower())
            for word in post:
                if (sid.polarity_scores(word)["compound"]) <= -0.5:
                    neg_word_num += 1
            return [neg_word_num / len(post)]

        return [negative_count(post) for post in data]

    # Hebrew version: negative words
    def hebrew_negative_words(data):
        from stopwords_and_lists import negative_list_hebrew

        return [prevalence_rate(post, negative_list_hebrew) for post in data]

    if data == "get feature names":
        return ["negative words"]

    if text_language(data[0]) == "hebrew":
        return hebrew_negative_words(data)
    # nltk.download('vader_lexicon')
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
            post = re.findall(r"\b\w[\w-]*\b", post.lower())
            for word in post:
                if (sid.polarity_scores(word)["compound"]) >= 0.5:
                    pos_word_num += 1
            return [pos_word_num / len(post)]

        return [positive_count(post) for post in data]

    # Hebrew version: positive words
    def hebrew_positive_words(data):
        from stopwords_and_lists import positive_list_hebrew

        return [prevalence_rate(post, positive_list_hebrew) for post in data]

    if data == "get feature names":
        return ["positive words"]

    if text_language(data[0]) == "hebrew":
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
    from stopwords_and_lists import time_expressions_hebrew
    from stopwords_and_lists import time_expressions_english

    lst = {"hebrew": time_expressions_hebrew, "english": time_expressions_english}[
        text_language(data[0])
    ]

    if data == "get feature names":
        return ["time expressions"]

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
    from stopwords_and_lists import doubt_expressions_hebrew
    from stopwords_and_lists import doubt_expressions_english

    lst = {"hebrew": doubt_expressions_hebrew, "english": doubt_expressions_english}[
        text_language(data[0])
    ]

    if data == "get feature names":
        return ["doubt expressions"]

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
    from stopwords_and_lists import emotion_expressions_hebrew
    from stopwords_and_lists import emotion_expressions_english

    lst = {
        "hebrew": emotion_expressions_hebrew,
        "english": emotion_expressions_english,
    }[text_language(data[0])]

    if data == "get feature names":
        return ["emotion expressions"]

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
    from stopwords_and_lists import first_person_expressions_hebrew
    from stopwords_and_lists import first_person_expressions_english

    lst = {
        "hebrew": first_person_expressions_hebrew,
        "english": first_person_expressions_english,
    }[text_language(data[0])]

    if data == "get feature names":
        return ["first person expressions"]

    return [prevalence_rate(post, lst, False) for post in data]


def second_person_expressions(data):
    """
    31
    :param data: the corpus
    :return: list the percentage of second person terms out of the total words in each post
    """
    from stopwords_and_lists import second_person_expressions_hebrew
    from stopwords_and_lists import second_person_expressions_english

    lst = {
        "hebrew": second_person_expressions_hebrew,
        "english": second_person_expressions_english,
    }[text_language(data[0])]

    if data == "get feature names":
        return ["second person expressions"]

    return [prevalence_rate(post, lst, False) for post in data]


def third_person_expressions(data):
    """
    32
    :param data: the corpus
    :return: list the percentage of third person terms out of the total words in each post
    """
    from stopwords_and_lists import third_person_expressions_hebrew
    from stopwords_and_lists import third_person_expressions_english

    lst = {
        "hebrew": third_person_expressions_hebrew,
        "english": third_person_expressions_english,
    }[text_language(data[0])]

    if data == "get feature names":
        return ["third person expressions"]

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
    from stopwords_and_lists import inclusion_expressions_hebrew
    from stopwords_and_lists import inclusion_expressions_english

    lst = {
        "hebrew": inclusion_expressions_hebrew,
        "english": inclusion_expressions_english,
    }[text_language(data[0])]

    if data == "get feature names":
        return ["inclusion expressions"]

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
    from stopwords_and_lists import power1_expressions_hebrew
    from stopwords_and_lists import power1_expressions_english

    lst = {"hebrew": power1_expressions_hebrew, "english": power1_expressions_english}[
        text_language(data[0])
    ]

    if data == "get feature names":
        return ["power 1"]

    return [prevalence_rate(post, lst, True) for post in data]


def power2(data):
    """
    35
    :param data: the corpus
    :return: list the percentage of terms from power 2 out of the total words in each post
    """
    from stopwords_and_lists import power2_expressions_hebrew
    from stopwords_and_lists import power2_expressions_english

    lst = {"hebrew": power2_expressions_hebrew, "english": power2_expressions_english}[
        text_language(data[0])
    ]

    if data == "get feature names":
        return ["power 2"]

    return [prevalence_rate(post, lst, True) for post in data]


def power3(data):
    """
    36
    :param data: the corpus
    :return: list the percentage of terms from power 3 out of the total words in each post
    """
    from stopwords_and_lists import power3_expressions_hebrew
    from stopwords_and_lists import power3_expressions_english

    lst = {"hebrew": power3_expressions_hebrew, "english": power3_expressions_english}[
        text_language(data[0])
    ]

    if data == "get feature names":
        return ["power 3"]

    return [prevalence_rate(post, lst, True) for post in data]


def power_minus1(data):
    """
    37
    :param data: the corpus
    :return: list the percentage of terms from power -1 out of the total words in each post
    """
    from stopwords_and_lists import powerm1_expressions_hebrew
    from stopwords_and_lists import powerm1_expressions_english

    lst = {
        "hebrew": powerm1_expressions_hebrew,
        "english": powerm1_expressions_english,
    }[text_language(data[0])]

    if data == "get feature names":
        return ["power -1"]

    return [prevalence_rate(post, lst, True) for post in data]


def power_minus2(data):
    """
    38
    :param data: the corpus
    :return: list the percentage of terms from power -2 out of the total words in each post
    """
    from stopwords_and_lists import powerm2_expressions_hebrew
    from stopwords_and_lists import powerm2_expressions_english

    lst = {
        "hebrew": powerm2_expressions_hebrew,
        "english": powerm2_expressions_english,
    }[text_language(data[0])]

    if data == "get feature names":
        return ["power -2"]

    return [prevalence_rate(post, lst, True) for post in data]


def power_minus3(data):
    """
    39
    :param data: the corpus
    :return: list the percentage of terms from power -3 out of the total words in each post
    """
    from stopwords_and_lists import powerm3_expressions_hebrew
    from stopwords_and_lists import powerm3_expressions_english

    lst = {
        "hebrew": powerm3_expressions_hebrew,
        "english": powerm3_expressions_english,
    }[text_language(data[0])]

    if data == "get feature names":
        return ["power -3"]

    return [prevalence_rate(post, lst, True) for post in data]


def power_minus4(data):
    """
    40
    :param data: the corpus
    :return: list the percentage of terms from power -4 out of the total words in each post
    """
    from stopwords_and_lists import powerm4_expressions_hebrew
    from stopwords_and_lists import powerm4_expressions_english

    lst = {
        "hebrew": powerm4_expressions_hebrew,
        "english": powerm4_expressions_english,
    }[text_language(data[0])]

    if data == "get feature names":
        return ["power -4"]

    return [prevalence_rate(post, lst, True) for post in data]


def all_powers(data):
    """
    41
    :param data: the corpus
    :return: list the percentage of terms from all powers out of the total words in each post
    """
    lst = []
    if text_language(data[0]) == "hebrew":
        from stopwords_and_lists import power1_expressions_hebrew
        from stopwords_and_lists import power2_expressions_hebrew
        from stopwords_and_lists import power3_expressions_hebrew

        lst += (
            power1_expressions_hebrew
            + power2_expressions_hebrew
            + power3_expressions_hebrew
        )
        from stopwords_and_lists import powerm1_expressions_hebrew
        from stopwords_and_lists import powerm2_expressions_hebrew
        from stopwords_and_lists import powerm3_expressions_hebrew
        from stopwords_and_lists import powerm4_expressions_hebrew

        lst += (
            powerm1_expressions_hebrew
            + powerm2_expressions_hebrew
            + powerm3_expressions_hebrew
            + powerm4_expressions_hebrew
        )
    else:
        from stopwords_and_lists import power1_expressions_english
        from stopwords_and_lists import power2_expressions_english
        from stopwords_and_lists import power3_expressions_english

        lst += (
            power1_expressions_english
            + power2_expressions_english
            + power3_expressions_english
        )
        from stopwords_and_lists import powerm1_expressions_english
        from stopwords_and_lists import powerm2_expressions_english
        from stopwords_and_lists import powerm3_expressions_english
        from stopwords_and_lists import powerm4_expressions_english

        lst += (
            powerm1_expressions_english
            + powerm2_expressions_english
            + powerm3_expressions_english
            + powerm4_expressions_english
        )

    if data == "get feature names":
        return ["all powers"]

    return [prevalence_rate(post, lst, True) for post in data]


# -----------------------------------------------------------------------------------------
# Topographic Features


def known_repeated_chars():
    """
    :param docs: the data
    :return: CSR Matrix with the num of repeation of each char in the list bellow
    """

    class InitTransformer(TransformerMixin, BaseEstimator):
        def __init__(self, char):
            self.char = char

        def fit(self, X, y=None):
            """All SciKit-Learn compatible transformers and classifiers have the
            same interface. `fit` always returns the same object."""
            return self

        def transform(self, X):
            def init_(post, char):
                return post.count(char) / len(post)

            X = [[init_(post, self.char)] for post in X]
            return csr_matrix(X)

        def get_feature_names(self):
            """Array mapping from feature integer indices to feature name"""
            return [self.char]

    feature_lst = []
    for char in [
        "!",
        "?",
        ".",
        "<",
        ">",
        "=",
        ")",
        "(",
        ":",
        "+",
        "*",
        "):",
        ":(",
        "(:",
        ":)",
    ]:
        feature_lst += [("KRC_" + char, InitTransformer(char))]

    return FeatureUnion(feature_lst)


# -----------------------------------------------------------------------------------------
# Repeated chars features

# return the normalized number of words with at least 3 repeated chars
def repeated_chars(data):
    # check for any repeated chars
    def in_word(word):
        for i in range(len(word) - 2):
            if word[i] == word[i + 1] == word[i + 2]:
                return True
        return False

    # check for the number of words with repeated chars
    def in_post(post):
        post = re.findall(r"\b\w[\w-]*\b", post.lower())
        repeated = 0
        for word in post:
            if in_word(word):
                repeated += 1
        return repeated / len(post)

    if data == "get feature names":
        return ["repeated chars"]

    # return the result
    return [[in_post(post)] for post in data]


def doubled_words(data):
    """
    :param data: the corpus
    :return: the num of doubled word normalized in the num f the words in the text
    """

    def in_post(post):
        num = 0
        post = re.findall(r"\b\w[\w-]*\b", post.lower())
        for i in range(len(post) - 1):
            if post[i] == post[i + 1]:
                num += 1
        return num / (len(post) - 1)

    if data == "get feature names":
        return ["doubled words"]

    return [[in_post(post)] for post in data]


def tripled_words(data):
    """
    :param data: the corpus
    :return: the num of tripled word normalized in the num of the words in the text
    """

    def in_post(post):
        num = 0
        post = re.findall(r"\b\w[\w-]*\b", post.lower())
        for i in range(len(post) - 2):
            if post[i] == post[i + 1] == post[i + 2]:
                num += 1
        return num / (len(post) - 2)

    if data == "get feature names":
        return ["tripled words"]

    return [[in_post(post)] for post in data]


def doubled_exclamation(data):
    """
    :param data: the corpus
    :return: the num of doubled ! normalized in the num of the letters in the text
    """
    if data == "get feature names":
        return ["doubled exclamation"]

    return [[post.count("!!") / len(post)] for post in data]


def tripled_exclamation(data):
    """
    :param data: the corpus
    :return: the num of doubled !! normalized in the num of the letters in the text
    """
    if data == "get feature names":
        return ["tripled exclamation"]

    return [[len(re.findall(r"!!!+", post)) / len(post)] for post in data]


def doubled_hyphen(data):
    """
    :param data: the corpus
    :return: the num of the words that contain at least 2 '-' normalized in the num of the words
    """

    def in_post(post):
        num = 0
        post = re.findall(r"\b\w[\w-]*\b", post.lower())
        for word in post:
            if word.count("-") >= 2:
                num += 1
        return num / len(post)

    if data == "get feature names":
        return ["doubled hyphen"]

    return [[in_post(post)] for post in data]


# -----------------------------------------------------------------------------------------
# General list
def general_list(data, lst):
    return [prevalence_rate(post, lst, True) for post in data]


# -----------------------------------------------------------------------------------------
# Language wealth


def words_wealth(data):
    """
    :param data: the corpus
    :return: list of the number of different types of words normalized in the number of words in the text
    """

    def in_post(post):
        post = re.findall(r"\b\w[\w-]*\b", post.lower())
        single_words = set(post)
        return len(single_words) / len(post)

    if data == "get feature names":
        return ["words wealth"]

    return [[in_post(post)] for post in data]


def n_word_in_post(post, num):
    unique = [
        1 for tup in get_top_n_words([post], 1, 1, filtering=False) if tup[1] == num
    ]
    post = re.findall(r"\b\w[\w-]*\b", post.lower())
    return len(unique) / len(post)


def once_words(data):
    """
    :param data: the corpus
    :return: the number of words that used only once in the post normalized in the number of the words
    """
    if data == "get feature names":
        return ["once words"]

    return [[n_word_in_post(post, 1)] for post in data]


def twice_words(data):
    """
    :param data: the corpus
    :return: the number of words that used only twice in the post normalized in the number of the words
    """
    if data == "get feature names":
        return ["twice words"]

    return [[n_word_in_post(post, 2)] for post in data]


def three_times_words(data):
    """
    :param data: the corpus
    :return: the number of words that used only three times in the post normalized in the number of the words
    """
    if data == "get feature names":
        return ["three times words"]

    return [[n_word_in_post(post, 2)] for post in data]


##################################################################
class StylisticFeaturesTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, featurizers, feature):
        self.featurizers = featurizers
        self.feature_name = feature

    def fit(self, X, y=None):
        """All SciKit-Learn compatible transformers and classifiers have the
        same interface. `fit` always returns the same object."""
        return self

    def transform(self, X):
        """Given a list of original data, return a list of feature vectors."""
        if isinstance(self.featurizers, list):
            return csr_matrix(general_list(X, self.featurizers))

        _X = self.featurizers(X)
        return csr_matrix(_X)

    def get_feature_names(self):
        """Array mapping from feature integer indices to feature name"""
        if isinstance(self.featurizers, list):
            return [self.feature_name]
        return self.featurizers("get feature names")


#################################################################
# dic of all supported stylistic features
# DO NOT USE THIS DICTIONARY BEFORE YOU ACTIVATE initialization_features_dict
stylistic_features_dict = {}


# Initialize dictionary by global properties
def initialize_features_dict():
    global stylistic_features_dict
    stylistic_features_dict = {
        "cc": chars_count,
        "wc": words_count,
        "sc": sentence_count,
        "emc": exclamation_mark_count,
        "qsmc": question_mark_count,
        "scc": special_characters_count,
        "qtmc": quotation_mark_count,
        "alw": average_letters_word,
        "als": average_letters_sentence,
        "aws": average_words_sentence,
        "awl": average_word_length,
        "ie": increasing_expressions,
        "dex": decreasing_expressions,
        "nw": negative_list_hebrew if glbs.LANGUAGE == "hebrew" else negative_words,
        "pw": positive_list_hebrew if glbs.LANGUAGE == "hebrew" else positive_words,
        # 'te': time_expressions,
        "te": time_expressions_hebrew
        if glbs.LANGUAGE == "hebrew"
        else time_expressions_english,
        "de": doubt_expressions,
        "ee": emotion_expressions,
        "fpe": first_person_expressions_hebrew
        if glbs.LANGUAGE == "hebrew"
        else first_person_expressions_english,
        "spe": second_person_expressions_hebrew
        if glbs.LANGUAGE == "hebrew"
        else second_person_expressions_english,
        "tpe": third_person_expressions_hebrew
        if glbs.LANGUAGE == "hebrew"
        else third_person_expressions_english,
        "ine": inclusion_expressions,
        "p1": power1,
        "p2": power2,
        "p3": power3,
        "pm1": power_minus1,
        "pm2": power_minus2,
        "pm3": power_minus3,
        "pm4": power_minus4,
        "ap": all_powers,
        "frc": known_repeated_chars,
        "pos": get_pos_transformer,
        "rc": repeated_chars,
        "dw": doubled_words,
        "tw": tripled_words,
        "dh": doubled_hyphen,
        "dx": doubled_exclamation,
        "tx": tripled_exclamation,
        "ww": words_wealth,
        "owc": once_words,
        "twc": twice_words,
        "ttc": three_times_words,
        "aof": anorexia_family if glbs.LANGUAGE == "hebrew" else anorexia_family_en,
        "fdf": food_family if glbs.LANGUAGE == "hebrew" else food_family_en,
        "ftf": fat_family if glbs.LANGUAGE == "hebrew" else fat_family_en,
        "anf": ana_family if glbs.LANGUAGE == "hebrew" else ana_family_en,
        "huf": hunger_family if glbs.LANGUAGE == "hebrew" else hunger_family_en,
        "mef": me_family if glbs.LANGUAGE == "hebrew" else me_family_en,
        "vof": vomiting_family if glbs.LANGUAGE == "hebrew" else vomiting_family_en,
        "pnf": pain_family if glbs.LANGUAGE == "hebrew" else pain_family_en,
        "agf": anger_family if glbs.LANGUAGE == "hebrew" else anger_family_en,
        "slf": sleep_family if glbs.LANGUAGE == "hebrew" else sleep_family_en,
        "spf": sport_family if glbs.LANGUAGE == "hebrew" else sport_family_en,
        "thf": thinness_family if glbs.LANGUAGE == "hebrew" else thinness_family_en,
        "caf": calories_family if glbs.LANGUAGE == "hebrew" else calories_family_en,
        "vuf": vulgarity_family if glbs.LANGUAGE == "hebrew" else vulgarity_family_en,
        "def": decreasing_family if glbs.LANGUAGE == "hebrew" else decreasing_family_en,
        "inf": increasing_family if glbs.LANGUAGE == "hebrew" else increasing_family_en,
        "sif": sickness_family if glbs.LANGUAGE == "hebrew" else sickness_family_en,
        "lof": love_family if glbs.LANGUAGE == "hebrew" else love_family_en,
        "xte": extended_time_expressions_hebrew
        if glbs.LANGUAGE == "hebrew"
        else extended_time_expressions_english,
        "nof": noun_family,
        "sxf": sex_family_en,
        "cuf": cursing_family_en,
        "alf": alcohol_family_en,
        "skf": smoke_family_en,
        "e50th": export_50_terms_he,
        "e50te": export_50_terms_en,
        "e50tth": export_50_terms_trans_he,
        "e50tte": export_50_terms_trans_en,
        "acf": {"wc", "cc", "sc", "alw", "als", "aws", "awl"},
        "ref": {"dw", "tw", "dh", "dx", "tx"},
        "wef": {"ww", "owc", "twc", "ttc"},
        "custom": custom_list
    }
    return stylistic_features_dict


def get_stylistic_features_vectorizer(feature):
    initialize_features_dict()
    vectorizers = []

    # return the CountVectorizer of lists of words
    if isinstance(stylistic_features_dict[feature.lower()], list):
        lst = set(stylistic_features_dict[feature.lower()])
        vectorizers += [TfidfVectorizer(vocabulary=lst)]
        if (
                feature != "e50th"
                and feature != "e50te"
                and feature != "e50tth"
                and feature != "e50tte"
        ):
            vectorizers += [
                StylisticFeaturesTransformer(stylistic_features_dict[feature], feature)
            ]

    elif isinstance(stylistic_features_dict[feature.lower()], set):
        for group_feature in stylistic_features_dict[feature.lower()]:
            vectorizers += get_stylistic_features_vectorizer(group_feature)

    elif feature.lower() == "frc" or feature.lower() == "pos":
        return [stylistic_features_dict[feature]()]

    # return the values of the features
    else:
        vectorizers += [
            StylisticFeaturesTransformer(stylistic_features_dict[feature], feature)
        ]

    return vectorizers


if __name__ == "__main__":
    vec = get_stylistic_features_vectorizer("pos")[0]
    cor = ["hey, whats up up up? do you want to build a snow man?", "whats going on?"]
    print(vec.fit_transform(cor))
    print(vec.get_feature_names())

import os
import random
import shutil
from os import path

import matplotlib.pyplot as plt

from global_parameters import GlobalParameters
from stopwords_and_lists import hebrew_stopwords, stopwords
from stylistic_features import text_language
from wordcloud import WordCloud
from terms_frequency_counts import get_top_n_words


def Random_color(*args, **kwargs):
    colors = [(66, 135, 245), (245, 66, 81), (182, 245, 66), (245, 230, 66)]
    return random.choice(colors)


def generate_word_clouds(max_words=200):
    glbs = GlobalParameters()
    training_path = glbs.TRAIN_DIR
    testing_path = glbs.TEST_DIR
    result_path = glbs.RESULTS_PATH + r"\Words Clouds"

    training = {}
    for file in os.listdir(training_path):
        if file.endswith('.txt'):
            training[file.replace('.txt', '')] = open(training_path + "\\" + file, "r", encoding="utf8", errors='replace').read().replace('\n', ' ').lower()

    testing = {}
    for file in os.listdir(testing_path):
        if file.endswith('.txt'):
            testing[file.replace('.txt', '')] = open(testing_path + "\\" + file, "r", encoding="utf8", errors='replace').read().replace('\n', ' ').lower()

    if text_language(testing[list(testing.keys())[0]]) == 'hebrew':
        for key, value in training.items():
            training[key] = value[::-1]
        for key, value in testing.items():
            testing[key] = value[::-1]
        stop_words = hebrew_stopwords
    else:
        stop_words = stopwords

    for name, text in training.items():
        title = "training " + name + " unigrams"
        freq = dict(get_top_n_words([text], 1, 1))
        generate_and_save(freq, max_words, result_path, stop_words, title)

        title = "training " + name + " bigrams"
        freq = dict(get_top_n_words([text], 2, 2))
        generate_and_save(freq, max_words, result_path, stop_words, title)

    for name, text in testing.items():
        title = "testing " + name + " unigrams"
        freq = dict(get_top_n_words([text], 1, 1))
        generate_and_save(freq, max_words, result_path, stop_words, title)

        title = "testing " + name + " bigrams"
        freq = dict(get_top_n_words([text], 2, 2))
        generate_and_save(freq, max_words, result_path, stop_words, title)


def generate_and_save(freq, max_words, result_path, stop_words, title):
    # Generate the WordCloud
    wordcloud = WordCloud(width=1920, height=1080,
                          max_words=max_words,
                          background_color='white',
                          stopwords=stop_words,
                          color_func=Random_color,
                          min_font_size=10,
                          font_path=r"C:\WINDOWS\FONTS\HEEBO-MEDIUM.TTF").generate_from_frequencies(freq)

    # plot the WordCloud image
    plt.figure(figsize=(17, 10), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    if path.exists(result_path):
        shutil.rmtree(result_path, ignore_errors=True)
    os.makedirs(result_path)

    plt.savefig(result_path + "\\" + title + '.jpg', format='jpg')


if __name__ == "__main__":
    pass
    # generate_word_clouds(r"C:\Users\user\Documents\test\dataset\training", r"C:\Users\user\Documents\test\dataset\testing", r"C:\Users\user\Documents\test\results")
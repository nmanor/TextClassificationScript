from sklearn.feature_extraction.text import CountVectorizer


def get_top_n_words(corpus, ngrams1=2, ngrams2=2, n=None):
    """
    :param n: [the number of words needed]
    :param ngrams2: [ngrams upper bound]
    :param corpus: list of strings
    :param ngrams1: [ngrams lower bound]
    :return: List the top n words in a vocabulary according to occurrence in a text corpus
    :rtype: list

    get_top_n_words(["I love Python", "Python is a language programming", "Hello world", "I love the world"]) ->
    [('python', 2),
     ('world', 2),
     ('love', 2),
     ('hello', 1),
     ('is', 1),
     ('programming', 1),
     ('the', 1),
     ('language', 1)]
    """
    vec = CountVectorizer(ngram_range=(ngrams1, ngrams2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


def lower_bound_test(top_words, corpus, threshold = 0.03, minimum_posts = 3):
    """
    :param top_words: list of most common words in the corpus in order
    :param corpus: the corpus itself
    :param threshold: only words that appear in at least 'threshold' percent of posts will enter
    :param minimum_posts: the minimum number of posts a words must appear in
    :return: Returns a list of the n most common words provided that each of the
            words appears in at least 3% of the posts in the corpus and in 3 different posts
    """

    # split the corpus from string to list
    corpus = str(corpus).split('\n')

    # the list of words to return
    new_list = []

    # for each words in the most common sequence of words:
    for word in top_words:
        repetition = 0
        # for each post in the corpus:
        for post in corpus:
            # if the words appears in the post, update the number of repetitions
            if word[0] in post:
                repetition += 1
            # if so far the words has been repeated more than #threshold percent in the database,
            # and the words also appeared in at least #minimum_posts different posts in the database,
            # then the words passed the tests successfully: add the words to the final list and stop the inner loop
            if repetition / len(corpus) >= threshold and repetition >= minimum_posts:
                new_list += [word]
                break

    # return the final list of words
    return new_list


def word_freq(corpus_path, out_path, ngrams, show_amaount=True):
    # read the content of the file
    text = open(corpus_path, "r", encoding="utf8", errors='replace').read()

    # collect the words in order of importance
    result = ''
    i = 1
    for tup in lower_bound_test(get_top_n_words([text.replace('\n', ' ').lower()], ngrams, ngrams), text)[:1000]:
        result += '\n' + str(i) + ":" + tup[0]
        if show_amaount:
            result += ' - ' + str(tup[1])

    # save the words into the output path
    title = "\\" + corpus_path.split('\\')[-1].split('.')[0] + " most freq words " + {2:"bigrams", 1:"unigrams", 3:"trigrams"}[ngrams] + ".txt"
    with open(out_path + title, "w", encoding="utf8", errors='replace') as file:
        file.write(result[1:])


if __name__ == '__main__':
    path = r"C:\Users\user\Documents\test\קבצי txt\Training Normal in English.txt"
    result = ''
    text = open(path, "r", encoding="utf8", errors='replace').read()
    for i, tup in enumerate(get_top_n_words([text], 1, 1, 2000)):
        result += '\n' + str(i + 1) + ": " + tup[0] + ' - ' + str(tup[1])
    print(result)
    with open(r'C:\Users\user\Documents\test\קבצי txt\r.txt', "w", encoding="utf8", errors='replace') as file:
        file.write(result[1:])

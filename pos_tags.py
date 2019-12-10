# -----------------------------------------------------------------------------------------
# Part Of Speech (POS) tags
import nltk
import requests
from scipy.sparse import csr_matrix


def get_english_pos_tags(post):
    tokens = nltk.word_tokenize(post)
    tags = nltk.pos_tag(tokens)
    post_dict = {}
    for _, pos in tags:
        if pos in post_dict:
            post_dict[pos] += 1
        else:
            post_dict[pos] = 1
    return post_dict


def get_hebrew_pos_tags(post):
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


if __name__ == '__main__':
    row_ind = []
    col_ind = []
    data = []
    for i in range(25):
        if i%2 == 1:
            row_ind += [i]
            col_ind += [i]
            data += [i]
    matrix = csr_matrix((data, (row_ind, col_ind)))
    print(matrix)

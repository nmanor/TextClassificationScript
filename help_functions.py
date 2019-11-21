import json
import os
import collections
import re
from random import shuffle

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from wordfreq import top_n_list

from normalization import remove_stop_words_hebrew_extended

# dir = the directory with all the files (txt & json of anorexia and normal)
# output = where to put the merged files
# split = the ratio of the split (default: 1/3)
def create_dataset(dir, output, split = 0.33):
    # get all the data and put it in the right list
    anorexia = []
    normal = []
    for file in os.listdir(dir):
        if file.endswith('.json'):
            with open(dir + "\\" + file, "r", encoding="utf8", errors='replace') as f:
                if json.load(f)["classification"] == "anorexia":
                    anorexia = anorexia + [file]
                else:
                    normal = normal + [file]

    # shuffle the lists
    shuffle(anorexia)
    shuffle(normal)

    # split the anorexia files
    test = anorexia[:int(split*len(anorexia))]
    train = anorexia[int(split*len(anorexia)):]

    # save the content of the anorexia test
    text = ''
    for file in test:
        with open(dir + "\\" + file, "r+", encoding="utf8", errors='replace') as f:
            dic = json.load(f)
            text += '\n' + open(dir + "\\" + dic["file_id"], "r", errors='replace', encoding='utf8').read().replace('\n', ' ')
    with open(output + "\\testing\\anorexia.txt", "w", encoding="utf8", errors='replace') as file:
        file.write(text[1:])

    # save the content of the anorexia train
    text = ''
    for file in train:
        with open(dir + "\\" + file, "r+", encoding="utf8", errors='replace') as f:
            dic = json.load(f)
            text += '\n' + open(dir + "\\" + dic["file_id"], "r", errors='replace', encoding='utf8').read().replace('\n', ' ')
    with open(output + "\\training\\anorexia.txt", "w", encoding="utf8", errors='replace') as file:
        file.write(text[1:])

    # split the normal files
    test = normal[:int(split*len(normal))]
    train = normal[int(split*len(normal)):]

    # save the content of the normal test
    text = ''
    for file in test:
        with open(dir + "\\" + file, "r+", encoding="utf8", errors='replace') as f:
            dic = json.load(f)
            text += '\n' + open(dir + "\\" + dic["file_id"], "r", errors='replace', encoding='utf8').read().replace('\n', ' ')
    with open(output + "\\testing\\normal.txt", "w", encoding="utf8", errors='replace') as file:
        file.write(text[1:])

    # save the content of the normal train
    text = ''
    for file in train:
        with open(dir + "\\" + file, "r+", encoding="utf8", errors='replace') as f:
            dic = json.load(f)
            text += '\n' + open(dir + "\\" + dic["file_id"], "r", errors='replace', encoding='utf8').read().replace('\n', ' ')
    with open(output + "\\training\\normal.txt", "w", encoding="utf8", errors='replace') as file:
        file.write(text[1:])



def count_stopwords(path):
    text = ''
    for file in os.listdir(path):
        with open(path + "\\" + file, "r", encoding="utf8", errors='replace') as f:
            text += f.read().replace('\n', ' ')

    for char in r".,;()[]{}:-–?!’'\"“”/&*@\\†‡°#~_|¦¶ˆ^•№§%‰‱¤$™®©":
        text = text.replace(char, '')

    text = [word for word in text.split(' ') if word != ' ']
    stopword = {}
    for word in text:
        if word in stopword:
            stopword[word] += 1
        else:
            stopword[word] = 1

    i = 0
    for w in sorted(stopword, key=stopword.get, reverse=True):
        print(w)


def pre_results(test_data, path):
    text = ""
    for i, post in enumerate(test_data):
        text += post[0]
        text += '\n----------------------'
        text += '\noriginal: ' + post[1][:-4] + '\n'
        for method in ['svc', 'rf', 'mlp', 'lr', 'mnb']:
            text += '!!' + method + str(hex(i)) + '!!\n'
        text += '\n'
    with open(path + "\\result.txt", "w", encoding="utf8", errors='replace') as file:
        file.write(text)

def write_result(prediction, classifier, path):
    dic = {1: "normal", 0: "anorexia"}
    prediction = list(prediction)
    with open(path + "\\result.txt", "r", encoding="utf8", errors='replace') as file:
        text = file.read()
    for i, predict in enumerate(prediction):
        text = text.replace('!!' + classifier + str(hex(i)) + '!!', classifier + ': ' + dic[predict])
    with open(path + "\\result.txt", "w", encoding="utf8", errors='replace') as file:
        file.write(text)

def extract_50_words(train_data, test_data, feature):
	# data = [(content,label)]
	# extract ngrams
    from features import extract_ngrams_args
    count, tfidf, type, n, k = extract_ngrams_args(feature)
    hebrew_voc_translated = ['פרו-אנה', 'צום', 'מרעב', 'להעניש', 'חזק', 'תוכנית ארוחות', 'אנא-אוכל', 'הישנות', 'להחלים', 'חסר ערך', 'תקף' , 'מקנא', 'משקל מטרה', 'משקל יעד סופי', 'כאב', 'ספק', 'מתרסק', 'צינור', 'מדוכא', 'צריכת', 'צבר', 'כוח', 'דיאטנית', 'נצרך', 'חתך' ,'עצמות ירך', 'בוני', 'מותן', 'פוק', 'אד', 'יחידה', 'סולם', 'הדוק', 'תשוקה', 'כפיפות בטן', 'מרווח', 'מגעיל', 'מלא', 'דוהה', 'מנותק', 'מתנגד', 'מפעיל', 'פיתוי', 'מקושטש', 'תרגיל', 'מגביל', 'בלגן', 'גרגרנות', 'להתמודד', 'אובססיה']
    hebrew_voc_original = ['גוף', 'ריסון', 'משקל', 'פחד', 'משלשלים', 'ספורט', 'קלוריות', 'ויתור', 'כישלון', 'הקאה', 'מצוקה', 'אסלה', 'בהמה', 'בולמוס', 'צמתי', 'תסכול', 'לרדת במשקל', 'התקף חרדה', 'הרעבה', 'תפריט', 'דיאטנית', 'אנה', 'מענישה', 'אוכל', 'פרו אנה', 'דימוי גוף', 'שליטה', 'גשמי', 'טוהר', 'קטן', 'למות', 'אנורקסיה', 'תת משקל', 'כמויות', 'טכניקות', 'בליסה', 'רזה', 'רזון', 'שטוחה', 'בולטות', 'זוויתית', 'עצמות', 'ידחפו', 'להקיא', 'גרון', 'פיתוי', 'בולסת', 'עצמות בריח', 'עודף', 'דיאטה']
    english_voc_translated = ['body', 'restraint', 'weight', 'fear', 'laxatives', 'sport', 'calories', 'concession', 'failure ', 'vomiting ', 'distress', 'toilet bowl', 'animal', 'binge', 'i was fasting', 'frustration ', 'lose weight', 'panic attack', 'starving', 'menu', 'dietician', 'anna', 'punishable', 'food', 'pro anna', 'body image', 'control', 'materialistic ', 'purity', 'small', 'die', 'anorexia ', 'underweight', 'quantity', 'technique', 'gluttony', 'thin', 'thinness', 'flat', 'sticking out', 'angular', 'bones', 'will push', 'vomit', 'throat', 'temptation', 'to overeat', 'collarbones', 'over', 'diet']
    english_voc_original = ['pro-ana', 'fasting', 'starve', 'to punish', 'strong', 'meal plan', 'ana-food', 'relapse', 'recover', 'worthless', 'valid', 'jealous', 'GW', 'UGW', 'pain', 'doubt', 'crashing', 'tube', 'depressed', 'intake', 'gained', 'force', 'dietician', 'consumed', 'cut', 'hipbones', 'boney', 'waist', 'puke', 'ed', 'unit', 'scale', 'tight', 'desire', 'crunches', 'caved', 'disgusting', 'full', 'fade', 'stashed', 'resist', 'triggering', 'temptation', 'binged', 'exercise', 'restricting', 'mess', 'gluttony', 'cope', 'obsession']
    vectorizer = CountVectorizer(max_features = 50, analyzer = type, lowercase = False, vocabulary = hebrew_voc_translated, ngram_range = (1, 2))
    x = vectorizer.fit_transform(train_data)
    y = vectorizer.transform(test_data)
    if tfidf == 'tfidf':
        tfidf = TfidfTransformer()
        x = tfidf.fit_transform(x.toarray())
        y = tfidf.transform(y.toarray())
    return x, y


def regex():
    text = open(r"C:\Users\user\Documents\test\dataset\all.txt", 'r', encoding="utf8", errors='ignore').read()

    result = re.findall(r'\w*גר[מם]\w*', text)

    tup = [(word, result.count(word)) for word in set(result)]
    for word in tup:
        print(word[0] + ': ' + str(word[1]))


if __name__ == "__main__":
    regex()

import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem
from pymorphy2 import MorphAnalyzer
import requests

mystem = Mystem()
m = MorphAnalyzer()
SPELL_URL = 'http://speller.yandex.net/services/spellservice.json/checkText'


def get_POS(item):
    if item:
        return str(item)
    else:
        return 'UNKN'


def spell_checker(input_text):
    params = {'text': input_text, 'lang': 'ru'}
    try:
        r = requests.get(SPELL_URL, params=params)
        if r.json():
            out = r.json()
            for i in range(len(out)):
                input_text = input_text.replace(out[i]['word'], out[i]['s'][0])
            return input_text
        else:
            return input_text
    except BaseException as e:
        print('Preprocessing without spell-checking')
        return input_text


def preprocessing(input_text, del_stopwords=True, del_digit=True):
    """
    :input: raw text
        1. lowercase, del punctuation, tokenize
        2. normal form
        3. del stopwords
        4. del digits
    :return: lemmas
    """
    russian_stopwords = set(stopwords.words('russian'))
    words = [x.lower().strip(string.punctuation+'»«–…') for x in word_tokenize(input_text)]
    lemmas = [mystem.lemmatize(x)[0] for x in words if x]

    lemmas_arr = []
    for lemma in lemmas:
        if del_stopwords:
            if lemma in russian_stopwords:
                continue
        if del_digit:
            if lemma.isdigit():
                continue
        lemmas_arr.append(lemma)
    return lemmas_arr

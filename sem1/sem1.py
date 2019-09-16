"""A little search tool"""

import os
import re
from collections import defaultdict
import pymorphy2
import pandas as pd
from string import punctuation
import pickle
from sklearn.metrics.pairwise import cosine_similarity as cos_sim


def preproc(root):
    """
    This function make a new collection of texts, where all words are normalised and cleaned from punctuation
    :param root: path to the directory with corpus of texts
    :return: path to the cleaned texts
    """

    morph = pymorphy2.MorphAnalyzer()
    goal_d = root+'_cleaned'
    if not os.path.exists(goal_d):
        os.mkdir(goal_d)
    for d in os.listdir(root):
        if d[0] != '.':
            goal_sez = os.path.join(goal_d, d)
            if not os.path.exists(goal_sez):
                os.mkdir(goal_sez)
            for f in os.listdir(os.path.join(root, d)):
                filepath = os.path.join(root, d, f)
                with open(filepath, 'r', encoding='utf-8') as t:
                    text = t.read()
                    cleaned = clean_split(text, morph=morph)
                    with open(os.path.join(goal_d, d, f), 'w', encoding='utf-8') as nf:
                        nf.write(' '.join(cleaned))
    return root


def clean_split(text, morph=pymorphy2.MorphAnalyzer()):
    """Clean and normalise using pymorphy2, return a list of words"""
    text = text.lower()
    text = re.sub("[0-9]", "", text)
    text = text.replace('\n', ' ')
    text_arr = text.split(' ')
    text_norm = []
    for word in text_arr:
        if word not in punctuation:
            word = word.strip(punctuation)
            norm_word = morph.parse(word)[0].normal_form
            text_norm.append(norm_word)
    return text_norm


def preproc_req(request):
    """

    :param request: string
    :return: cleaned request
    """
    morph = pymorphy2.MorphAnalyzer()
    req = ' '.join(clean_split(request, morph))
    return req


def iter_by_files(root='/home/seraslain/PycharmProjects/hseinfosearch/friends_cleaned'):
    """
    :param root: path to the directory with cleaned texts
    :return: generator of texts
    """
    for root, dirs, files in os.walk(root):
        for f in files:
            path = os.path.join(root, f)
            with open('files_index.txt', 'a') as idx:
                idx.write(path+'\n')
            with open(path, 'r') as doc:
                t = doc.read()
                yield t


def get_index(root='/home/seraslain/PycharmProjects/hseinfosearch/friends_cleaned'):
    """

    :param root:
    :return: make a pickle with countvectorizer, term-doc array and tf-idf array
    """
    from sklearn.feature_extraction.text import CountVectorizer
    idx = CountVectorizer()
    index_matrix = idx.fit_transform(iter_by_files(root=root)).toarray()
    from sklearn.feature_extraction.text import TfidfTransformer
    index_matrix_tfidf = TfidfTransformer().fit_transform(index_matrix)
    with open('Vectorized', 'wb') as f:
        pickle.dump((idx, index_matrix, index_matrix_tfidf), f)


def search(vectorizer, index_matrix, query=''):
    """

    :param vectorizer: CountVectorizer or TfIdfVectorizer
    :param index_matrix: tf-idf array
    :param query: string
    :return: an array of document paths, sorted by relevance
    """
    if query != '':
        clean_req = [preproc_req(query)]
        with open('files_index.txt', 'r') as f_idx:
            paths = f_idx.read().strip('\n').split('\n')
        q = vectorizer.transform(clean_req).toarray().reshape(1, -1)
        rel_dict = defaultdict()
        for i in range(len(index_matrix)):
            rel_dict[paths[i]] = cos_sim(index_matrix[i].reshape(1, -1), q)
        result = sorted(rel_dict, key=rel_dict.get, reverse=True)
        print(result)
        return search(vectorizer, index_matrix, input('Введите запрос или нажмите enter, чтобы закончить'))
    else:
        return None


def search_season(name, tdf):
    """

    :param name: string
    :param tdf: pd.DataFrame, term-document index
    :return: int, a season, where name appears most frequently
    """
    name = preproc_req(name)
    with open('files_index.txt', 'r') as f_idx:
        paths = f_idx.read().strip('\n').split('\n')
    column = tdf[name]
    column.index = paths
    seasons = defaultdict(int)
    for path in paths:
        s = re.search('[0-9]x[0-9]', path)[0][0]
        seasons[s] += column[path]
    return sorted(seasons, key=seasons.get, reverse=True)


def most_popular(freq_d):
    """

    :param freq_d: a list of words in documents, sorted by frequency
    :return: string, name of most frequent main character
    """
    main_ch = ['рэйчел', 'моника', 'фиби', 'джоуи', 'чендлер', 'росс']
    for i in freq_d:
        i = preproc_req(i)
        if i in main_ch:
            print('Самый популярный персонаж главный герой: ', i)
            break


def sem_answer(termdoc, idx):
    # Ответы на вопросы из семинара:
    tdf = pd.DataFrame(data=termdoc, columns=idx.get_feature_names())
    count_dict = {}
    check = defaultdict(int)
    everytime = set()
    for column in tdf:
        count_dict[column] = tdf[column].sum()
        for i in tdf[column]:
            if i != 0:
                check[column] += 1
    freq_d = sorted(count_dict, key=count_dict.get, reverse=True)
    for i in check:
        if check[i] == 165:
            everytime.add(i)
    print(sorted(check.items(), key=lambda l: l[1], reverse=True)[:10])
    print('Самое частое слово: ', freq_d[0])
    print('Самое редкое слово: ', freq_d[-1])
    print('Есть во всех сериях: ', everytime)
    print('Самый популярный у Чендлера: ', search_season('чендлер', tdf)[0])
    print('Самый популярный у Моники: ', search_season('моника', tdf)[0])
    most_popular(freq_d)


def main():
    # Предобработка текстов
    root = '/home/seraslain/PycharmProjects/hseinfosearch/friends'
    root = preproc(root)
    # Сделаем индексирование
    get_index(root)
    # Открываем индексированные данные
    with open('Vectorized', 'rb') as f:
        idx, termdoc, index_matrix = pickle.load(f)
    sem_answer(termdoc, idx)
    # поиск
    index_matrix = index_matrix.toarray()
    print(search(idx, index_matrix, query=input('Запрос: ')))


if __name__ == "__main__":
    main()

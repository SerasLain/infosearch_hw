from bm25_sparse import clean_split, bm25, idf
from numpy import mean
import numpy as np
import pymorphy2
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import time
from collections import Counter

path = 'quora_question_pairs.csv'
k = 2.0
b = 0.75

def get_train(n=1500):
    """

    :param n: how many docs should be in corpus
    :return:
    """
    corp = pd.read_csv('Cleaned.csv', sep='\t', nrows=n)
    corp.columns = ['text_id', 'text', 'cleaned_text', 'size', 'query', 'isdupl']
    return corp.dropna()


def save_termdoc(corp):
    """
    Saves sparse term-doc matrix and feature_names
    :return: None
    """
    vect = CountVectorizer()
    matrix = vect.fit_transform(corp['cleaned_text']).toarray()
    names = vect.get_feature_names()
    with open('Termdoc_Names', 'wb') as f:
        pickle.dump((matrix, names), f)


def search_loop(query, corp, td, b_c=b):
    """
    Little searchtool, it is not efficient by memory, but it works
    :param query: str, query
    :param corp: pd.DataFrame, corpus with texts
    :param td: pd.DataFrame, term-doc matrix
    :return: dict, bm25 for every doc in the corpus
    """
    q = clean_split(query)
    idf_d = {}
    for word in q:
        if word in td.columns:
            idf_d[word] = sum(td[word] != 0)
    avgdocsize = mean(corp['size'])
    corp_size = corp.shape[0]
    results = {}
    for doc in corp.iterrows():
        i = doc[1]['text_id']
        word_info = {word: (idf_d[word], td.loc[i][word]) for word in idf_d}
        info = doc[1]
        docsize = info['size']
        scores = bm25(k, b_c, docsize, avgdocsize, word_info, corp_size)
        results[i] = scores
    return results


def show_results(indexes, corp):
    """
    prints texts of documents by relevance
    :param indexes: int, indexes of documents
    :param corp:
    :return:
    """
    for i in indexes:
        print(corp.iloc[i[0]-1]['text'], '\nbm25: '+ str(i[1]))


def upd_matrix(td, corp):
    avgdocsize = mean(corp['size'])
    corp_size = corp.shape[0]
    for column in td:
        print(td[column])
        word_doc = sum(td[column] != 0)
        idf_score = idf(corp_size, word_doc)
        for row in td[column].index:
            if td[column][row] != 0:
                doc_size = corp['size'][row]
                w_freq = td[column][row]
                td[column][row] = score_w = idf_score * (
                    (float(w_freq) * (k + 1)) / (float(w_freq) + k * (1 - b + b * (float(doc_size) / avgdocsize))))
    td.to_csv('bm25_matrix', sep='\t')


def code_query(cleaned_query, columns):
    query = Counter(cleaned_query)
    v = []
    for name in columns:
        print(name)
        if name in query:
            v.append(query[name])
        else:
            v.append(0)
    return np.array(v).reshape(7843, 1)


def search_matrix(td_bm, query_v):
    # Не готово. Я так и не понимаю, как работает умножение матрицы на вектор в это случае :(
    data = td_bm.T.values
    result = data * query_v
    return result


def score(test_df, corp, td, b_c):
    l = test_df.shape[0]
    counter = 0
    n = 0
    for elem in test_df.iterrows():
        i = elem[1]['text_id']
        q = elem[1]['query']
        result = search_loop(q, corp, td, b_c=b_c)
        if i in sorted(result, key=result.get)[:10]:
            counter += 1
        n += 1
        if n > 5:
            break
    score = counter / l
    return score


def main():
    # preprocess corpus
    morph = pymorphy2.MorphAnalyzer()
    #write_cleaned(path, morph)
    corp = get_train(5000).dropna()
    #save_termdoc(corp)
    with open('Termdoc_Names', 'rb') as f:
        termdoc, columns = pickle.load(f)
    td = pd.DataFrame(data=termdoc, index=corp.index, columns=columns)
    t = time.time()
    result = search_loop('стать как хорошим геологом?', corp, td)
    s_r = sorted(result.items(), key= lambda x: x[1], reverse=True)[:10]
    print('Первые десять результатов!')
    show_results(s_r, corp)
    t1 = time.time()
    # Сколько времени уходит на поиск одного запроса
    print(t1 - t)

    # Точность поиска

    test_df = corp.loc[corp['isdupl'] == 1]
    print('b=0.75 ', score(test_df, corp, td, 0.75))
    print('b=0 ', score(test_df, corp, td, 0))
    print('b=1 ', score(test_df, corp, td, 1))

    # Попытка реализации через матрицу на вектор
    #upd_matrix(td, corp)
    #td_bm = pd.read_csv('bm25_matrix', sep='\t')
    #v = code_query(clean_split('Война с водой', morph), td_bm.columns)
    #print(v)
    #print(search_matrix(td_bm, v))







if __name__ == "__main__":
    main()

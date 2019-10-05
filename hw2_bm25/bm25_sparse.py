from math import log10
import csv
import pymorphy2
import string
import re
from numpy import mean
import scipy
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import timeit

k = 2.0
b = 0.75
path = 'quora_question_pairs_rus.csv'
morph = pymorphy2.MorphAnalyzer()


def get_document(path, morph):
    """
    yields a text from collection
    :param path: str, path to quora corpus
    """

    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        for line in reader:
            if len(line) != 4:
                print(line)
                pass
            else:
                n, q1, q2, dup = line
                print(n)
                cleaned_q1 = clean_split(q1, morph=morph)
                size = str(len(cleaned_q1))
                yield (n, q1, ' '.join(cleaned_q1), size, q2, dup)


def clean_split(text, morph=pymorphy2.MorphAnalyzer()):
    """Clean and normalise using pymorphy2, return a list of words"""
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    text = text.lower()
    text = regex.sub('', text)
    text = text.replace('\n', ' ')
    text_arr = text.split(' ')
    text_norm = []
    for word in text_arr:
        norm_word = morph.parse(word)[0].normal_form
        text_norm.append(norm_word)
    return text_norm



def write_cleaned(path, morph):
    with open('Cleaned.csv', 'a', encoding='utf-8') as f:
        for line in get_document(path, morph):
            f.write('\t'.join(line) + '\n')
            print(line[:7])


def idf(corp_size, word_count):
    """

    :param corp_size: int, number of doc in the corpus
    :param word_count: int, number of doc with the word in the corpus
    :return: idf score, float
    """
    idf_score = log10((float(corp_size) - float(word_count) + 0.5) / (float(word_count) + 0.5))
    return idf_score


def bm25(k, b, doc_size, avgdocsize, query, corp_size) -> float:
    """

    :param k: float, coefficient
    :param b: float, coefficient
    :param doc_size: lenght of the doc
    :param avgdocsize: float, average length of the doc in collection,
    :param query: dict with query words as keys and tuples with info for every word in query about
    word_doc (how many texts with the word are in collection) and
    w_freq (how many times the word appears in the document) as values
    :param corp_size: int how many doc in collection
    :return:
    """
    scores = []
    for word in query:
        word_doc, w_freq = query[word]
        idf_score = idf(corp_size, word_doc)
        score_w = idf_score * ((float(w_freq) * (k + 1)) / (float(w_freq) + k * (1 - b + b * (float(doc_size) / avgdocsize))))
        scores.append(score_w)
    return sum(scores)


def gen_text(stop=100000):
    """

    :param stop: int, max number of docs in the sample
    :return: yields text
    """
    counter = 0
    with open('Cleaned.csv', 'r', encoding='utf-8') as f:
        for line in f:
            if counter <= stop:
                l = line.strip('\n').split('\t')
                if l[0] != "":
                    text = l[2]
                    counter += 1
            else:
                break
            yield text


def save_termdoc(name):
    """
    Saves sparse term-doc matrix and feature_names
    :return: None
    """
    vect = CountVectorizer()
    matrix = vect.fit_transform(gen_text())
    scipy.sparse.save_npz(name, matrix)
    names = vect.get_feature_names()
    with open('Names', 'wb') as f:
        pickle.dump(names, f)


def rebuild_names():
    """
    Rebuild a list dump as dict dump
    :return: None
    """
    with open('Names', 'rb') as f:
        names = pickle.load(f)
    names_d = {}
    for i in range(len(names)):
        names_d[names[i]] = i
    with open('Names.pckl', 'wb') as f:
        pickle.dump(names_d, f)


def process_query(q, morph):
    """

    :param q: str, query
    :return: quest_map, arr is an array of column indices in termdoc matrix
    """
    query = set(clean_split(q, morph))
    with open('Names.pckl', 'rb') as f:
        names = pickle.load(f)
    quest_map = []
    for word in query:
        try:
            quest_map.append(names[word])
        except KeyError:
            pass
    return quest_map

def get_wordidf_dict(sparse):
    """
    Makes dictionary for every word, where keys are word_ids and values are how many docs have this word
    :param sparse: scipy sparse matrix (csc!)
    :return: dict, word_id:(how many doc with the word in corpus, how many times word appears in corpus)
    """
    rows, cols = sparse.shape
    idf_dict = {}
    for i in range(cols):
        first = sparse.indptr[i]
        next_first = sparse.indptr[i + 1]
        word_doc = next_first - first
        w_freq = sum(sparse.data[first:next_first])
        idf_dict[i] = word_doc
    return idf_dict


def docsizes(stop=100000):
    """

    :param stop: int
    :return: dict
    """
    docsizes_dict = {}
    counter = 0
    with open('Cleaned.csv', 'r', encoding='utf-8') as f:
        for line in f:
            if counter <= stop:
                l = line.strip('\n').split('\t')
                i = l[0]
                size = l[3]
                docsizes_dict[i] = size
                counter += 1
            else:
                break
    return docsizes_dict


def get_query_info(query, doc_id, sparse, word_idf):
    """:param query: str
   :param doc_id: id of document in matrix
   :param sparse:
   :param word_idf: word_doc dict
   :return: word_doc, w_freq
   """
    query_info = {}
    for word_id in process_query(query, morph):
        query_info[word_id] = (word_idf[word_id], get_item(doc_id, word_id, sparse))
    return query_info


def indexed():
    termdoc = scipy.sparse.load_npz('Little_sparse.npz').tocsc()
    word_idf = get_wordidf_dict(termdoc)
    docsizes_dict = docsizes()
    corp_size = len(docsizes_dict)
    avgdocsize = mean([int(i) for i in docsizes_dict.values()])
    with open('Index', 'wb') as f:
        pickle.dump((word_idf, docsizes_dict, corp_size, avgdocsize), f)


def search_loop(query, docsizes_dict, avgdocsize, corp_size, word_idf, termdoc):
    results = {}
    for doc in docsizes_dict:
        query_info = get_query_info(query, doc, termdoc, word_idf)
        score = bm25(k, b, docsizes_dict[doc], avgdocsize, query_info, corp_size)
        results[doc] = score
    return sorted(results, key=results.get, reverse=True)


def get_item(row_index, column_index, matrix):
    # Get column values
    col_start = matrix.indptr[column_index]
    col_end = matrix.indptr[column_index + 1]
    col_values = matrix.data[col_start:col_end]

    # Get row indices of occupied values
    index_start = matrix.indptr[column_index]
    index_end = matrix.indptr[column_index + 1]

    # contains indices of occupied cells at a specific row
    column_indices = list(matrix.indices[index_start:index_end])
    # Find a positional index for a specific column index
    try:
        value_index = column_indices.index(row_index)
        return col_values[value_index]
    except ValueError:
            return 0


def bm25matrix(termdoc, docsize):
    liltd = termdoc.tolil()
    idf_score = idf(corp_size, word_doc)
    score_w = idf_score * (
                (float(w_freq) * (k + 1)) / (float(w_freq) + k * (1 - b + b * (float(doc_size) / avgdocsize))))



def main():
    morph = pymorphy2.MorphAnalyzer()
    # write_cleaned(path, morph)

    #save_termdoc('Little_sparse.npz')
    #rebuild_names()
    indexed()
    termdoc = scipy.sparse.load_npz('Little_sparse.npz').tocsc()
    with open('Index', 'rb') as f:
       word_idf, docsizes_dict, corp_size, avgdocsize = pickle.load(f)
    #results = search_loop('что я должен делать, чтобы быть великим геологом?', docsizes_dict,
                     # avgdocsize, corp_size, word_idf)
    #print(results['6'])
    #print(sorted(results, key=results.get, reverse=True)[:20])

    doc_s = docsizes_dict['']
    t = timeit.Timer(lambda: search_loop('война воды', docsizes_dict, avgdocsize, corp_size, word_idf, termdoc))
    with open('log.txt', 'a', encoding='utf-8') as f:
        f.write('Loop ' + str(t.timeit(1)))


if __name__ == "__main__":
    main()
import pickle
from preproc import preprocessing
from metrics import cos_sim
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from collections import defaultdict

logging.basicConfig(
    filename='Search_project.log',
    format='%(levelname)s %(name)s %(asctime)s : %(message)s',
    level=logging.DEBUG
)
log = logging.getLogger('Tf-idf searcher')


def get_index_matrix():
    log.info("Indexing starts")

    with open("Lemmatized_corpus.pickle", 'rb') as f:
        texts = pickle.load(f)
    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(texts)

    with open("Tf_idf_model.pickle", 'wb') as f:
        pickle.dump((tfidf, matrix), f)
    log.info("Indexing finished!")


class TfIdfSearcher:
    indexed_corpora = None
    model = None

    def __init__(self, indexing=False):
        if indexing:
            log.info('Indexing needed')
            get_index_matrix()
        with open("Tf_idf_model.pickle", 'rb') as f:
            log.info('Model loaded!')
            self.model, self.indexed_corpora = pickle.load(f)

    def transform(self, query):
        lemm_query = ' '.join(preprocessing(query))
        log.info('Lemmatized: %s', lemm_query)
        query_vector = self.model.transform([query])
        return query_vector

    def search(self, query):
        query_vector = self.transform(query).toarray()[0]

        relevance_dict = defaultdict()
        for _id, doc in enumerate(self.indexed_corpora):
            doc_vector = doc.toarray()[0]
            relevance = cos_sim(query_vector, doc_vector)
            relevance_dict[_id] = relevance
        result = sorted(relevance_dict.items(), key=lambda x: x[1], reverse=True)
        log.info('Result_ids %s', ' '.join([str(i) for i in result[:5]]))
        return result


def main():
    from searcher import transform_results

    query = input('Введите запрос: ')
    query = "Как стать геологом?"
    log.info("%s", query)
    arg = bool(int(input('Нужно индексировать корпус? 0, если не нужно, 1, если нужно: ')))
    tfidf = TfIdfSearcher(indexing=arg)
    result = tfidf.search(query)
    print(transform_results(result, 5))

if __name__ == "__main__":
    main()












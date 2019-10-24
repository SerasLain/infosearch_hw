from gensim.models.keyedvectors import KeyedVectors
import logging
import pickle
from preproc import preprocessing
from metrics import cos_sim
import numpy as np

logging.basicConfig(
    filename='Search_project.log',
    format='%(levelname)s %(name)s %(asctime)s : %(message)s',
    level=logging.DEBUG
)
log = logging.getLogger('Fasttext searcher')

fast_model = '181/model.model'


def lookup(doc, wv):
    """
    Checks if all words in model and returns a vector

    :param doc: text string
    :param wv: model.wv, WordVectors
    :return: vector of the document
    """
    checked = []
    for word in doc:
        try:
            word in wv
        except KeyError:
            log.warning("Word not in model: %s", word)
            continue
        checked.append(wv[word])
    vec = np.mean(checked, axis=0)
    return vec


class FasttextSearcher:
    indexed_corpora = None

    def __init__(self, indexing=False):
        if indexing:
            self.model = KeyedVectors.load(fast_model)
            log.info("Indexing starts")
            with open("Lemmatized_corpus.pickle", 'rb') as f:
                texts = pickle.load(f)

            indexed = []
            for doc in texts:
                indexed.append(lookup(doc.split(' '), self.model.wv))

            with open("Fasttext_matrix.pickle", 'wb') as f:
                pickle.dump(indexed, f)
            self.indexed_corpora = indexed

        self.model = KeyedVectors.load(fast_model)
        if not self.indexed_corpora:
            with open("Fasttext_matrix.pickle", 'rb') as f:
                self.indexed_corpora = pickle.load(f)

    def transform(self, query):
        """
        Get vector of the query
        :param query: str
        :return: np.array
        """
        query = preprocessing(query)
        return lookup(query, self.model.wv)

    def search(self, query):
        query_vector = self.transform(query)
        log.debug("query %s", str(query_vector))
        relevance_dict = {}
        for _id, doc in enumerate(self.indexed_corpora):
            relevance = cos_sim(query_vector, doc)
            if type(relevance) is np.float32: # To avoid an array of nan if something goes wrong
                relevance_dict[_id] = relevance

        result = sorted(relevance_dict.items(), key=lambda x: x[1], reverse=True)
        log.info('Result_ids %s', ' '.join([str(i) for i in result[:5]]))
        return result


def main():
    # Example
    from searcher import transform_results

    def get_dist(text1, text2, wv):
        """
        Counts distanse between two documents
        :param text1: arr
        :param text2: arr
        :param wv: model.wv, WordVectors

        """
        t1 = lookup(text1, wv)
        t2 = lookup(text2, wv)
        dist = cos_sim(t1, t2)
        return dist

    fs = FasttextSearcher()
    print(transform_results(fs.search("Что такое число фибоначчи"), 5))
    #print(fs.transform('Ничего не будет хорошо'))


if __name__ == "__main__":
    main()

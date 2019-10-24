import logging
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from tfidfsearcher import TfIdfSearcher
import pickle
import numpy as np
from scipy.sparse import csr_matrix
from metrics import okapi_best_match


logging.basicConfig(
    filename='Search_project.log',
    format='%(levelname)s %(name)s %(asctime)s : %(message)s',
    level=logging.DEBUG
)
log = logging.getLogger('BM25 searcher')

def get_bm25_index_matrix():
    log.info("Indexing starts")

    with open("Lemmatized_corpus.pickle", 'rb') as f:
        texts = pickle.load(f)
    countvec = CountVectorizer()

    matrixtf = countvec.fit_transform(texts)
    matrixidf = TfidfTransformer().fit_transform(matrixtf)
    log.info("Matrixes shapes %s %s", str(matrixtf.shape), matrixidf.shape)

    avglen = np.mean([doc.nnz for doc in matrixtf])
    log.info(avglen)

    indptr = [0]
    indices = []
    data = []
    for doc_tf, doc_idf in zip(matrixtf, matrixidf):
        doc_l = doc_tf.nnz
        indices += list(doc_tf.indices)
        tf_vec, idf_vec = doc_tf.data, doc_idf.data

        bm_25 = []
        for tf, idf in zip(tf_vec, idf_vec):
            bm_25.append(okapi_best_match(tf, idf, doc_l, avglen))
        data += bm_25
        indptr.append(len(indices))

    bm25_matrix = csr_matrix((data, indices, indptr), dtype=float, shape=matrixtf.shape)
    with open("BM25_matrix.pickle", 'wb') as f:
        pickle.dump((countvec, bm25_matrix), f)

    log.info("Indexing finished!")


class BM25Searcher(TfIdfSearcher):


    def __init__(self, indexing=False):
        if indexing:
            get_bm25_index_matrix()

        with open("BM25_matrix.pickle", 'rb') as f:
            self.model, self.matrix = pickle.load(f)

    def search(self, query):
        q = self.transform(query)[0] != 0
        res_vec = np.dot(self.matrix, q.T).toarray()
        relevance_dict = {}
        for _id, score in enumerate(res_vec):
            relevance_dict[_id] = score[0]
        result = sorted(relevance_dict.items(), key=lambda x: x[1], reverse=True)
        log.info('Result_ids %s', ' '.join([str(i) for i in result[:5]]))
        return result


def main():
    from searcher import transform_results
    bm_searcher = BM25Searcher(indexing=False)
    res = bm_searcher.search("Я так больше не могу")
    print(res)
    print(transform_results(res, 5))


if __name__ == "__main__":
    main()

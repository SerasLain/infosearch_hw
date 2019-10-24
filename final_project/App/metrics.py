import numpy as np


def cos_sim(v1, v2):
    """Counts cosine similarity between two vectors"""
    return np.inner(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def okapi_best_match(tf_q, idf_q, doc_len, avgdl, k=0.2, b=0.75):
    """
    Okapi Bestmatch (BM25 by default)
    :param tf_q: word frequency in document
    :param idf_q: inverted document frequency of the word
    :param doc_len: how many words in document
    :param avgdl: average length of document in collection
    :param k: float, coefficient
    :param b: float, coefficient
    :return: float
    """
    score = (tf_q * idf_q * (k + 1)) / (tf_q + k * (1 - b + b * doc_len / avgdl))
    return score

import numpy as np


def cos_sim(v1, v2):
    """Counts cosine similarity between two vectors"""
    return np.inner(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def okapi_best_match(tf_q, idf_q, doc_len, avgdl, k=0.2, b=0.75):
    score = (tf_q * idf_q * (k + 1)) / (tf_q + k * (1 - b + b * doc_len / avgdl))
    return score

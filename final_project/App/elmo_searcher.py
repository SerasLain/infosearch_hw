import pickle
from metrics import cos_sim
import logging
import numpy as np
import tensorflow as tf
from elmo_helpers import tokenize, get_elmo_vectors, load_elmo_embeddings
import csv


elmo_path = 'ELMO'
corpus = "quora_question_pairs_rus.csv"
#
# logging.basicConfig(
#     filename='Search_project.log',
#     format='%(levelname)s %(name)s %(asctime)s : %(message)s',
#     level=logging.INFO
# )
log = logging.getLogger('Elmo searcher')


def get_data_elmo(corpus, stop=5000):
    """
    Проходит по корпусу и токенизирует тексты.

    :param corpus: path to csv file with corpus
    :param stop: int, how many lines we want to get
    :return:
        indexed -> list of list of strings
        id_to_text -> dict, map of text_id to raw text.
        query_to_dupl -> dict, query:id of its duplicate

    """
    indexed = []
    counter = 0

    with open(corpus, 'r', encoding='utf-8') as f:
        r = csv.reader(f)
        for line in r:

            if line[0] == '':
                continue

            _id, text, query, isduplicate = line
            indexed.append(tokenize(text))

            counter += 1
            if counter >= stop:
                break
    return indexed


def crop_vec(vect, sent):
    """
    Crops dummy values

    :param vect: np.array, vector from ELMo
    :param sent: list of str, tokenized sentence
    :return: np.array

    """
    cropped_vector = vect[:len(sent), :]
    cropped_vector = np.mean(cropped_vector, axis=0)
    return cropped_vector


class ELMoSearcher:
    model = None
    matrix = None

    def __init__(self, index=False):
        batcher, sentence_character_ids, elmo_sentence_input = load_elmo_embeddings(elmo_path)
        self.model = (batcher, sentence_character_ids, elmo_sentence_input)

        if index:
            log.info('Indexing starts!')
            cleaned = get_data_elmo(corpus)
            indexed = self.indexing(cleaned)
            with open('ELMO_matrix.pickle', 'wb') as f:
                pickle.dump(indexed, f)
            self.matrix = indexed
            log.info('Indexing finished')

        if not self.matrix:
            with open('ELMO_matrix.pickle', 'rb') as f:
                self.matrix = pickle.load(f)

    def transform(self, query):
        """
        Gets vector of query

        :param query: str
        :return: vector of query
        """
        batcher, sentence_character_ids, elmo_sentence_input = self.model
        q = [tokenize(query)]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            vector = crop_vec(get_elmo_vectors(sess, q, batcher,
                                               sentence_character_ids,
                                               elmo_sentence_input)[0], q[0])
        return vector

    def search(self, query):
        """
        Search query in corpus

        :param: query: str
        :return: list, sorted results
        """
        log.info("Searching...")
        q = self.transform(query)
        log.debug("Shape of query vec %s", str(q.shape))
        result = {}
        for i, doc_vector in enumerate(self.matrix):
            score = cos_sim(q, doc_vector)
            if type(score) is np.float32:
                result[i] = score

        return sorted(result.items(), key=lambda x: x[1], reverse=True)

    def indexing(self, cleaned):
        """
        Indexing corpus
        :param cleaned: list if lists of str, tokenized documents from the corpus

        :return: matrix of document vectors
        """
        batcher, sentence_character_ids, elmo_sentence_input = self.model
        with tf.Session() as sess:
            # It is necessary to initialize variables once before running inference.
            sess.run(tf.global_variables_initializer())
            indexed = []
            for i in range(200, len(cleaned) + 1, 200):
                sentences = cleaned[i - 200: i]
                elmo_vectors = get_elmo_vectors(
                    sess, sentences, batcher, sentence_character_ids, elmo_sentence_input)

                for vect, sent in zip(elmo_vectors, sentences):
                    cropped_vector = crop_vec(vect, sent)
                    indexed.append(cropped_vector)
        return indexed


def main():
    from searcher import transform_results
    elmo_searcher = ELMoSearcher()
    print(transform_results(elmo_searcher.search("Мне все надоело"), 5))


if __name__ == "__main__":
    main()



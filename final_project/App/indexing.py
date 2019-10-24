from preproc import preprocessing
import csv
import pickle
import logging

logging.basicConfig(
    filename='Search_project.log',
    format='%(levelname)s %(name)s %(asctime)s : %(message)s',
    level=logging.DEBUG
)
log = logging.getLogger('Indexing')


def get_preprocessed_data(corpus, stop=5000):
    """
    :param corpus: path to csv file with corpus
    :param stop: int, how many lines we want to get

    :return:
        indexed -> list of lemmatized documents
        id_to_text -> dict, map of text_id to raw text.
    """
    log.info("Indexing starts! Corpus file %s , lines in corpus %d", corpus, stop)

    indexed = []
    id_to_text = {}
    counter = 0

    with open(corpus, 'r', encoding='utf-8') as f:
        r = csv.reader(f)
        for line in r:
            if line[0] == '': # skip header
                continue
            _id, text, query, isduplicate = line
            id_to_text[_id] = text

            indexed.append(' '.join(preprocessing(text)))

            counter += 1

            if counter >= stop:
                break
    return indexed, id_to_text


def write_idtotext(id_to_text):
    with open('Id_to_text.csv', 'a', encoding='utf-8') as f:
        for _id in id_to_text:
            f.write('\t'.join([_id, id_to_text[_id]]) + '\n')

def main():
    cleaned, mapping = get_preprocessed_data("quora_question_pairs_rus.csv")
    with open('Lemmatized_corpus.pickle', 'wb') as f:
        pickle.dump(cleaned, f)

    write_idtotext(mapping)
    log.info('Indexing finished!')


if __name__ == "__main__":
    main()


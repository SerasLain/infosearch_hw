from flask import Flask
from flask import render_template, request
import logging
from tfidfsearcher import TfIdfSearcher
from bm25 import BM25Searcher
from elmo_searcher import ELMoSearcher
from fasttext_searcher import FasttextSearcher

logging.basicConfig(filename="searcher.log",
                    level=logging.INFO,
                    format='%(levelname)s %(name)s %(asctime)s : %(message)s')

log = logging.getLogger("main")

app = Flask(__name__)


def search(query, model_id):
    if model_id == '0':
        model = TfIdfSearcher()

    if model_id == '1':
        model = BM25Searcher()

    if model_id == '2':
        model = FasttextSearcher()

    if model_id == '3':
        model = ELMoSearcher()

    results = model.search(query)
    return results


def transform_results(results, n):
    """
    Form a string of results
    :param results: array of tuples (id, relevance)
    :param n: how many results should be presented
    :return: string of texts and relevance
    """
    if n > 5000:
        n = 5000
    mapping = {}
    with open("Id_to_text.csv", 'r', encoding='utf-8') as f:
        for line in f.readlines():
            _id, doc = line.strip('\n').split('\t')
            mapping[int(_id)] = doc
    results = '\n'.join([str(t[1]) + ': ' + mapping[t[0]] for t in results[:n]])
    return results


@app.route('/')
def index():
    q = ""
    n = ""
    result = ""
    model_name = "Не задано :("
    if request.args:
        log.info('Here we go!')
        q = request.args.get('query')
        if q == "":
            result = "Задан пустой запрос."
        else:
            n = request.args.get('n')
            model_id = request.args.get('model')
            print(model_id)
            if model_id is None:
                model_id = '0'

            models = {'0': 'Tf-idf', '1': 'BM25', '2': 'Fasttext', '3': 'ELMo'}
            model_name = models[model_id]
            log.info("Search %s with %s" % (q, model_name))

            result = transform_results(search(q, model_id), int(n))

        log.info("Results %s", result.split('/n')[:5])
    return render_template('index.html', query=q, result=result.split('\n'), model=model_name, n=n)


if __name__ == '__main__':
    app.run(debug=True)

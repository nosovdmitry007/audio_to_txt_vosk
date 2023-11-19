import sys
import time
from transformers import logging
from recasepunc.recasepunc import CasePuncPredictor
from recasepunc.recasepunc import WordpieceTokenizer
from recasepunc.recasepunc import Config

logging.set_verbosity_error()

def prepinaniy(text):


# text = " ".join(open().readlines())

    predictor = CasePuncPredictor('recasepunc/checkpoint', lang="ru")
    tokens = list(enumerate(predictor.tokenize(text)))

    results = ""
    for token, case_label, punc_label in predictor.predict(tokens, lambda x: x[1]):
        prediction = predictor.map_punc_label(predictor.map_case_label(token[1], case_label), punc_label)
        if token[1][0] != '#':
           results = results + ' ' + prediction
        else:
           results = results + prediction

    return results.strip()

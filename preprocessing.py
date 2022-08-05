import pandas as pd
import numpy as np
from gensim import utils
import gensim.parsing.preprocessing as gsp

# Importing spacy
import spacy
# Loading model
nlp = spacy.load('en_core_web_sm',disable=['parser', 'ner'])
# Lemmatization


def format_amount(value):
    value = str(value)
    if len(str(value)) == 2:
        return float(value[:-2] + '0.' + value[-2:])
    return float(value[:-2] + '.' + value[-2:])

filters = [gsp.strip_punctuation, gsp.strip_multiple_whitespaces, gsp.remove_stopwords, gsp.strip_numeric]
def remove_tags_punctuation_stopword_whitespaces(text):
    text = utils.to_unicode(text)
    text =  gsp.strip_short(text, minsize=2)
    for f in filters:
        text = f(text)
    return text


def clean_preprocessing(data):
    transactions_data = pd.DataFrame(data)
    transactions_data['balance'] = transactions_data['balance'].apply(lambda x: format_amount(x))
    transactions_data['amount'] = transactions_data['amount'].apply(lambda x: format_amount(x))
    transactions_data['clean_narration'] = transactions_data['narration'].apply(lambda x: x.lower())


    transactions_data['clean_narration'] = transactions_data['clean_narration'].apply(lambda x: remove_tags_punctuation_stopword_whitespaces(x))
    transactions_data['date_transform'] = pd.Series(transactions_data['date'])
    transactions_data['clean_narration'] = transactions_data['clean_narration'].apply(lambda x: ' '.join([token.lemma_ for token in list(nlp(x)) ]))
    transactions_data['clean_narration'] = transactions_data['clean_narration'].values.astype('U')


    return transactions_data




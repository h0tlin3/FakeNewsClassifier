from utils.utils import process_line, clean_text
import pickle

import numpy as np

from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

W2V_text = Word2Vec.load('/Users/rustemmatiev/Projects/FakeNewsClassifier/models/word2vec_text.model')
W2V_title = Word2Vec.load('/Users/rustemmatiev/Projects/FakeNewsClassifier/models/word2vec_title.model')

tfidf_text = pickle.load(open("/Users/rustemmatiev/Projects/FakeNewsClassifier/models/vectorizer_text.pk", "rb"))
tfidf_title = pickle.load(open("/Users/rustemmatiev/Projects/FakeNewsClassifier/models/vectorizer_title.pk", "rb"))

def clean_input(text, title):
    title_line = process_line(clean_text(title), tfidf=tfidf_title, w2v=W2V_title)
    text_line = process_line(clean_text(text), tfidf=tfidf_text, w2v=W2V_text)

    final_line = np.concatenate((text_line, title_line),axis=0)
    final_line = final_line.reshape(1,-1)

    return final_line

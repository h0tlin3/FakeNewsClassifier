import pandas as pd
import numpy as np
import re


from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

import string

import pickle

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

def clean_text(text):
    ps = PorterStemmer()
    
    #print('Clearing...')
    
    text = ''.join([c if c not in string.punctuation else ' ' for c in text])
    text = text.lower()
    text = ' '.join([word for word in text.split(' ') if word not in set(stopwords.words('english'))])
    
    #print('Stemming...')
    text = ' '.join([ps.stem(word) for word in text.split(' ')])
    
    text = text.rstrip(' ').lstrip(' ')
    text = re.sub(' +', ' ', text)
    
    #print('Done!')
    
    return text

def process_line(line, upgraded=True, tfidf=None, w2v=None):
    tf_test_str = tfidf.transform([line]).toarray()[0]
    tf_test_dict = {}
    for i in tfidf.vocabulary_.keys():
        tf_test_dict[i] = tf_test_str[tfidf.vocabulary_[i]]

    final_vector = []
    for word in line.split(' '):
        if (word in w2v.wv) and (word in tf_test_dict.keys()):
            if upgraded == True:
                vector = w2v.wv[word]*tf_test_dict[word]
            else:
                vector = w2v.wv[word]
            final_vector.append(vector)

    final_vector = np.array(final_vector)
    final_vector = np.mean(final_vector, axis=0)


    return final_vector

def get_dataset(data, labeled=True, upgraded=True, tfidfs=None, w2vs=None):
    y_labels_dict = {'True':1,'Fake':0}
    
    X = data.CleanText.apply(lambda x: process_line(x, upgraded, tfidf=tfidfs[0], w2v=w2vs[0])).to_numpy()
    X_fixed = []
    for i in range(len(X)):
        try:
            X_fixed.append(list(X[i]))
        except:
            X_fixed.append([0 for i in range(300)])
    X_fixed = np.array(X_fixed)
    
    X_title = data.CleanTitle.apply(lambda x: process_line(x, upgraded, tfidf=tfidfs[1], w2v=w2vs[1])).to_numpy()
    X_title_fixed = []
    for i in range(len(X_title)):
        try:
            X_title_fixed.append(list(X_title[i]))
        except:
            X_title_fixed.append([0 for i in range(300)])

    X_title_fixed = np.array(X_title_fixed)
    
    X = np.concatenate((X_fixed, X_title_fixed),axis=1)
    
    if labeled:
        y = np.array(data.Label.apply(lambda x: y_labels_dict[x]).values)
        return X, y
    
    return X
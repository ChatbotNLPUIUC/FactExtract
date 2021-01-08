from sklearn.datasets import fetch_20newsgroups
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(400)
import nltk
nltk.download('wordnet')
import pandas as pd
stemmer = SnowballStemmer("english")

newsgroups_test = fetch_20newsgroups(subset='test', shuffle = True)

lda_model = gensim.models.LdaModel.load("model")

'''
Testing on unseen documents
'''
num = 100
unseen_document = newsgroups_test.data[num]
#print(unseen_document)

bow_vector = dictionary.doc2bow(preprocess(unseen_document))

for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))

print(newsgroups_test.target[num])

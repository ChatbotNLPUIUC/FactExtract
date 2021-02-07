#Latent Dirichlet Allocation (LDA)
#https://towardsdatascience.com/nlp-extracting-the-main-topics-from-your-dataset-using-lda-in-minutes-21486f5aa925
from sklearn.datasets import fetch_20newsgroups
import gensim
import sklearn
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
import os.path
from os import path

'''

Process:

1. Tokenization: Splitting text into sentences and sentences into words. Lowercase words and remove punct

2. Remove words with length < 3 and stopwords

3. Lemmatize words

4. Stem words

'''

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

# Tokenize and lemmatize
def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
            
    return result

'''
OPTIONAL STEP
Remove very rare and very common words:

- words appearing less than 15 times
- words appearing in more than 10% of all documents
'''
#dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n= 100000)

#bow_doc_x = bow_corpus[20]

#for i in range(len(bow_doc_x)):
    #print("Word {} (\"{}\") appears {} time.".format(bow_doc_x[i][0], 
                                                     #dictionary[bow_doc_x[i][0]], 
                                                     #bow_doc_x[i][1]))


'''
Running LDA with Bag of Words

Parameters:

num_topics 
    - number of requested latent topics to be extracted from the training corpus.

id2word
    - mapping from word ids (integers) to words (strings) to determine vocab size, as well as debugging and topic printing.

workers 
    - number of extra processes to use for parallelization. Uses all available cores by default

alpha and eta are hyperparameters that affect sparsity of the document-topic (theta) and topic-word (lambda) distributions. (default value is 1/num_topics)
    - Alpha: per document topic distribution.
    - High alpha: Every document has a mixture of all topics(documents appear similar to each other).
    - Low alpha: Every document has a mixture of very few topics

    - Eta: per topic word distribution.
    - High eta: Each topic has a mixture of most words(topics appear similar to each other).
    - Low eta: Each topic has a mixture of few words.

passes
    - number of training passes through the corpus

For example, if the training corpus has 50,000 documents, chunksize is 10,000, passes is 2, then online training is done in 10 updates
Each being of 10,000 documents per update and repeated:
    #1: 0 - 9999
    #2: 10000 - 19999
    #3: 20000 - 29999
    #4: 30000 - 39999
    #5: 40000 - 49999
    #6 to #10 repeats this

'''

# LDA multicore 

'''
Train your lda model using gensim.models.LdaMulticore and save it to lda_model
'''
#lda_model =  gensim.models.LdaMulticore(bow_corpus, 
                                   #num_topics = 8, 
                                   #id2word = dictionary,                                    
                                   #passes = 10,
                                   #workers = 2)

def process(dataName):
    #doc_train = dataName
    processed_docs = []

    for doc in dataName:
        processed_docs.append(preprocess(doc))
    
    '''
    Create a dictionary from 'processed_docs' containing the number of times a word appears 
    in the training set using gensim.corpora.Dictionary and call it 'dictionary'
    '''
    dictionary = gensim.corpora.Dictionary(processed_docs)
    return processed_docs, dictionary

def train(processed, dictionary):

    '''
    Gensim: doc2bow

    Converts list of words into bag of words format (list of 2-tuples [token_id, token_count])
    '''
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed]

    '''
    Split one to training and one to doing LDA
    '''
    # LDA mono-core -- fallback code in case LdaMulticore throws an error on your machine
    lda_model = gensim.models.LdaModel(bow_corpus, 
                                        num_topics = 10, 
                                        id2word = dictionary,                                    
                                        passes = 50)
    return lda_model

if __name__ == "__main__":
    traindocs = fetch_20newsgroups(subset='train', shuffle = True)
    testdocs = fetch_20newsgroups(subset='test', shuffle = True)
    count = 0
    for files in os.listdir("SepTrain"):
        print(files)
        f = open("SepTrain/"+files, "r")
        np.append(traindocs.data,f.read())
        np.append(traindocs.target,20)
        np.append(traindocs.filenames,"SepTrain/"+files)
        np.append(traindocs.DESCR,"")
        np.append(traindocs.target_names,"textbook")
        f.close()

    for files in os.listdir("SepTest"):
        print(files)
        f = open("SepTest/"+files, "r")
        np.append(testdocs.data,f.read())
        np.append(testdocs.target,20)
        np.append(testdocs.filenames,"SepTest/"+files)
        np.append(testdocs.DESCR,"")
        np.append(testdocs.target_names,"textbook")
        f.close()
    
    processed, dictionary = process(traindocs.data)
    if not path.exists("LDAmodel"):
        train(processed, dictionary).save("LDAmodel")
    else:
        lda_model = gensim.models.LdaModel.load("LDAmodel")
        for idx, topic in lda_model.print_topics(-1):
            print("Topic: {} \nWords: {}".format(idx, topic ))
            print("\n")

        '''
        Testing on unseen documents
        '''
        num = 100
        unseen_document = testdocs.data[num]
        bow_vector = dictionary.doc2bow(preprocess(unseen_document))

        for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
            print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))

        print(testdocs.target[num])

# Python program to generate word vectors using Word2Vec

import re
from gensim.models import Word2Vec
import numpy as np
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import warnings

warnings.filterwarnings(action='ignore')

import nltk

nltk.download('punkt_tab')
nltk.download('stopwords')
stop_words = stopwords.words('english')

PATH = ""

# Reads ‘alice.txt’ file
sample = open("./alice.txt")
s = sample.read()

# Replaces escape character with space
f = s.replace("\n", " ")

data = []

# iterate through each sentence in the file
for i in sent_tokenize(f):
    temp = []

    # tokenize the sentence into words
    for j in word_tokenize(i):
        if j not in stop_words:
            temp.append(j.lower())

    data.append(temp)

n = len(data)
for idx in range(n):
    data[idx] = [re.sub(r'[^\w\s]', '', i) for i in data[idx] if len(re.sub(r'[^\w\s]', '', i)) > 2]

if __name__ == '__main__':
    # Create CBOW model
    model1 = gensim.models.Word2Vec(data, min_count=1,
                                    vector_size=100, window=5)

    # Create Skip Gram model
    model2 = gensim.models.Word2Vec(data, min_count=1, vector_size=100,
                                    window=5, sg=1)

    model1.wv.most_similar('alice', topn=10)

    print("Cosine similarity between 'alice' " +
          "and 'wonderland' - Skip Gram : ",
          model2.wv.similarity('alice', 'wonderland'))

    print("Cosine similarity between 'alice' " +
          "and 'machines' - Skip Gram : ",
          model2.wv.similarity('alice', 'machines'))

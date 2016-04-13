import sqlite3
import pandas as pd
import numpy as np

import string
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
#nltk.download('punkt')
#nltk.download('stopwords')

con = sqlite3.connect('data/database.sqlite')

# https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words
def preprocess(text):
	tokens = text.str.lower().str.split()
	tokens = word_tokenize(tokens)
	return tokens

def bag_of_words(data):
	bow = {}
	tokens = preprocess(data['Text'])
	for t in tokens:
	   if word in bow:
		bow[t] += 1.0
	   else:
		bow[t] = 1.0
	return bow

low = pd.read_sql_query("""
SELECT Score, Summary, Text
FROM Reviews
WHERE Score < 3
""", con)

med = pd.read_sql_query("""
SELECT Score, Summary, Text
FROM Reviews
WHERE Score == 3
""", con)

high = pd.read_sql_query("""
SELECT Score, Summary, Text
FROM Reviews
WHERE Score > 3
""", con)

lowBOW = bag_of_words(low)
medBOW = bag_of_words(med)
highBOW = bag_of_words(high)


print highBOW


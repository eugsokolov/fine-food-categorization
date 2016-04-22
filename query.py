import sqlite3
import pandas as pd
import numpy as np

import string
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
#nltk.download('punkt')
#nltk.download('stopwords')

con = sqlite3.connect('data/database.sqlite')

def preprocess(text):
	tokens = word_tokenize(text)
	s1 = dict((k,1) for k in stopwords.words('english'))
	s2 = dict((k,1) for k in string.punctuation)
	tokens = [i for i in tokens if i not in s1 and i not in s2]
	#st = PorterStemmer()
	#tokens = [st.stem(i) for i in tokens]
	tokens = ngrams(tokens, 2)
	return tokens

def bag_of_words(data):
	bow = {}
	for text in data:
	  tokens = preprocess(text)
	  for token in tokens:
	     if token in bow:
		bow[token] += 1.0
	     else:
		bow[token] = 1.0
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

lowBOW = bag_of_words(low['Text'])

for k,v in lowBOW.iteritems():
	if v > 150:
		print k

#medBOW = bag_of_words(med['Text'])
#highBOW = bag_of_words(high['Text'])



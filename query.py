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

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

con = sqlite3.connect('data/database.sqlite')

def processModel(model, train):
	count = 0
	for traincv, testcv in cv:
		pass

	accuracy = 1-count/float(1)
	print "accuracy: " + str(accuracy)

def preprocess(text):
	tokens = word_tokenize(text)
	s1 = dict((k,1) for k in stopwords.words('english'))
	s2 = dict((k,1) for k in string.punctuation)
	tokens = [i for i in tokens if i not in s1 and i not in s2]
	#st = PorterStemmer()
	#tokens = [st.stem(i) for i in tokens]
	#tokens = ngrams(tokens, 2)
	return tokens

def partition(x):
	if x < 3:
		return 'negative'
	else:
		return 'positive'

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

q = pd.read_sql_query("""
SELECT Score, Summary, Text
FROM Reviews
WHERE Score != 3
""", con)

reviews = q['Text'] 
score = q['Score']
score = score.map(partition)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(reviews, score, test_size=0.2, random_state=42)

#print bag_of_words(Xtrain) # returns hash BOW
### Training Set

#vectorizer = CountVectorizer(max_features = 500, stop_words='english')
#matrix  = vectorizer.fit_transform(train)

### Test Set


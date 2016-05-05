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

positive_words = np.genfromtxt('Positive_&_Negative_Words.csv', dtype = 'str', usecols = (0, ), skip_header = 1, delimiter = ',') # Unique positive words from Harvard lexicon
negative_words = np.genfromtxt('Positive_&_Negative_Words.csv', dtype = 'str', usecols = (1, ), skip_header = 1, delimiter = ',') # Unique negative words from Harvard lexicon

print(positive_words)
print('\n---------------------------------------\n')
print(negative_words)
def processModel(model, train):
	count = 0
	for traincv, testcv in cv:
		pass

	accuracy = 1-count/float(1)
	print("accuracy: " + str(accuracy))

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
                bow[token]+=1.0
	     else:
                bow[token]=1.0
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

### Training Set
#print bag_of_words(Xtrain) # returns hash BOW
#l = [preprocess(i) for i in Xtrain)] # returns list of preprocessed items in Xtrain

count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(corpus)



### Test Set
#test = [preprocess(i) for i in Xtrain]
#X_new_counts = count_vect.transform(test)
#X_test_tfidf = tfidf_transformer.transform(X_new_counts)




### Model Analysis
prediction = dict()


import sqlite3
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import string
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import SA_Functions as saf
import pickle

con = sqlite3.connect('data/database.sqlite')
q = pd.read_sql_query("""
	SELECT Score, Summary, Text
 	FROM Reviews
 	WHERE Score != 3
	limit 10
 	""", con)
 	
reviews = q['Text']
score = q['Score']


# The reviews are preprocessed (as of now: lowercased, stemmed)
reviews_preprocessed, scores_posneg, lengths = saf.preprocess(reviews, score)

train_reviews_preprocessed = reviews_preprocessed[0:10]
train_scores_posneg = scores_posneg[0:10]

positive_words, negative_words = saf.count_top(train_reviews_preprocessed, train_scores_posneg)


#saf.save_obj(positive_words, 'positive_words')
#saf.save_obj(negative_words, 'negative_words')
positive_words_check = saf.load_obj('positive_words')
negative_words_check = saf.load_obj('negative_words')

print(positive_words)
print('---------------------------------------')
print(positive_words_check)

"""
for i in range(100):
	print(reviews_preprocessed[i])
	print('\n')
"""

# Split reviews into a training set and testing set 
reviews_train, reviews_test, reviews_train_labels, reviews_test_labels = train_test_split(reviews_preprocessed, scores_posneg, test_size = 0.2, random_state = 42)

# Apply Tf-IDf weighting scheme to the training reviews & test reviews
train_tfidf, test_tfidf = saf.tfidf_weights(reviews_train, reviews_test)

# Create dictionary for storing different model performances
prediction = dict()

# Apply Logistic Regression model for classification on test set
prediction_LR = saf.LogReg(train_tfidf, reviews_train_labels, test_tfidf)
# Display confusion matrix for Logistic Regression
saf.confusion(prediction_LR, reviews_test_labels)

# Apply Multinomial Naive Bayes model for classification on test set
prediction_MNB = saf.MultiNB(train_tfidf, reviews_train_labels, test_tfidf)
# Display confusion matrix for Multinomial Naive Bayes
saf.confusion(prediction_MNB, reviews_test_labels)

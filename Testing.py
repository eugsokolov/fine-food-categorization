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

con = sqlite3.connect('data/database.sqlite')
q = pd.read_sql_query("""
	SELECT Score, Summary, Text
 	FROM Reviews
 	WHERE Score != 3
 	limit 10
 	""", con)
 	
reviews = q['Text'] 
score = q['Score']

# Load in positive/negative word lexicons
# stemmer = PorterStemmer()
# positive_words, negative_words = saf.load_lexicons()
# positive_words = dict((stemmer.stem(k),1) for k in positive_words)
# negative_words = dict((stemmer.stem(k),1) for k in negative_words)

# The reviews are preprocessed (as of now: lowercased, stemmed, rid of punctuation & stopwords)
reviews_preprocessed, scores_posneg, lengths = saf.preprocess(reviews, score)
 
reviews_preprocessed = np.array(reviews_preprocessed)
scores_posneg = np.array(scores_posneg)
lengths = np.array(lengths)

# np.save('reviews_preprocessed.npy', reviews_preprocessed)
# np.save('scores_posneg.npy', scores_posneg)
# np.save('lengths.npy', lengths)

# reviews_preprocessed = np.load('reviews_preprocessed.npy')
# scores_posneg = np.load('scores_posneg.npy')
# lengths = np.load('lengths.npy')

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

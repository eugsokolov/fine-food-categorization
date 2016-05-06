import sqlite3
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import string
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

def partition(x):
	"""
	This function rates reviews less than 3 negative, and above 3 positive.
	"""
	if x < 3:
		return (-1)
	else:
		return (1)
		
def tokenize_stem(text):
	"""
	This function makes a review lower case and stems it.
	"""
	tokens = word_tokenize(text.lower())
	stemmer = PorterStemmer()
	s1 = dict((k,1) for k in stopwords.words('english'))
	s2 = dict((k,1) for k in string.punctuation)
	tokens_stemmed = [stemmer.stem(i) for i in tokens if i not in s1 and i not in s2]
	return ' '.join(tokens_stemmed)

def preprocess_stem(reviews, score):
	"""
	This function preprocesses the reviews by lowercasing & stemming them.
	This function rates reviews less than 3 negative, and above 3 positive.
	"""
	print('Preprocessing...')
	score = score.map(partition)
	reviews_stemmed = [tokenize_stem(text) for text in reviews]
	print('Preprocessing Finished!\n')
	return reviews_stemmed, score

# Tf-IDf Weighting Scheme

def tfidf_weights(preprocessed_training_reviews, preprocessed_test_reviews):
	"""
	This function applies the Tf-IDf weighting scheme to the training & test documents.
	"""
	from sklearn.feature_extraction.text import CountVectorizer
	from sklearn.feature_extraction.text import TfidfTransformer
	
	count_vect = CountVectorizer()
	tfidf_trans = TfidfTransformer()

	training_word_counts = count_vect.fit_transform(preprocessed_training_reviews)
	train_tfidf = tfidf_trans.fit_transform(training_word_counts)
	
	test_word_counts = count_vect.transform(preprocessed_test_reviews)
	test_tfidf = tfidf_trans.transform(test_word_counts)
	
	return train_tfidf, test_tfidf
	
# Model Functions

def LogReg(train_review_features, train_review_labels, test_review_features):
	"""
	This function applies a logistic regression model for classification.
	"""
	print('Applying Logistic Regression...')
	from sklearn import linear_model
	logreg = linear_model.LogisticRegression(C=1e5, class_weight='balanced')
	logreg.fit(train_review_features, train_review_labels)
	prediction = logreg.predict(test_review_features)
	print('Finished Logistic Regression!\n')
	return prediction
	
def MultiNB(train_review_features, train_review_labels, test_review_features):
	"""
	This function applies a multinomial naive bayes model for classification.
	"""
	print('Applying Multinomial Naive Bayes...')
	from sklearn.naive_bayes import MultinomialNB
	model = MultinomialNB(fit_prior=True).fit(train_review_features, train_review_labels)
	prediction = model.predict(test_review_features)
	print("Finished Multinomial Naive Bayes!\n")
	return prediction

# Display confusion matrices for models

def confusion(model_prediction, test_review_labels):
	"""
	This function computes confusion matrix for each model applied during testing.
	"""
	test_review_labels = np.array(test_review_labels).astype('str')
	#for model, predicted in model_prediction_dict.items():
 	predicted = np.array(model_prediction).astype('str')
 	print(metrics.classification_report(test_review_labels, predicted))
 	print('\n')

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
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics

con = sqlite3.connect('data/database.sqlite')

def partition(x):
	if x < 3:
		return (-1)
	else:
		return (1)

q = pd.read_sql_query("""
SELECT Score, Summary, Text
FROM Reviews
WHERE Score != 3
""", con)

reviews = q['Text']
score = q['Score']
score = score.map(partition)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(reviews, score, test_size=0.2, random_state=42)

intab = string.punctuation
outtab = "                                "
trantab = str.maketrans(intab, outtab)

stemmer = PorterStemmer()
from nltk.corpus import stopwords
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = word_tokenize(text)
    #tokens = [word for word in tokens if word not in stopwords.words('english')]
    stems = stem_tokens(tokens, stemmer)
    return ' '.join(stems)

### Training Set
corpus = []
i=0
for text in Xtrain:
    text = text.lower()
    text = text.translate(trantab)
    text=tokenize(text)
    corpus.append(text)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(corpus)


tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

### Test Set
test_set = []
for text in Xtest:
    text = text.lower()
    text = text.translate(trantab)
    text=tokenize(text)
    test_set.append(text)

X_new_counts = count_vect.transform(test_set)
X_test_tfidf = tfidf_transformer.transform(X_new_counts)

### Model Analysis
prediction = dict()

from sklearn import linear_model
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X_train_tfidf, Ytrain)
prediction['Logistic'] = logreg.predict(X_test_tfidf)

from sklearn import svm
clf = svm.SVR().fit(X_train_tfidf, Ytrain)
prediction['SVM'] = logreg.predict(X_test_tfidf)

from sklearn.ensemble import RandomForestClassifier
#rf = RandomForestClassifier(n_estimators=2).fit(X_train_tfidf, Ytrain)
#prediction['Random Forest'] = rf.predict(X_test_tfidf)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(X_train_tfidf, Ytrain)
prediction['Multinomial'] = model.predict(X_test_tfidf)

from sklearn.naive_bayes import BernoulliNB
model = BernoulliNB().fit(X_train_tfidf, Ytrain)
prediction['Bernoulli'] = model.predict(X_test_tfidf)

y = np.array(Ytest).astype('str')
for model, predicted in prediction.items():
 pred = np.array(predicted).astype('str')
 print(model)
 print(metrics.classification_report(y, pred))

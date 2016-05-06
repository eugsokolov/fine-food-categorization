import sqlite3
import pandas as pd
import numpy as np
#nltk.download('punkt')
#nltk.download('stopwords')
from sklearn.cross_validation import train_test_split
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

### Preprocessing
import string
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

# TODO add lemmatization

s1 = dict((k,1) for k in stopwords.words('english'))
s2 = dict((k,1) for k in string.punctuation)
def tokenize(text):
    tokens = word_tokenize(text.lower())
    #tokens = [word for word in tokens if word not in stopwords.words('english')]
    tokens = [i for i in tokens if i not in s1 and i not in s2]
    #tokens = ngrams(tokens, 2)
    stems = stem_tokens(tokens, stemmer)
    return ' '.join(stems)

### Training Set
corpus = []
for text in Xtrain:
    text = tokenize(text)
    corpus.append(text)
exit()

#create a matrix of token counts
from sklearn.feature_extraction.text import CountVectorizer
X_train_counts = CountVectorizer.fit_transform(corpus)

#create a matrix of tfidf
from sklearn.feature_extraction.text import TfidfTransformer
X_train_tfidf = TfidfTransformer.fit_transform(X_train_counts)

### Test Set
test_set = []
for text in Xtest:
    text = tokenize(text)
    test_set.append(text)

X_new_counts = CountVectorizer.transform(test_set)
X_test_tfidf = TfidfTransformer.transform(X_new_counts)

print("done preprocessing")

### Feature Reduction
# take top N thousand words
# use only semantic lexicons

### Model Analysis
prediction = dict()

from sklearn import linear_model
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X_train_tfidf, Ytrain)
prediction['Logistic'] = logreg.predict(X_test_tfidf)
print("done logistic")

#from sklearn import svm
#clf = svm.SVR().fit(X_train_tfidf, Ytrain)
#prediction['SVM'] = logreg.predict(X_test_tfidf)
#print("done svm")

#from sklearn.ensemble import RandomForestClassifier
#rf = RandomForestClassifier(n_estimators=2).fit(X_train_tfidf, Ytrain)
#prediction['Random Forest'] = rf.predict(X_test_tfidf)
#print("done rf")

#from sklearn.naive_bayes import MultinomialNB
#model = MultinomialNB().fit(X_train_tfidf, Ytrain)
#prediction['Multinomial'] = model.predict(X_test_tfidf)
#print("done mnb")

#from sklearn.naive_bayes import BernoulliNB
#model = BernoulliNB().fit(X_train_tfidf, Ytrain)
#prediction['Bernoulli'] = model.predict(X_test_tfidf)
#print("done bnb")

y = np.array(Ytest).astype('str')
for model, predicted in prediction.items():
 pred = np.array(predicted).astype('str')
 print(model)
 print(metrics.classification_report(y, pred))

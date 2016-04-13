import sqlite3
import pandas as pd
import numpy as np
import nltk
import string

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer

con = sqlite3.connect('database.sqlite')

def bag_of_words(data):

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





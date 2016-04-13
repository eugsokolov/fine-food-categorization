import sqlite3
import pandas as pd
import numpy as np
import nltk
import string

con = sqlite3.connect('data/database.sqlite')

def bag_of_words(data):

 return 0

low = pd.read_sql_query("""
SELECT Score, Summary, Text
FROM Reviews
WHERE Score < 3
""", con)

print(low['Summary'])

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





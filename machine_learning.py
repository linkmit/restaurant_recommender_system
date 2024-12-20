# for Linux operation systems

# !sudo apt install espeak
# !sudo apt install espeak-ng

import numpy as np
import pandas as pd
import Levenshtein
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pyttsx3
import time

FILE_PATH = 'dialog_acts.dat'

df = pd.read_table(FILE_PATH, header=None, names=['combined'])

## First dataset
df['combined'] = df['combined'].astype(str)
df[['dialog_act', 'utterance_content']] = df['combined'].str.split(n=1, expand=True)
df = df[['dialog_act', 'utterance_content']]
df['utterace_content'] = df['utterance_content'].str.lower()

# Test-train split
x_train, x_test, y_train, y_test = train_test_split(df['utterance_content'], df['dialog_act'], test_size=0.15)

## Second Dataset
df_nodupes = df.drop_duplicates(subset=['utterance_content'])

# Test-train split
x_train_nodupes, x_test_nodupes, y_train_nodupes, y_test_nodupes = train_test_split(df_nodupes['utterance_content'], df_nodupes['dialog_act'], test_size=0.15)

### Machine Learning ########################################################################
# Bag of Words
vectorizer = CountVectorizer()
vectorizer.fit(x_train)
x_train_bow = vectorizer.transform(x_train)

# Logistic Regression Classifier
clf_logreg = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
clf_logreg.fit(x_train_bow, y_train)
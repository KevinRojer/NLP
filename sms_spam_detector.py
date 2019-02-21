import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

# load the data
print("Loading the data...")
dir_path = '~/Documents/Data/SmsSpam/'
file = dir_path + 'spam.csv'
data = pd.read_csv(file)

print(data.head(10))

# Pre-process the data
tf_idf = TfidfVectorizer()
X = tf_idf.fit_transform(sentences)


# Split the data into train and test sets
Xtrain = X[:-100,]
Xtest = X[-100:,]
Ytrain = Y[:-100,]
Ytest = Y[-100:,]

# Train the ML model
print("Training the models...")
model_nb = MultinomialNB()
model_ab = AdaBoostClassifier()
model_nb.fit(Xtrain, Ytrain)
model_ab.fit(Xtrain, Ytrain)

# Evaluate the model
score_nb = model_nb.score(Xtest, Ytest)
score_ab = model_ab,score(Xtest, Ytest)
print("Classification rate for NB: {0}".format(score_nb))
print("Classification rate for AB: {0}".format(score_ab))

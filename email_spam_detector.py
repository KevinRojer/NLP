import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier

# Load the data
print("Loading the data...\n")
dir_path = '~/Documents/Data/Spambase/'
file = dir_path + 'spambase.txt'
data = pd.read_csv(file).values

# Randomly split the data into train and test sets
np.random.shuffle(data) #inplace shuffle (different every time)

# first 48 columns are relevant features, last column is label
X = data[:, :48]
Y = data[:, -1]

# Deterministic split, last 100 rows are test set
Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest = X[-100:,]
Ytest = Y[-100:,]

# Train the ML model
print("Training ML model Naive Bayes...")
model = MultinomialNB()
model.fit(Xtrain, Ytrain)
model_score = model.score(Xtest, Ytest)
print("Classification rate  for NB: {0}\n".format(model_score))

print("Training ML model AdaBoost...")
model_ada = AdaBoostClassifier()
model_ada.fit(Xtrain, Ytrain)
ada_score = model_ada.score(Xtest, Ytest)
print("Classification rate  for Adaboost: {0}\n".format(ada_score))

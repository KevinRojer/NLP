import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud

# load the data
print("Loading the data...")
dir_path = '~/Documents/Data/SmsSpam/'
file = dir_path + 'spam.csv'
data = pd.read_csv(file, encoding='ISO-8859-1') #because of invalid characters

# Pre-process the data
# drop unnecessary columns
data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

# rename columns
data.columns = ['labels', 'data']

# Create binary labels
data['b_labels'] = data['labels'].map({'ham': 0, 'spam': 1})

# Create features
#count_vectorizer = CountVectorizer(decode_error='ignore')
#X = count_vectorizer.fit_transform(data['data'])
tf_idf = TfidfVectorizer(decode_error='ignore') # ignore invalid character error
X = tf_idf.fit_transform(data['data'])
Y = data['b_labels'].values

# Split the data into train and test sets (or, implement cross-validation)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)

# Train the ML model
print("Training the models...")
model_nb = MultinomialNB()
model_ab = AdaBoostClassifier()
model_nb.fit(Xtrain, Ytrain)
model_ab.fit(Xtrain, Ytrain)

# Evaluate the model
score_nb = model_nb.score(Xtest, Ytest)
score_ab = model_ab.score(Xtest, Ytest)
print("Classification rate for NB: {0}".format(score_nb))
print("Classification rate for AB: {0}".format(score_ab))

# Visualize the data
def visualize_text(label):
    words = ''
    for sms in data[data['labels'] == label]['data']:
        sms = sms.lower()
        words += sms + ' '
    wordcloud = WordCloud(width=600, height=400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

visualize_text('spam')
visualize_text('ham')

# Analysis to determine what we wrongly predict
data['predictions_nb'] = model_nb.predict(X)
data['predictions_ab'] = model_ab.predict(X)

# Determine sms that by-passed the classifier
print("Missclassified spam text Naive-bayes:")
sneaky_spam_nb = data[(data['predictions_nb'] == 0) & (data['b_labels'] == 1)]['data']
for sms in sneaky_spam_nb:
    print(sms)

print("Missclassified spam text Adaboost:")
sneaky_spam_ab = data[(data['predictions_ab'] == 0) & (data['b_labels'] == 1)]['data']
for sms in sneaky_spam_ab:
    print(sms)


print("Missclassified ham text Naive-bayes:")
not_spam_nb = data[(data['predictions_nb'] == 1) & (data['b_labels'] == 0)]['data']
for sms in not_spam_nb:
    print(sms)

print("Missclassified ham text Adaboost:")
not_spam_ab = data[(data['predictions_ab'] == 1) & (data['b_labels'] == 0)]['data']
for sms in not_spam_ab:
    print(sms)

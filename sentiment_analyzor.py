import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup

word_net_lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')

# Functions
def my_tokenizer(sentence):
    """ Tried using the NLTK tokenizer before. However, the tokinzer does
        not lowercase the words, which increases our vocabulary. Therefore,
        I create a customized tokenizer.
    """
    sentence = sentence.lower()
    tokens = nltk.tokenize.word_tokenize(sentence)
    tokens = [t for t in token if len(t) > 2] # Filter noise words
    tokens = [word_net_lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if t not in stop_words] # remove stop words
    return (tokens)

def tokens_to_vector(tokens, label):
    x = np.zeros(len(word_index_map) + 1)
    for t in tokens:
        i = word_index_map[t]
        x[1] += 1
    x = x / x.sum()
    x[-1] = label
    return (x)

# Load the data (http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html)
print("Loading the data...")
dir_path = '~/Documents/Data/AmazonReviews/electronics/'
file_negative = dir_path + 'negative.review'
file_positive = dir_path + 'positive.review'
negative_reviews = BeautifulSoup(open(file_negative).read())
positive_reviews = BeautifulSoup(open(file_positive).read())
negative_reviews = negative_reviews.findAll('review_text')
positive_reviews = positive_reviews.findAll('review_text')

# There are more positive reviews than negative reviews
# We could use two approach:
# 1.) Undersample positive reviews
#   np.random.shuffle(positive_reviews)
#   positive_reviews = positive_reviews[:len(negative_reviews)]
# 2.) Oversample the engative reviews
diff = len(positive_reviews) - len(negative_reviews)
# Determine unique index for vocabulary mapping.
idxs = np.random.choice(len(negative_reviews), size=diff)
extra = [negative_reviews[i] for i in idxs]
negative_reviews += extra

# Create vocabulary mapping. Each unique word has a unique index.
word_index_map = {}
current_index = 0

positive_tokenized = []
negative_tokenized = []

for review in positive_reviews:
    tokenized_reviews = my_tokenizer(review.text)
    positive_tokenized.append(tokenized_reviews)
    for token in tokenized_reviews:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1

for review in negative_reviews:
    tokenized_reviews = my_tokenizer(review.text)
    negative_tokenized.append(tokenized_reviews)
    for token in tokenized_reviews:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1

N = len(positive_tokenized) + len(negative_tokenized)

data = np.zeros(N, len(word_index_map) + 1)
i=0
for tokens in positive_tokenized:
    xy = tokens_to_vector(tokens, 1)
    data[i,:] = xy
    i += 1

for tokens in negative_tokenized:
    xy = tokens_to_vector(tokens, 0)
    data[i,:] = xy
    i += 1

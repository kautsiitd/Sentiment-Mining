import sys
import random
import re
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# process tweets
def process_tweet(tweet):
	# removing links and @names from tweets
	tweet = re.sub(r"(?:\@|https?\://)\S+", "", tweet)
	# converting to lowercase and removing repeateting letters and special characters and removing -s, -es, -ly etc
	tweet = [stemmer.stem(re.sub(r'([a-z])\1+', r'\1', e)) for e in tokenizer.tokenize(tweet.lower())]
	# removing stop words and words with length less than 4 lengths
	tweet = [e for e in tweet if (len(e) >= 4 and e not in stop_words)]
	return tweet

with open('training_parameter_V.pickle', 'rb') as f:
	V = pickle.load(f)
print 1

with open('training_parameter_logis.pickle', 'rb') as f:
	logis = pickle.load(f)
print 2

with open(str(sys.argv[1])) as f:
# with open('small.csv') as f:
	content = f.readlines()
print 3
# testing tweets
test_tweets 	= [x.decode("utf8","replace") for x in content]
# test_tweets     = [process_tweet(x[1]).decode("utf8","replace") for x in content[int(.9*l):]]
# testing
print 4
x_test = V.transform(test_tweets)
z = logis.predict(x_test)
print 5
f = open(str(sys.argv[2]), 'w')
for x in z:
	f.write(str(x))
	f.write('\n')
f.close()	

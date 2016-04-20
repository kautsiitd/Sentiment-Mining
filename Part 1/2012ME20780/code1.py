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

# different variables
tokenizer = RegexpTokenizer(r'[a-z]+')
stop_words = stopwords.words('english')
stemmer = SnowballStemmer("english")

# process tweets
def process_tweet(tweet):
	# removing links and @names from tweets
	tweet = re.sub(r"(?:\@|https?\://)\S+", "", tweet)
	# converting to lowercase and removing repeateting letters and special characters and removing -s, -es, -ly etc
	tweet = [stemmer.stem(re.sub(r'([a-z])\1+', r'\1', e)) for e in tokenizer.tokenize(tweet.lower())]
	# removing stop words and words with length less than 4 lengths
	tweet = [e for e in tweet if (len(e) >= 4 and e not in stop_words)]
	return tweet

# learning file
# with open('kejriwal_train.txt') as f:
with open('training.csv') as f:
# with open('small.csv') as f:
	content = [x.split(',',1) for x in f.readlines()]
print "1"
random.shuffle(content)
l = len(content)
# training tweets
train_sentiments = [int(x[0][1:-1]) for x in content[:int(.2*l)]]
train_tweets 	 = [x[1].decode("utf8","replace") for x in content[:int(.2*l)]]
# train_tweets = [process_tweet(x[1]).decode("utf8","replace") for x in content[:int(.3*l)]]
print "2"
V = TfidfVectorizer(ngram_range=(1, 2))
x_train = V.fit_transform(train_tweets)
y_train = train_sentiments
logis   = LogisticRegression()
logis.fit(x_train, y_train)
# storing data
with open('training_parameter_V.pickle', 'wb') as f:
	pickle.dump(V, f)
with open('training_parameter_logis.pickle', 'wb') as f:
	pickle.dump(logis, f)

# testing tweets
test_tweets 	= [x[1].decode("utf8","replace") for x in content[int(.9*l):]]
# test_tweets     = [process_tweet(x[1]).decode("utf8","replace") for x in content[int(.9*l):]]
test_sentiments = [int(x[0][1:-1]) for x in content[int(.9*l):]]
# testing
x_test = V.transform(test_tweets)
y_test = test_sentiments
l_test = len(y_test)
z = logis.predict(x_test)
print z
correct = 0
incorrect = 0
for i in range(l_test):
	if(y_test[i]==z[i]):
		correct+=1
	else:
		incorrect+=1
print (correct*100.0)/(incorrect+correct)
import random
import re
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from collections import Counter

# print tweets
def print_it(x):
	print x
	# for i in x.most_common:
	# 	print i

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

# selecting feature words
def get_feature_words():
	all_words = []
	# getting frq of all words
	for (words, sentiment) in tweets:
		all_words.extend(words)
	return nltk.FreqDist(all_words).keys()

# Classifier
def extract_features(document):
	document_words = set(document)
	features = {}
	for word in feature_words:
		features['contains(%s)' % word] = (word in document_words)
	return features

# reading training file
with open('training.csv') as f:
# with open('small.csv') as f:
	content = [x.split(',',1) for x in f.readlines()]
print "1"
random.shuffle(content)
l = len(content)
# processing tweets	
tweets = [(process_tweet(x[1]),x[0]) for x in content[:int(.09*l)]]
print "2"
# finding word_features based on frequency
feature_words = get_feature_words()
print "3"
training_set = nltk.classify.apply_features(extract_features, tweets)
print "4"
classifier = nltk.NaiveBayesClassifier.train(training_set)
print "5"
print classifier.show_most_informative_features(1000)
correct=0
incorrect=0
for tweet in content[int(.99*l):]:
	print correct,incorrect
	print classifier.classify(extract_features(process_tweet(tweet[1]))),tweet[0]
	if(classifier.classify(extract_features(process_tweet(tweet[1]))) == tweet[0]):
		correct+=1
	else:
		incorrect+=1
print (correct*100.0)/(incorrect+correct)
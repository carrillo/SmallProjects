__author__ = "Fernando Carrillo"
__email__ = "fernando at carrillo.at"

import pandas as pd 
import numpy as np 
import re
from matplotlib import pyplot as plt


from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
from sklearn import metrics
###########################
# Train a text classifier. 
###########################
class MessageClassifier(object):
	"""docstring for MessageClassifier"""
	def __init__(self):
		self.clf = None

	def train(self, train_X, train_y, labels,
		pipeline=Pipeline([('vect', CountVectorizer(encoding='utf-8', decode_error='strict')), ('tfidf', TfidfTransformer()), ('nb', MultinomialNB())]), 
		param_grid={'vect__ngram_range': [(1, 2)], 'nb__alpha': [10**-4,10**-3,10**-2]}
		): 
		"""
		Trains a classifier on training data. 

		Parameters
    	----------
    	train_X : array 
        	Text to train on 

    	train_Y : array
        	Text classes in numerical form 

        labels : array
        	Text class names

        pipeline : Pipeline
        	Classification pipeline
        	The 

        param_grid = dictionary 
        	Metaparameters to grid search. 
		"""
		grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, verbose=3, cv=5, n_jobs=-1).fit(train_X, train_y)
		print( ('Best score %s of estimator %s') % (grid_search.best_score_, grid_search.best_estimator_))
		self.clf = grid_search.best_estimator_
		self.clf.fit(train_X, train_y)
		self.labels = labels

	def dump(self, filename): 
		"""
		Dump the model to a file. 

		Parameters
    	----------
    	filename : String 
        	Filename for saved model. 
		"""
		_ = joblib.dump(self.clf, filename + '_model.p', compress=9)
		_ = joblib.dump(self.labels, filename + '_labels.p', compress=9)

	def load(self, filename): 
		"""
		Loads model from file 

		Parameters
    	----------
    	filename : String 
        	Filename for saved model. 
		"""
		self.clf = joblib.load(filename + '_model.p')
		self.labels = joblib.load(filename + '_labels.p')

	def test(self, valid_X, valid_y): 
		"""
		Tests model on validation data set

		Parameters
		----------
		valid_X : array 
			Text to classify 

		valid_y : array 
			Label of text

		valid_targetnames : array 
			Text of targets
		"""
		predicted = self.clf.predict(valid_X)
		print(metrics.classification_report(valid_y, predicted, target_names=self.labels))		

	def predict_proba(self, text): 
		"""
		Classifies text provided. 
		"""
		return self.clf.predict_proba(text)

	def time_prediction(self, text=['God is love', 'OpenGL on the GPU is fast', 'What is this sentence about?'], iterations=100):
		"""
		times execution of prediction 
		"""
		from time import time
		t0 = time()
		for _ in range(iterations): 
			self.predict_proba(text)
		duration = time() - t0
		print('Took %fs to classify %d phrases %d times.' %(duration, len(text), iterations) )

	# def top_keys(self, n=10):
	# 	for i, category in enumerate(self.labels)
	# 		print(i)

	def top_keywords(self, n=10): 
		"""
		Get the top n keywords per class. 
		"""
		feature_names = np.array(self.clf.named_steps['vect'].get_feature_names())
		for i, category in enumerate(self.labels):
			keywords = np.argsort(self.clf._final_estimator.coef_[i])[-n:]
			print("%s: %s" % (category, " ".join(feature_names[keywords])))

if __name__ == '__main__':
	# Load toy data set. Replace training with sth different. 
	twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=32)
	twenty_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=32)

	tc = MessageClassifier()
	#pipeline=Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('nb', MultinomialNB())])
	#param_grid={'vect__ngram_range': [(1, 2)], 'nb__alpha': [10**-3]}

	# pipeline=Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('svm', SGDClassifier(loss='modified_huber', penalty='l2', n_iter=5, random_state=32))])
	# param_grid={'vect__ngram_range': [(1, 2)], 'svm__alpha': [10**-8, 10**-6, 10**-4, 10**-2]}
	# tc.train(train_X=twenty_train.data, train_y=twenty_train.target, labels=twenty_train.target_names, pipeline=pipeline, param_grid=param_grid)

	#tc.dump('data/tweet_clf_svm')
	tc.load('data/tweet_clf_svm')

	tc.test(valid_X=twenty_test.data, valid_y=twenty_test.target)

	tc.time_prediction(iterations=10)
	#tc.top_keywords(n=10)

	docs_new = ['God is love', 'OpenGL on the GPU is fast', 'What is this sentence about?', 'RT neuroph: Cool Neuroph neural network visualization using Gephi contributed by Fernando Carrillo http://t.co/MG8LPxzFNr', 'Eloquent commit message: "(\u256f\u2035\u25a1\u2032)\u256f\ufe35\u253b\u2501\u253b" (perfect, too: important information that cannot be deduced from the diff...) https://t.co/w6SosMdTrA']
	predicted = tc.predict_proba(docs_new)
	for doc, probs in zip(docs_new, predicted):
		print('%r => %s' % (doc, twenty_train.target_names[np.argmax(probs)]))

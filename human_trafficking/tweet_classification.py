import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics
###########################
# Train a text classifier. 
###########################
# Load toy data set. Replace training with sth different. 
twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=32)
twenty_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=32)

# Generate bag-of-word representation and train classifier 
# 1. Count occurances in the entire corpus
# 2. tf-idf 
#	a) Get term frequencies (normalizing to document length) -> term frequency tf
#	b) Downscale weights which occur in many document -> inverse document frequency idf 
steps = [('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('nb', MultinomialNB())] 
clf = Pipeline(steps)
param_grid = {'vect__ngram_range': [(1, 2)], 'nb__alpha': [10**-4,10**-3,10**-2]}
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, verbose=3, cv=5).fit(twenty_train.data, twenty_train.target)
print( ('Best score %s of estimator %s') % (grid_search.best_score_, grid_search.best_estimator_))

# Fit the entire data to the best predictor. 
clf = grid_search.best_estimator_
#clf.fit(twenty_train.data, twenty_train.target)

# Test on unseen data and report performance
predicted = clf.predict(twenty_test.data)
print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))


docs_new = ['God is love', 'OpenGL on the GPU is fast']
predicted = clf.predict_proba(docs_new)
for doc, probs in zip(docs_new, predicted):
	print('%r => %s' % (doc, twenty_train.target_names[np.argmax(probs)]))




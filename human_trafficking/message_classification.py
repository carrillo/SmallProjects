__author__ = "Fernando Carrillo"
__email__ = "fernando at carrillo.at"


import numpy as np
import scipy as sp
from time import time 

from message_classifier import MessageClassifier

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import exists
from db_tables import Base, Message, create_sqlite_db

# Set-up connection to message data_base 
print('Connect to database.')
engine = create_engine('sqlite:///data/testuser.db')
Base.metadata.bind = engine 
DBSession = sessionmaker(bind=engine)
session = DBSession()
print('Connect to database. Done.')

# Load trained classfier. At this point this is trained on the 
# newsgroup data, but will be trained on a data-set of the domain of interest. 
print('Load classifier.')
mc = MessageClassifier()
mc.load('data/tweet_clf_nb')
print('Load classifier. Done.')

messages = [getattr(x, "text").encode('utf-8') for x in session.query(Message).all()]
print('Predict.')
t0 = time() 
probabilities = mc.predict_proba(messages)
print('Predicted %d messages in %fs.' % (len(messages), (time()-t0)))

for message, probs in zip(messages, probabilities): 
	if sp.stats.entropy(probs) < 1.0: 
		print('%s with a probability of\t%f\t%r' % (mc.labels[np.argmax(probs)], np.max(probs), message))
		print('Entropy of prediction: %f' % (sp.stats.entropy(probs)))
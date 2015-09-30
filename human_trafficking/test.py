import pandas as pd 
import numpy as np 
import cPickle as pickle 
import gzip 
import re 
from matplotlib import pyplot as plt


from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn import metrics

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import exists

from db_tables import Base, User, Connection, Message, Location, create_sqlite_db

source_node_name = 'fcarrillo81'
engine = create_engine('sqlite:///data/' + source_node_name + '.db')
Base.metadata.bind = engine 
DBSession = sessionmaker(bind=engine)
session = DBSession()

for message in session.query(Message).all(): 
	print message.text.encode('utf-8')
	text = str(message.text.strip().encode('utf-8'))
	text = re.sub('[@#]', '', text) # remove twitter specific characters
	#text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text) # remove links 
	print( text )

	
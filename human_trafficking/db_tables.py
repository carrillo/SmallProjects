__author__ = "Fernando Carrillo"
__email__ = "fernando at carrillo.at"

import os 
import sys
from sqlalchemy import Column, ForeignKey, Boolean, String, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import relationship
		
Base = declarative_base()

class User(Base):
	"""
	Set-up user table: 
	Columns: username and isVisited
	"""
	__tablename__ = 'user'
	name = Column(String(50), primary_key=True)
	visited = Column(Boolean, nullable=False)
	depth = Column(Integer, nullable=False)

class Connection(Base): 
	"""
	Set-up connection table: 
	id, source node (user1), target node (user2), weight 
	"""
	__tablename__ = 'connection'
	id = Column(Integer, primary_key=True)
	user_1_name = Column(String(50), ForeignKey('user.name'))
	user_2_name = Column(String(50), ForeignKey('user.name'))
	weight = Column(Integer)
	
	user_1 = relationship("User", foreign_keys=[user_1_name])
	user_2 = relationship("User", foreign_keys=[user_2_name])

class Location(Base):
	"""
	Set-up location table: 
	id, user_name, geojson, location  
	"""
	__tablename__ = 'location'
	id = Column(Integer, primary_key=True)
	user_name = Column(String(50), ForeignKey('user.name'))
	geojson = Column(Text)
	location = Column(Text)

	user = relationship("User", foreign_keys=[user_name])

class Message(Base): 
	"""
	Set-up location table: 
	id, user_name, text
	"""
	__tablename__ = 'message'
	id = Column(Integer, primary_key=True)
	user_name = Column(String(50), ForeignKey('user.name'))
	text = Column(Text)

	user = relationship("User", foreign_keys=[user_name])

# Create an engine that stores data in the local path 
engine = create_engine('sqlite:///data/twitter_search.db')

# Create all tables in the engine 
Base.metadata.create_all(engine)

def create_sqlite_db(name):
	# Create an engine that stores data in the local path 
	engine = create_engine('sqlite:///data/' + name + '.db')

	# Create all tables in the engine 
	Base.metadata.create_all(engine)



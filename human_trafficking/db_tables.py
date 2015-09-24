import os 
import sys
from sqlalchemy import Column, ForeignKey, Boolean, String, Integer 
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


# Create an engine that stores data in the local path 
engine = create_engine('sqlite:///data/twitter_search.db')

# Create all tables in the engine 
Base.metadata.create_all(engine)

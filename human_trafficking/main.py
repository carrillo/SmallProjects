import pandas as pd
import numpy as np 


from users import Twitter_user
from twitter_auth import Twitter_auth
from json_parser import ParseJSON

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import exists

from db_tables import Base, User, Connection

class Twitter_search(object):
	"""
	Performs a search around a given starting username of depth n. 
	"""
	def __init__(self, user_name, db_session, depth=1): 
		self.user_name = user_name
		self.db_session = db_session
		self.depth = depth
		self.add_user(user_name)
			
	def add_user(self, user_name): 
		"""
		Adds a user to the database.
		1. Check if already present. 
		2. If not, add. 
		"""
		if not self.db_session.query(exists().where(User.name == user_name)).scalar(): 
			new_user = User(name=user_name, visited=False)
			self.db_session.add(new_user)
			self.db_session.commit()

	def get_next_user(self): 
		"""
		Get the next user who has not been visited. 
		"""
		try:
			return(self.db_session.query(User).filter(User.visited == False).one().name)
		except Exception, e:
			return 

	def get_user(self, user_name): 
		return(self.db_session.query(User).filter(User.name == user_name).one())
		
	def set_user_visited(self, user_name): 
		"""
		Adds the visited flag to the specified user. 
		"""
		self.db_session.query(User).filter(User.name == user_name).update({"visited": True})
		self.db_session.commit()

	def add_connection(self, user_name1, user_name2, weight): 
		"""
		Add connection between user 1 and user 2 in CONNECTION relationship. 
		"""
		new_connection = Connection(user_1_name=user_name1, user_2_name=user_name2, weight=weight )
		self.db_session.add(new_connection)
		self.db_session.commit()

	def run(self, message_count=100, dump=True): 
		"""
		1. Connect to twitter api 
		2. Download messages
		3. Dump messages
		"""
		u1 = self.get_next_user()
		user = Twitter_user(u1, Twitter_auth().authenticate())
		user.load(message_count)
		if (dump): user.dump('data/' + u1 + '.json')

		nodes = user.get_nodes()
		for name, weight in nodes.iteritems(): 
			u2 = name.strip('\@')
			self.add_user(user_name=u2)
			self.add_connection(user_name1=u1, user_name2=u2, weight=weight)
		self.set_user_visited(user_name=u1)
		
if __name__ == '__main__':

	# Set up the database connection
	engine = create_engine('sqlite:///data/twitter_search.db')
	Base.metadata.bind = engine 
	DBSession = sessionmaker(bind=engine)
	session = DBSession()

	search = Twitter_search(user_name='fcarrillo81', db_session=session)
	search.run()

	# user = Twitter_user("test", Twitter_auth().authenticate())
	# user.load(100)
	# user.dump("data/test.json")

	# nodes = user.get_nodes()
	# print(nodes)

	# locations = user.get_locations()
	# print(locations)

	# messages = user.get_messages()
	# print(messages)
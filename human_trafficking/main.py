import pandas as pd
import numpy as np 


from users import Twitter_user
from twitter_auth import Twitter_auth
from json_parser import ParseJSON

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import exists

from db_tables import Base, User, Connection, Message, Location

class Network_search(object):
	"""
	Performs a search around a given starting username of depth n. 
	"""
	def __init__(self, user_object, db_session, depth=1): 
		"""
		@param user_object Subclass of User_base class
		@param db_session database connection 
		@depth depth of search. 
		"""
		self.user_object = user_object 
		self.root_node = getattr(user_object, 'user_name')
		self.db_session = db_session
		self.cur_depth = 0
		self.add_user(self.root_node)
		self.cur_depth += 1 
		self.max_depth = depth
		
			
	def add_user(self, user_name): 
		"""
		Adds a user to the database.
		1. Check if already present. 
		2. If not, add. 
		"""
		if not self.db_session.query(exists().where(User.name == user_name)).scalar(): 
			new_user = User(name=user_name, visited=False, depth=self.cur_depth)
			self.db_session.add(new_user)
			self.db_session.commit()

	def get_next_user(self): 
		"""
		Get the next user who has not been visited. 
		"""
		try:
			return(self.db_session.query(User).filter(User.visited == False).first().name)
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

	def add_message(self, user_name, text): 
		"""
		Add message to MESSAGE relationship. 
		"""
		new_message = Message(user_name=user_name, text=text)
		self.db_session.add(new_message)
		self.db_session.commit()

	def add_location(self, user_name, geojson, location): 
		"""
		Add message to LOCATION relationship. 
		"""
		if (geojson == "nan"): geojson = "NULL"
		if (location == "nan"): location = "NULL"
		new_location = Location(user_name=user_name, geojson=geojson, location=location)
		self.db_session.add(new_location)
		self.db_session.commit()

	def run(self, message_count=100, dump=True): 
		"""
		1. Download and dump messages
		2. Get nodes and add to users and connection tables. 
		3. Get location and add to location table. 
		"""
		u1 = self.get_next_user()
		setattr(self.user_object, 'user_name', u1)
		#user = Twitter_user(u1, Twitter_auth().authenticate())
		self.user_object.load(message_count)
		if (dump): self.user_object.dump('data/' + u1 + '.json')

		for text in self.user_object.get_messages(): 
			self.add_message(u1, text)

		nodes = self.user_object.get_nodes()
		for name, weight in nodes.iteritems(): 
			u2 = name.strip('\@')
			self.add_user(user_name=u2)
			self.add_connection(user_name1=u1, user_name2=u2, weight=weight)

		locations = self.user_object.get_locations()
		for i in locations.index: 
			g = str(locations['geojson'][i])
			l = str(locations['location'][i])
			self.add_location(user_name=u1, geojson=g, location=l)

		self.set_user_visited(user_name=u1)
		
if __name__ == '__main__':

	# Set up the database connection
	engine = create_engine('sqlite:///data/twitter_search.db')
	Base.metadata.bind = engine 
	DBSession = sessionmaker(bind=engine)
	session = DBSession()

	# Add root node user. 
	user = Twitter_user('fcarrillo81', Twitter_auth().authenticate())
	search = Network_search(user_object=user, db_session=session)
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
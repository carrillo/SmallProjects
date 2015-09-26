__author__ = "Fernando Carrillo"
__email__ = "fernando at carrillo.at"

import tweepy 
import json 
import abc 
import re 
import pandas as pd
import numpy as np

from twitter_auth import Twitter_auth
from nltk.tokenize import TweetTokenizer
from json_parser import ParseJSON

class User_base(object): 

	def __init__(self, user_name): 
		self.user_name = user_name

	@abc.abstractmethod
	def load(self, args): 
		"""
		Load data from the social net. For example tweets. 
		"""
		return

	@abc.abstractmethod
	def dump(self, file_name):
		"""
		Dumps data to file. 
		"""
		return

	@abc.abstractmethod
	def get_nodes(self): 
		"""
		Get network nodes. These are typically connections in social media. 
		Implementation in subclasses should return nodes with weights. 
		"""
		return 	

	@abc.abstractmethod
	def get_locations(self):
		"""
		Get the location of the user. 
		Implementation in subclasses should return a list of gps_coordintates. 
		If not available, return Null. 
		"""
		return 

	@abc.abstractmethod
	def get_messages(self): 
		"""
		Get messages by the user. 
		Implementation should return a list of strings containing one 
		String per message. 
		"""
		return 

class Twitter_user(User_base):
	"""
	Generates a twitter user instance
	"""
	def __init__(self, user_name, twitter_auth):
		super(Twitter_user, self).__init__(user_name)
		self.auth = twitter_auth

	def load(self, tweet_count=100):
		self.api = tweepy.API(self.auth)
		try:
			t = self.api.user_timeline(screen_name = self.user_name, count = tweet_count)
		except Exception, e:
			t = [] 
		
		tweets = [] 
		for tweet in t: 
			tweets.append(dict(tweet._json))
		self.tweets = tweets
	
	def dump(self, file_name): 
		with open(file_name, 'w') as outfile: 
			for tweet in self.tweets: 
				json.dump(tweet, outfile)	
				outfile.write("\n")

	def get_nodes(self, from_tweets=True, max_from_followers=0, user_id_to_screen_name=False): 
		"""
		Get screen_names of users mentioned in the tweets and of followers. 
		Be careful the from_followers is really expensive.
		"""
		nodes = [] 
		# Retrieve usernames from tweet texts
		if (from_tweets): 
			for tweet in self.tweets: 
				nodes = nodes + re.findall('\B\@\w+', str(tweet))

		# Retrive usernames from followers.
		if (max_from_followers > 0): 
			followers = [] 
			for page in tweepy.Cursor(self.api.followers_ids, screen_name=self.user_name).pages(): # Get all follower ids. 
				followers.extend(page)
				if (len(followers) >= max_from_followers):
					break
			
			if (user_id_to_screen_name): # Do not convert user ids into screen names, too expensive in API terms. Just add the user_ids. 
				for user_id in followers: 
					nodes.extend("@" + self.api.get_user(user_id).screen_name)
			else:
				nodes.extend(followers)	

		return pd.Series(nodes).value_counts(sort=True)

	def get_locations(self): 
		"""
		Returns the gps coordinates of the tweets. 
		1. Get data from coordinates field. 
		2. Get data from location field. 
		"""
		rows_list = [] 
		for tweet in self.tweets:
			geostring = str(tweet["coordinates"])
			if (geostring == "None"): geostring = np.nan
			rows_list.append({"geojson":geostring, "location":tweet["user"]["location"]} )
		loc = pd.DataFrame(data=rows_list, columns=["geojson","location"])
		loc = loc.applymap(lambda x: np.nan if isinstance(x, basestring) and not x else x)
		loc.dropna(how='all', inplace=True)
		return(loc)

	def get_messages(self): 
		"""
		Returns the messages as a list
		"""
		return(self.get_field_values("text"))

	def get_field_values(self, key): 
		"""
		Retrieve data of specific field
		"""
		l = []
		for tweet in self.tweets: 
			l.append(tweet[key])
		return l 

	def get_words(self): 
		tknzr = TweetTokenizer()

	def get_coordinates(self): 
		return(self.get_field_values("coordinates"))

	def get_geo(self): 
		return(self.get_field_values("geo"))

if __name__ == '__main__':
	user = Twitter_user('fcarrillo81', Twitter_auth().authenticate())
	user.load()
	loc = user.get_locations()
	for i in loc.index: 
		g = str(loc['geojson'][i])
		l = str(loc['location'][i])
		if (g == "nan"): print("match")
		print('%s\t%s' % (g, l))
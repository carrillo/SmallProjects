import pandas as pd
from tweepy import OAuthHandler 

class Twitter_auth(object):
	"""
	Performs twitter connection with given credentials. 
	"""
	def __init__(self):
		self.access_token = ""
		self.access_token_secret = ""
		self.consumer_key = ""
		self.consumer_secret = ""

	def authenticate(self): 
		"""
		Set up the connection
		"""
		auth = OAuthHandler(self.consumer_key, self.consumer_secret)
		auth.set_access_token(self.access_token, self.access_token_secret)
		return(auth)


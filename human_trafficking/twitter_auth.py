import pandas as pd
from tweepy import OAuthHandler 

class Twitter_auth(object):
	"""
	Performs twitter connection with given credentials. 
	"""
	def __init__(self, credential_file):
		self.auth_params = {}
		with open(credential_file, 'rU') as data:
			for line in data:
				entry = line.split(',')
				self.auth_params[entry[0]] = entry[1]

	def authenticate(self): 
		"""
		Set up the connection
		"""
		self.auth = OAuthHandler(self.auth_params["consumer_key"], self.auth_params["consumer_secret"])
		self.auth.set_access_token(self.auth_params["access_token"], self.auth_params["access_token_secret"])

	def get_auth(self): 
		return self.auth
				
		
if __name__ == '__main__':
	auth = Twitter_auth("data/twitter_credentials.csv")
	auth.authenticate()



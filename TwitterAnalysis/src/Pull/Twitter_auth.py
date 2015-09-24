import pandas as pd
from tweepy import OAuthHandler 

class Twitter_auth(object):
	"""
	Performs twitter connection with given credentials. 
	"""
	def __init__(self):
		self.access_token = "61738905-dlt5UVevRhpXxaM4fghrarLFxVgUGkU38oiG6NTZo"
		self.access_token_secret = "uDPDqUPOHEiVTl2Hzp60zBdbShidkeb7Qalv8B5MztOu0"
		self.consumer_key = "u2eWoIroKfwmpiWfp2zyuJxKt"
		self.consumer_secret = "7dhEWLsFv6kYdKdkoG5pfQ7IZBtxcnEZyPMXLGaGedudWntn9B"

	def authenticate(self): 
		"""
		Set up the connection
		"""
		auth = OAuthHandler(self.consumer_key, self.consumer_secret)
		auth.set_access_token(self.access_token, self.access_token_secret)
		return(auth)


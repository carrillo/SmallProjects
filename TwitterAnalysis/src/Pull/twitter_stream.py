# Download data from Twitter API. Example code from: http://adilmoujahid.com/posts/2014/07/twitter-analytics/

# import stuff
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler 
from tweepy import Stream
import sys
import getopt
from Twitter_auth import Twitter_auth


# Variables that contains the user credentials to access Twitter API 
access_token = ""
access_token_secret = ""
consumer_key = ""
consumer_secret = ""

#
class STdOutListener(StreamListener):
	"""
	This is a basic listener that just prints received tweets to stdout 
	"""
	def on_data(self, data):
		print data
		return True

	def on_error(self, status):
		print status 


if __name__ == '__main__':

	# Read filter keywords from argv array. 
	keywords = sys.argv[1:]
	#print(keywords)     	
	
	# This handles Twitter authentification and the connection to Twitter Streaming API 
	l = STdOutListener()


	# Read credentials from file: 
	auth_params = {} 
	with open("data/twitter_credentials.csv", 'rU') as data:
			for line in data:
				entry = line.split(',')
				auth_params[entry[0]] = entry[1]

	
	auth = OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)
	
	stream = Stream(auth, l)

	# This line filters Twitter streams to track keywords. 
	try:
		stream.filter(track=keywords)
	except Exception, e:
		print >> sys.stderr, e 
		pass
	
	
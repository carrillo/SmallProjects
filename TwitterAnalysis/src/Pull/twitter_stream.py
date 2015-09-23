# Download data from Twitter API. Example code from: http://adilmoujahid.com/posts/2014/07/twitter-analytics/

# import stuff
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler 
from tweepy import Stream
import sys
import getopt
from twitter_auth import Twitter_auth

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

	# Get login from Twitter_auth module. 
	stream = Stream(Twitter_auth().authenticate(), l)

	# This line filters Twitter streams to track keywords. 
	try:
		stream.filter(track=keywords)
	except Exception, e:
		print >> sys.stderr, e 
		pass
	
	
# Download data from Twitter API. Example code from: http://adilmoujahid.com/posts/2014/07/twitter-analytics/

# import stuff
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler 
from tweepy import Stream
import sys
import getopt


# Variables that contains the user credentials to access Twitter API 
access_token = "61738905-dlt5UVevRhpXxaM4fghrarLFxVgUGkU38oiG6NTZo"
access_token_secret = "uDPDqUPOHEiVTl2Hzp60zBdbShidkeb7Qalv8B5MztOu0"
consumer_key = "u2eWoIroKfwmpiWfp2zyuJxKt"
consumer_secret = "7dhEWLsFv6kYdKdkoG5pfQ7IZBtxcnEZyPMXLGaGedudWntn9B"

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
	auth = OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)
	stream = Stream(auth, l)

	# This line filters Twitter streams to capture data by the keywords: 'python', 'javascript', 'ruby'
	try:
		stream.filter(track=keywords)
	except Exception, e:
		print >> sys.stderr, e 
		pass
	
	
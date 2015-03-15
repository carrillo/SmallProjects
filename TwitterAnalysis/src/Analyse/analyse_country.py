import pandas as pd
from parseJson import ParseJSON
#import matplotlib.pyplot as plt
import re 
import cStringIO

def keyword_in_text(tweet, keyword):
	"""
	Checks if a given keyword is in the given tweet text. 
	
	1. Checks main text. 
	2. If not present check retweet text, if present. 
	
	:param tweet: Tweet in Dictionary format as provided from ParseJSON
	:param keyword: A keyword or regular expression to search for 
	:returns: True if text contains keyword, False otherwise
	"""
	k = keyword.lower()
	if(re.search(k,tweet['text'].lower())):
		return True
	elif "retweeted_status" in tweet and re.search(k,tweet["retweeted_status"]["text"].lower()):
		return True
	else:
		return False 

def keyword_in_hashtags(tweet, keyword):
	"""
	Checks if a given keyword is in the given tweet hashtags. 

	:param tweet: Tweet in Dictionary format as provided from ParseJSON
	:param keyword: A keyword or regular expression to search for 
	:returns: True if text contains keyword, False otherwise
	"""
	for hashtag in tweet['entities']['hashtags']:
		if re.search(keyword.lower(), hashtag['text'].lower()):
			return True
	return False	

def keyword_in_urls(tweet, keyword):
	"""
	Checks if a given keyword is in the given tweet urls. 
	
	1. Checks urls and 
	2. Meida urls 

	:param tweet: Tweet in Dictionary format as provided from ParseJSON
	:param keyword: A keyword or regular expression to search for 
	:returns: True if text contains keyword, False otherwise
	"""
	k = keyword.lower()
	for url in tweet['entities']['urls']:
		if re.search(k, url['expanded_url'].lower()):
			return True
	
	if "media" in tweet['entities']:
		for item in tweet['entities']['media']:
			if re.search(k, item['expanded_url'].lower()):
				return True 
		
	return False
	
def tweet_has_keyword(tweet, keyword):
	"""
	Checks if a given keyword is in the given tweet.
	
	It checks if the keyword exists in: 
	1. the text (main and retweet if available), 
	2. the hashtags,
	3. the urls,
	4. the media urls

	:param tweet: Tweet in Dictionary format as provided from ParseJSON
	:param keyword: A keyword or regular expression to search for 
	:returns: True if text contains keyword, False otherwise
	"""
	if(keyword_in_text(tweet, keyword)):
		return True
	elif(keyword_in_hashtags(tweet, keyword)):
		return True
	elif(keyword_in_urls(tweet, keyword)):
		return True
	else:
		return False

def tweet_has_coordinates(tweet):
	"""
	Checks if a tweet contains coordinates. 

	:param tweet: Tweet in Dictionary format as provided from ParseJSON
	:returns: True if text contains coordinates, False otherwise
	"""	
	
	if tweet['coordinates'] is not None: 
		return True
	else: 
		return False

def tweet_info(tweet,keyword):
	"""
	Returns the tweet info  

	:param tweet: Tweet in Dictionary format as provided from ParseJSON
	:param keyword: Keyword the tweet is grouped into 
	:returns: A comma separated string containing tweet date, longitude, latidude and keyword
	"""	
	
	lon = "NA"
	lat = "NA"
	if tweet_has_coordinates(tweet):
		lon = tweet['coordinates']['coordinates'][0]
		lat = tweet['coordinates']['coordinates'][1]
	
	return(tweet['created_at'] + "," + str(lon) + "," + str(lat) + "," + keyword + "\n")
	
if __name__ == '__main__':
	tweet_dict = {"ravens":[],"patriots":[]}
	parser = ParseJSON("/Users/carrillo/workspace/TwitterTest/data/ravensPatriots.json")
	out = open("/Users/carrillo/workspace/TwitterTest/resources/ravensPatriots.csv", 'w')
	out.writelines('date,lon,lat,group\n')
	for tweet in parser:
		error = True
		if 'text' in tweet:
			for keyword in tweet_dict.keys():
				if(tweet_has_keyword(tweet, keyword)):
					out.writelines(tweet_info(tweet,keyword))
					
					#tweet_dict[keyword].append(tweet)
					error = False
		
		if( error ):
			print(tweet)
	out.flush()
	out.close()	
		#if tweet_has_coordinates(tweet):
		
					
# 	for keyword in tweet_dict.keys():
# 		print( "%s dictionary contains %i tweets"  % (keyword, len(tweet_dict[keyword])))
# 
# 	#Write tweets to file 
# 	
# 	#df = pd.DataFrame(columns=('date', 'lon', 'lat', 'keyword'))
# 	row_list = []
# 	for keyword in tweet_dict.keys():
# 		for tweet in tweet_dict[keyword]:
# 			
# 			
# 			lon = "NA"
# 			lat = "NA"
# 			if tweet_has_coordinates(tweet):
# 				lon = tweet['coordinates']['coordinates'][0]
# 				lat = tweet['coordinates']['coordinates'][1]
# 				
# 			row_list.append({ "date":tweet['created_at'], "lon":lon, "lat":lat, "group":keyword })
# 			
# 	df = pd.DataFrame(row_list).ix[:, ['date','lon','lat','group']]
# 	df.to_csv("/Users/carrillo/workspace/TwitterTest/resources/ravensSteelers3Tweets.csv",index=False)
# 	#print(df)
			
# tweets = pd.DataFrame()
# tweets['text'] = map(lambda tweet: tweet['text'], tweets_data)
# tweets['lang'] = map(lambda tweet: tweet['lang'], tweets_data)
# tweets['country'] = map(lambda tweet: tweet['place']['country'] if tweet['place'] != None else None, tweets_data)

# tweets_by_lang = tweets['lang'].value_counts()
# fig, ax = plt.subplots()
# ax.tick_params(axis='x', labelsize=15)
# ax.tick_params(axis='y', labelsize=10)
# ax.set_xlabel('Languages', fontsize=15)
# ax.set_ylabel('Number of tweets' , fontsize=15)
# ax.set_title('Top 5 languages', fontsize=15, fontweight='bold')
# tweets_by_lang[:5].plot(ax=ax, kind='bar', color='red')

# tweets_by_country = tweets['country'].value_counts()
# fig, ax = plt.subplots()
# ax.tick_params(axis='x', labelsize=15)
# ax.tick_params(axis='y', labelsize=10)
# ax.set_xlabel('Countries', fontsize=15)
# ax.set_ylabel('Number of tweets' , fontsize=15)
# ax.set_title('Top 5 countries', fontsize=15, fontweight='bold')
# tweets_by_country[:5].plot(ax=ax, kind='bar', color='blue')

# Use regular expression to test whether a certain keyword is in the text field. 


This set of scripts is thought to facilitate information gathering from social media. 

1. network_search.py

  Explores the messages, locations and connection of a user account of interest. 
  Information is stored in an sqlite database created for the source_node. 
  One potential risk is the exponential growth for each network depth. The user can counteract this in two ways: i) Retrieve      information from only the top x% connections and ii) limit search to a specified network depth. 
  
  At the moment this is only implemented for twitter. 
  
  How to use network_search.py
  1. Make sure to have all dependencies installed, including sqlite
  2. Make a data folder (this will contain the sqlite database file) 
  3. Specify your twitter api login credentials in twitter_auth.py. If you don't have credentials you can easily create them as   described for example here: https://goo.gl/SXIslG
  4. Set up the search under network_search.py 
      a) Specify the user to start your search "source_node_name" (replace 'testuser')
      b) Fine tune your search: 
            max_depth (how many hops edges away from the main node to search, be careful this will blow up quickly),   
            message_count (number of messages to retrieve), 
            fraction_connections (what percentage of edges to keep. Edges are sorted by weight == number of mentions in tweets)
            dump (should the tweets be dumped in json format) 
  5. Analyze: Connect to the database ("data/'source_node_name'.db") and get information on: 
      Table USER: user_name | visited_during_search | network_depth 
      Table CONNECTION: id | source_user_name | target_user_name | weight 
      Table MESSAGE: id | user_name | text
      Table LOCATION: id | user_name | geojson_if_available | location_if_available

  
  

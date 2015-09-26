This set of scripts is thought to facilitate information gathering from social media. 

1. network_search.py

Explores the messages, locations and connection of a user account of interest. 

Information is stored in an sqlite database created for the source_node. 

One potential risk is the exponential growth for each network depth. The user can counteract this in two ways: 
i) Retrieve information from only the top x% connections and ii) limit search to a specified network depth.
At the moment this is only implemented for twitter. 

import pandas as pd
import numpy as np 

from users import Twitter_user
from twitter_auth import Twitter_auth
from json_parser import ParseJSON


if __name__ == '__main__':

	user = Twitter_user("test", Twitter_auth().authenticate())
	user.load(100)
	user.dump("data/test.json")

	nodes = user.get_nodes()
	print(nodes)

	locations = user.get_locations()
	print(locations)

	messages = user.get_messages()
	print(messages)
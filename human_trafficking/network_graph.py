__author__ = "Fernando Carrillo"
__email__ = "fernando at carrillo.at"

import networkx as nx
import operator
from matplotlib import pyplot as plt

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import exists
from db_tables import Base, Connection, create_sqlite_db


class NetworkGraph(object):
	"""docstring for NetworkGraph"""
	def __init__(self, db_session):
		self.db_session = db_session
		self.graph = nx.Graph() 

	def build(self): 
		"""
		Loads connections from database and 
		creates Graph. 
		"""
		for edge in self.db_session.query(Connection).all(): 
			self.graph.add_edge(edge.user_1_name, edge.user_2_name, weight=edge.weight)

	def draw(self): 
		"""
		Draws the graph. 
		"""
		nx.draw_spring(self.graph)
		#nx.draw(self.graph)
		plt.show()

	def page_rank(self): 
		"""
		Returns the page rank of the nodes in the graph
		"""
		return nx.pagerank(self.graph)

	def get_top_nodes(self, n=-1): 
		"""
		Returns a list of the top n nodes in sorted order. 
		If n=-1 all nodes are returned. 
		"""
		pr = self.page_rank()
		sorted_pr = sorted(pr.items(), key=operator.itemgetter(1), reverse=True)
		if n == -1: 
			return sorted_pr
		else: 
			return sorted_pr[:n]
	
if __name__ == '__main__':
	# Set-up connection to message data_base 
	engine = create_engine('sqlite:///data/testuser.db')
	Base.metadata.bind = engine 
	DBSession = sessionmaker(bind=engine)
	session = DBSession()

	ng = NetworkGraph(db_session=session)
	ng.build()
	ng.draw()
	node_rank = ng.get_top_nodes(n=10)
	print(node_rank)
	
		
import json

class ParseJSON:
	"""
	Generate an iterator over tweets read from the JSON data. 
	Implementation based on suggestions from http://districtdatalabs.silvrback.com/simple-csv-data-wrangling-with-python
	"""
	def __init__(self, json_data_path):
		self.path = json_data_path
						

	def __iter__(self):
		"""
		Generate an iterator over tweets read from the JSON data. Implementation based on suggestions from http://districtdatalabs.silvrback.com/simple-csv-data-wrangling-with-python

		Quote: 
		A couple of key points with the code above:
		Always wrap the CSV reader in a function that returns a generator (via the yield statement).
		Open the file in universal newline mode with 'rU' for backwards compatibility.
		Use context managers with [callable] as [name] to ensure that the handle to the file is closed automatically.
		"""
		with open(self.path, 'rU') as data:
			for line in data:
				try:
					yield json.loads(line)
				except:
					continue

if __name__ == '__main__':
	#twitter_data_path = "/Users/carrillo/workspace/SmallProjects/TwitterAnalysis/data/packersFalcons.json"	
	twitter_data_path = "/Users/carrillo/workspace/SmallProjects/human_trafficking/data/fcarrillo81.json"	
	parser = ParseJSON(twitter_data_path)
	
	for tweet in parser:
		print tweet
		
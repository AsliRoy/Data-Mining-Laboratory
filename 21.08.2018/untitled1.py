import pyfpgrowth
transactions = [['A','B'],
               ['B','C','D'],
               ['A','C','D','E'],
               ['A','D','E'],
               ['A','B','C'],
               ['A','B','C','D'],
	       ['B','C'],
	       ['A','B','C'],
	       ['A','B','D'],
	       ['B','C','E']]
patterns = pyfpgrowth.find_frequent_patterns(transactions, 2)
print patterns

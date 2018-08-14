from math import *
from decimal import Decimal
def euclidian_distance(x,y):
	return sqrt(sum(pow(a-b,2) for a,b in zip(x,y)))

def manhattan_distance(x,y):
	return sum(abs(a-b) for a,b in zip(x,y))

def nth_root(value,n_root):
	root_value=1/float(n_root)
	return round(Decimal(value) ** Decimal(root_value),3)

def minkowski_distance(x,y,p_value):
	return nth_root(sum(pow(abs(a-b),p_value) for a,b in zip (x,y)),p_value)

print ("Euclidian distance:")
print euclidian_distance([0,3,4,5],[7,6,3,-1])

print ("Manhattan Distance:")
print manhattan_distance([0,3,4,5],[7,6,3,-1])

print ("Minkowski Distance:")
print minkowski_distance([0,3,4,5],[7,6,3,-1],3)
	

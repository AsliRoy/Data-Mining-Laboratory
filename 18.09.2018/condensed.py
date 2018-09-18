
import csv
import random
import math
import operator

import matplotlib.pyplot as plt


change = {'Iris-setosa': 1, 'Iris-virginica': 2, 'Iris-versicolor': 3}

def loadDataset(filename, split, trainingSet=[] , testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        random.shuffle(dataset)
        for x in range(len(dataset)-1):
            #for y in range(6):
            for y in range(4):
                if(dataset[x][y] in change):
                    dataset[x][y] = float(change[dataset[x][y]])
                else:
                    try:
                        dataset[x][y] = float(dataset[x][y])
                    except ValueError:
                        pass
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])
 
 
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((float(instance1[x]) - float(instance2[x])), 2)
	return math.sqrt(distance)
 
def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors
 
def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]
 
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
	
def main():
    # prepare data
    trainingSet=[]
    testSet=[]
    split = 0.67
    #loadDataset('car.data', split, trainingSet, testSet)
    loadDataset('iris.csv', split, trainingSet, testSet)
    print ('Train set: ' + repr(len(trainingSet)))
    print ('Test set: ' + repr(len(testSet)))
    #creating condensed set
    condSet=[]
    condSet.append(trainingSet[0])
    print(trainingSet[0])
    k=1
    for x in range(len(trainingSet)):
        neigh = getNeighbors(condSet, trainingSet[x], k)
        res = getResponse(neigh)
        #print(res)
        if (res != trainingSet[x][-1]):
            condSet.append(trainingSet[x])
    print("Length of training Set : ",len(trainingSet))
    print("Length of Condensed Set : ",len(condSet))
    print(condSet)
    # generate predictions
    predictions=[]
    for x in range(len(testSet)):
        neighbors = getNeighbors(condSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        #print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '% for k-value: ',k)
	
main()
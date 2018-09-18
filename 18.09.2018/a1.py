import random
import operator
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import euclidean


class kNN(object):
    def testkNN(self,trainX,trainY,testX,k):
        #print(datetime.datetime.now())
        p=[]
        for y in range(len(testX)): 
            dist=[]# iterate for loop for every test point
            dist=euclidean(trainX[0],testX[y]) # find eucliean distance
            temp=zip(dist,trainY) # zip with label
            sortedtemp = sorted(temp,key=lambda tup: tup[0])
            Nearestneighbors=sortedtemp[:k] # find k nearest neighbour
            NearestneighborsClass = map(operator.itemgetter(1), Nearestneighbors)
            c=Counter(NearestneighborsClass).most_common() # find most common class label
            p.append(c[0][0]) # append into prediction array
        return p

    def modelaccuracy(self,testY, predictiontestY): # check accuracy of model
        tp = 0
        for x in range(len(testY)):
            if testY[x][-1] == predictiontestY[x]:
                tp += 1 # if its is matching then increasing count
        total = float(len(testY))
        return (tp/total) * 100.0 # actual / total divide

    def condensedata(self,trainX, trainY):
        index= [] # dummy variable
        SS=[] # declare  forSubset train X
        SSL=[] # declare for subset train Y
        temp=[]
        condensedIdx=[] # declare final array to store condensed indices
        index = range(len(trainX))
        SS.append(trainX[0])
        SSL.append(trainY[0])
        while sum(index): # execute till all element of train X
            nonzero_index = np.nonzero(index) # finding remaining train X
            RI=random.choice(nonzero_index[0]) # Select random index from remaining train X
            index[RI]=0 # Make index zero if it is used once
            temp=[]
            temp.append(trainX[RI])
            predictedtestY = self.testkNN(SS,SSL,temp,1) # execute 1NN for given subset
            if(predictedtestY[0] != trainY[RI]): # if its matching
                SS.append(trainX[RI]) # append if not matching bcz its required
                SSL.append(trainY[RI])
                condensedIdx.append(RI) # store that index into final array
        return condensedIdx

#####################################################################HELER CODE ############
nTrain = 15000
nTest = 5000
k=3
df = pd.read_csv('letter-recognition.data', header=None)
trainX = np.array(df.iloc[0:nTrain,1:])

trainY = np.array(df.iloc[0:nTrain,0])

testX = np.array(df.iloc[0:nTest,1:])
testY = np.array(df.iloc[nTest:,0])

condensedtrainX = []
condensedtrainY = []

knn = kNN()
print 'Train set: ' + repr(len(trainX))
print 'Test set: ' + repr(len(testX))
print 'K Value :' + repr(k)

predictedtestY1 = knn.testkNN(trainX,trainY,testX,k)  # EXECUTE ENTIRE kNN

accuracy=knn.modelaccuracy(testY,predictedtestY1) # find accuracy


print('Accuracy with entire training set')
print(accuracy) # print accuracy
lab=np.unique(trainY)
print(lab)
c=confusion_matrix(testY,predictedtestY1,labels=lab)
print(c)
print('############################################# Condensed ')
print 'Train set: ' + repr(len(trainX))
print 'Test set: ' + repr(len(testX))
print 'K Value :' + repr(k)

condensedIdx=knn.condensedata(trainX,trainY)
for i in condensedIdx:# find condensed training set from new condesed indices
    condensedtrainX.append(trainX[i])
    condensedtrainY.append(trainY[i])

condensedtrainXtemp = np.array(condensedtrainX)


print(len(condensedIdx))
predictedtestY = knn.testkNN(condensedtrainXtemp,condensedtrainY,testX,k) # execute kNN on condensed train Set
accuracy1=knn.modelaccuracy(testY,predictedtestY)

print('Accuracy with Condensed training set')
print(accuracy1)


import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import random
import operator
from collections import Counter
# Importing the dataset
dataset = pd.read_csv('car.csv')

#assign name to the columns
dataset.columns = ['buying','maint','doors','persons','lug_boot','safety','classes']

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

#Categorical values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
X = X.apply(LabelEncoder().fit_transform)
onehotencoder = OneHotEncoder(categorical_features=[2,3])
X = onehotencoder.fit_transform(X).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# creating odd list of K for KNN
myList = list(range(1,50))

# subsetting just the odd ones
neighbors = filter(lambda x: x % 2 != 0, myList)

# empty list that will hold cv scores
cv_scores = []

# perform 3-fold cross validation
from sklearn.cross_validation import cross_val_score
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=3, scoring='accuracy')
    cv_scores.append(scores.mean())
    
# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print "The optimal value of k for the given dataset is %d" % optimal_k

# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel('K value')
plt.ylabel('Error in misclassification')
plt.show()

# Fitting K-NN to the Training set

classifier = KNeighborsClassifier(n_neighbors = optimal_k, metric = 'manhattan', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


from sklearn import tree,metrics
count_misclassified = (y_test != y_pred).sum()
accuracy = metrics.accuracy_score(y_test, y_pred)

print('Misclassified samples: {}'.format(count_misclassified))
print("Confusion Matrix: ",cm)
print('Accuracy: {:.2f}'.format(accuracy))

def testkNN(trainX,trainY,testX,k):
        #print(datetime.datetime.now())
        p=[]
        for y in range(len(testX)): # iterate for loop for every test point
            dist=[]
            dist=np.linalg.norm(trainX-testX[y],axis=1) # find eucliean distance
            temp=zip(dist,trainY) # zip with label
            sortedtemp = sorted(temp,key=lambda tup: tup[0])
            Nearestneighbors=sortedtemp[:k] # find k nearest neighbour
            NearestneighborsClass = map(operator.itemgetter(1), Nearestneighbors)
            c=Counter(NearestneighborsClass).most_common() # find most common class label
            p.append(c[0][0]) # append into prediction array
        return p


index= [] # dummy variable
SS=[] # declare  forSubset train X
SSL=[] # declare for subset train Y
temp=[]
condensedIdx=[] # declare final array to store condensed indices
index = range(len(X_train))
SS.append(X_train[0])
SSL.append(y_train[0])
print SS
print SSL

while sum(index): # execute till all element of train X
    nonzero_index = np.nonzero(index) # finding remaining train X
    RI=random.choice(nonzero_index[0]) # Select random index from remaining train X
    index[RI]=0 # Make index zero if it is used once
    temp=[]
    temp.append(X_train[RI])
    predictedtestY = testkNN(SS,SSL,temp,1)
if(predictedtestY[0] != y_train[RI]): # if its matching
    SS.append(X_train[RI]) # append if not matching bcz its required
    SSL.append(y_train[RI])
    condensedIdx.append(RI) # store that index into final array
print condensedIdx













#Import the necessary packages
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#Read the dataset from the CSV file
df=pd.read_csv("car.csv",sep=",",names = ["Cost", "maintenance", "door", "person","LuggageBoot","Safety","CarClass"])

#print df

Cost = {'vhigh': 1,'high': 2,'med':3,'low':4}
maintenance = {'vhigh': 1,'high': 2,'med':3,'low':4}
door = {'2':1,'3':2,'4':3,'5more':4}
person = {'2':1,'4':2,'more':3}
LuggageBoot = {'small': 1,'med':2,'big':3}
Safety = {'high': 1,'med':2,'low':3}
CarClass = {'unacc': 1,'acc':2,'good':3,'vgood':4}

 
# traversing through dataframe
# Gender column and writing
# values where key matches
#print data.iloc[1:, [1]]
df.Cost = [Cost[item] for item in df.Cost]
df.maintenance = [maintenance[item] for item in df.maintenance]
df.door = [door[item] for item in df.door]
df.person = [person[item] for item in df.person]
df.LuggageBoot = [LuggageBoot[item] for item in df.LuggageBoot]
df.Safety = [Safety[item] for item in df.Safety]
df.CarClass = [CarClass[item] for item in df.CarClass]

#print df

X=df.values[:,0:6]
Y=df.values[:,-1]

print X
print Y

msk = np.random.rand(len(df)) < 0.8
X_train = X[msk]
Y_train = Y[msk]

X_test = X[~msk]
Y_test = Y[~msk]


print X_train.shape
print Y_train.shape

print X_test.shape
print Y_test.shape

clf_gini = DecisionTreeClassifier(criterion = "gini",splitter="best")
clf_gini.fit(X_train, Y_train)

pred=clf_gini.predict(X_test)
print pred, Y_test

print "The accuracy score of the given dataset using gini index is:"
print accuracy_score(Y_test,pred)*100





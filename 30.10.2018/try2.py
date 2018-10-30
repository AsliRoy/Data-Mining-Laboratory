
import random
import operator
import numpy as np 
from copy import deepcopy

class point:
    def __init__(self,listLine):
        self.feature = listLine
        self.label = None           
class label:
    def __init__(self,listLine):
        self.feature = listLine
        self.cluster = []


filenamme = 'iris.csv'
points=[]
data = [l.strip() for l in open('iris.data') if l.strip()]
features = [tuple(map(float, x.split(',')[:-1])) for x in data]
labels = [x.split(',')[-1] for x in data]



def initialization_label(k,points):
    labels=[]

    
    data = [l.strip() for l in open('iris.data') if l.strip()]
    features = [tuple(map(float, x.split(',')[:-1])) for x in data]
    labels = [x.split(',')[-1] for x in data]
    return labels


def show_points(points):
    for index, item in enumerate(points):
        t=""
        for findex, fitem in enumerate(item.feature):
            t += str(findex)+':'+str(fitem)+'\t'
        print(str(index)+':'+"kichi bhi"+str(item.label)+"\tfeature-"+str(t))
        
# show labels feature        
def show_labels(labels):
    for index, item in enumerate(labels):
        t=""
        for findex, fitem in enumerate(item.feature):
            t += str(findex)+':'+str(fitem)+'\t'
        print("label"+str(index)+"\tfeature-"+str(t))

def kmeans(k,labels,points,num_basic_kmeans):
    # sse = sum of square error
    sse=0

    for a in range(len(labels)):
        labels[a].cluster=[]
    # step1: cluster assignment
    for a in range(len(points)):
      
        tp=[]
        for b in range(len(labels)):
            point_features_error=0
            for c in range(dimension):
                point_features_error += (labels[b].feature[c]-points[a].feature[c])**2
            tp.append(point_features_error)            
        points[a].label = tp.index(min(tp))+num_basic_kmeans
        sse+=tp[tp.index(min(tp))]
        
        labels[ tp.index(min(tp)) ].cluster.append(points[a])     
        
    # step2: move centroid
    for a in range(len(labels)):
        if len(labels[a].cluster) !=0:
            for c in range(dimension):
                temp = 0
                for b in range(len( labels[a].cluster )):
                    temp+=labels[a].cluster[b].feature[c]
                labels[a].feature[c]=float(temp)/float(len(labels[a].cluster))  
                
  
    return (sse , labels)

# basic_kmeans
def basic_kmeans(k,points,num_basic_kmeans,times):
    labels = optimization(k,points,times)
    psse, plabelList = kmeans(k,labels,points,num_basic_kmeans)
    sse, labelList = kmeans(k,labels,points,num_basic_kmeans)
    count=1
    while psse != sse:
        psse = sse
        sse, labelList = kmeans(k,labels,points,num_basic_kmeans)
        count += 1
    return (sse , labelList)


def costfunction(labels,points):
    # sse = sum of square error
    sse=0
    for a in range(len(points)):
        
        tp=[]
        for b in range(len(labels)):
            point_features_error=0
            for c in range(dimension):
                point_features_error += (labels[b].feature[c]-points[a].feature[c])**2
            tp.append(point_features_error)            
        sse+=tp[tp.index(min(tp))]
    return sse

# bisecting_Kmeans
def bisecting_Kmeans(points,nbk,times):
    
    sse,labels = basic_kmeans(2,points,0,times)

    show_points(labels[0].cluster)
    print('@@')
    show_points(labels[1].cluster)
    print('@@')
    minsseDict = {}
    clusterDict = {}

    for nbk in range(nbk-2):
        sse0 = costfunction(labels,labels[0].cluster)
        minsseDict[nbk]=sse0
        clusterDict[nbk]=labels[0].cluster
       
        sse1 = costfunction(labels,labels[1].cluster)
        minsseDict[nbk+2]=sse1
        clusterDict[nbk+2]=labels[1].cluster
        # find minsseDict min value
        key = max(minsseDict.iteritems(), key=operator.itemgetter(1))[0]
        tppoints = clusterDict[key]
        del minsseDict[key]
        del clusterDict[key]
        sse,labels = basic_kmeans(2,tppoints,nbk*2+2,times)
        
        print('@@')
        show_points(tppoints)


def optimization(k,points,times):
    if times != 0:
        sseList=[]
        labelList=[]
        for a in range(times):
            labels = initialization_label(k,points)
            sse = costfunction(labels,points)
            sseList.append(sse)
            labelList.append(labels)
            best_labels = labelList[sseList.index(min(sseList))]
    else:
        best_labels = initialization_label(k,points)
    return best_labels

def choose_k(points,k_range,times):
    k_candidate=[]

    for k in range(1,k_range):
        best_labels = optimization(k,points,times)
        sse,labelList = basic_kmeans(k,points,0,times)
        k_candidate.append(sse)
    sse_slopeList=[0,0]
    print(k_candidate)
    for a in range(len(k_candidate)-1):
        
        sse_slope = k_candidate[a+1]-k_candidate[a]
        sse_slopeList.append(sse_slope)
       
    best_k = sse_slopeList.index(min(sse_slopeList))
    return best_k

# basic_kmeans
basic_kmeans(4,points,0,10)
show_points(points)

# depend on elbow theorem find bestk
choose_k(points,10,10)

# bisecting_Kmeans
bisecting_Kmeans(points,4,10)
show_points(points)
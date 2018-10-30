from __future__ import division
from collections import defaultdict
import random
from matplotlib import pyplot as plt
import numpy as np


def load_data():
    data = [l.strip() for l in open('iris.data') if l.strip()]
    features = [tuple(map(float, x.split(',')[:-1])) for x in data]
    labels = [x.split(',')[-1] for x in data]
    return dict(zip(features, labels))

def dist2(f1, f2):
    a = np.array
    d = a(f1)-a(f2)
    return np.sqrt(np.dot(d, d))

def mean(feats):
    return tuple(np.mean(feats, axis=0))

def assign(centers):
    new_centers = defaultdict(list)
    for cx in centers:
        for x in centers[cx]:
            best = min(centers, key=lambda c: dist2(x,c))
            new_centers[best] += [x]
    return new_centers

def update(centers):
    new_centers = {}
    for c in centers:
        new_centers[mean(centers[c])] = centers[c]
    return new_centers

def kmeans(features, k, maxiter=100):
    centers = dict((c,[c]) for c in features[:k])
    centers[features[k-1]] += features[k:]
    for i in xrange(maxiter):
        new_centers = assign(centers)
        new_centers = update(new_centers)
        if centers == new_centers:
            break
        else:
            centers = new_centers
    return centers

def counter(alist):
    count = defaultdict(int)
    for x in alist:
        count[x] += 1
    return dict(count)

def demo(seed=123):
   

    data = load_data()
    
    features = data.keys()
    random.seed(seed)
    random.shuffle(features)
    clusters = kmeans(features, 4)

    for c in clusters:
        for x in clusters[c]:
            print data[x]
            print x[0]
            if data[x]=="Iris-setosa":
                plt.scatter(x[0],x[1],c="blue",s=20)
            if data[x]=="Iris-versicolor":
                plt.scatter(x[0],x[1],c="orange",s=20)
            if data[x]=="Iris-virginica":
                plt.scatter(x[0],x[1],c="purple",s=20)
            plt.scatter(c[0],c[1],c='black',marker="*",s=100)
                
if __name__ == "__main__":
    demo()

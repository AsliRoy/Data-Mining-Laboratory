import math
import csv
from matplotlib import pyplot as plt
class flower:
   def __init__(self, sepal_l, sepal_w, petal_l, petal_w, type, group="not-visited"):
       self.sepal_l = sepal_l
       self.sepal_w = sepal_w
       self.petal_l = petal_l
       self.petal_w = petal_w
       self.type = type
       self.group = group

def dbscan(dataset, eps, MinPts):
   c = 0

   for item in dataset:
       if item.group != "not-visited":
           continue

       item.group = "visited"
       NeighborPts = regionQuery(item, eps, dataset)

       if(len(NeighborPts) < MinPts):
           item.group = "outlier"
       else:
           c = c + 1
           expandCluster(item, NeighborPts, c, eps, MinPts, dataset)

   return c

def expandCluster(item, NeighborPts, c, eps, MinPts, dataset):
   item.group = c
   for itens in NeighborPts:
       if itens.group == "not-visited":
           itens.group = "visited"
           NeighborPts_ = regionQuery(itens, eps, dataset)
           if(len(NeighborPts_) >= MinPts):
               NeighborPts.union(NeighborPts_)
           if itens.group != "not-visited":
               itens.group = c

def regionQuery(item, eps, dataset):
   NeighborPts = set()
   for itens in dataset:
       if(euclidian(item, itens) <= eps):
           NeighborPts.add(itens)
   return NeighborPts

def euclidian(subject1, subject2):
   soma = pow(subject1.sepal_l - subject2.sepal_l, 2) + \
          pow(subject1.sepal_w - subject2.sepal_w, 2) + \
          pow(subject1.petal_l - subject2.petal_l, 2) + \
          pow(subject1.petal_w - subject2.petal_w, 2);
   result = math.sqrt(soma)
   return result

def main():
   dataset = list()
   radius = 3
   density = 4

   with open('iris.csv', 'r') as csvfile:
       lines = csv.reader(csvfile)
       for row in lines:
           dataset.append(flower(float(row[0]), float(row[1]), float(row[2]), float(row[3]), row[4]))

   clusters = dbscan(dataset, radius, density)

   print("Radius: {}".format(radius))
   print("Density: {}".format(density))
   print("Clusters: {}".format(clusters))

if __name__ == '__main__':
    main()
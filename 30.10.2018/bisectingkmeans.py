import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from kmeans import BisectingKMeansClusterer, KMeansClusterer


def main():
    
    iris = datasets.load_iris()
    X = iris.data[:, 2:4]
    data = X

    
    c = BisectingKMeansClusterer(
        data, max_k=3, min_gain=0.1)

   
    plt.figure(1)
    
    for i in range(c.k):
        plt.plot(c.C[i][:, 0], c.C[i][:, 1], 'D')
    
    plt.plot(c.u[:, 0], c.u[:, 1], 'ko')
    plt.show()


if __name__ == '__main__':
    main()
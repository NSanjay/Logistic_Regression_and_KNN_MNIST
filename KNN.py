__author__ = "Sanjay Narayana"

import load_dataset as dataset
import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
import time

np.random.seed(0)

class KNearestNeighbors(object):
    def __init__(self):
        """
        Class implements KNN search using Euclidean Distance as metric.
        Computes distance of each test sample against all of the train samples through Batch Processing
        """
        self.number_of_classes = 10
        self.nearest_neigbors = np.empty((0, 100), dtype=np.int8)
        self.batch_size = 2000

    def load_data(self):
        self.train_labels, train_images = dataset.read()
        self.train_data_length = len(self.train_labels)
        self.test_labels, test_images = dataset.read(dataset='testing')
        self.test_data_length = len(self.test_labels)

        """normalize data set"""
        train_images = train_images / 255
        test_images = test_images / 255
        self.train_images_flattened = train_images.reshape(train_images.shape[0], -1)
        self.test_images_flattened = test_images.reshape(test_images.shape[0], -1)

        #train_std = np.nanstd(self.train_images_flattened, axis=0)
        train_mean = np.nanmean(self.train_images_flattened)
        self.train_images_flattened = self.train_images_flattened - train_mean
        self.test_images_flattened = self.test_images_flattened - train_mean

        self.train_images_squared = np.einsum('ij,ij->i',self.train_images_flattened,self.train_images_flattened)

    def find_batch_nearest_neighbors(self, test_batch):
        """Sanjay - Batch processing of KNN"""

        """(a-b)**2 = a**2 + b**2 - 2ab"""
        dot_product = np.dot(test_batch,self.train_images_flattened.T)

        """Sanjay - somehow einsum is very slow here
        dot_product = np.einsum('ij,kj->ik',test_batch, self.train_images_flattened)"""

        test_squared = np.einsum('ij,ij->i',test_batch,test_batch)
        distances = -2 * dot_product + np.mat(test_squared).T + self.train_images_squared
        k_nearest_neigbors = np.argsort(distances, axis=1)
        labels = (self.train_labels[k_nearest_neigbors])[:,:100]
        self.nearest_neigbors = np.append(self.nearest_neigbors, labels,axis=0)

if __name__ == '__main__':
    knn = KNearestNeighbors()
    knn.load_data()
    k_sizes = [1, 3, 5, 10, 30, 50, 70, 80, 90, 100]
    accuracies = []

    for i in range(len(knn.test_labels) // knn.batch_size):
        print("Batch " + str(i + 1), "of",str(int(len(knn.test_labels) / knn.batch_size)))
        start_time = time.time()
        knn.find_batch_nearest_neighbors(knn.test_images_flattened[i * knn.batch_size:(i + 1) * knn.batch_size])
        end_time = time.time()
        print("Batch completion took", str(end_time - start_time), "seconds.")

    print("final shape:::",knn.nearest_neigbors.shape)
    print("Completed predicting the test data.")
    for k in k_sizes:
        neighba = knn.nearest_neigbors[:, :k]
        predictions = np.squeeze(mode(neighba, axis=1)[0])
        accuracy = np.sum(predictions == knn.test_labels) / len(knn.test_labels)
        print("accuracy::",k," ", accuracy)
        accuracies.append(accuracy)

    plt.plot(k_sizes, accuracies)
    plt.title("Plot of k vs Test Accuracy")
    plt.xlabel('k-Neighbors')
    plt.ylabel('Test Accuracy')
    plt.show()
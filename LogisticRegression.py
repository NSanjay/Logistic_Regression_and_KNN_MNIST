__author__ = "Sanjay Narayana"

import load_dataset as dataset
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

np.random.seed(1)

class LogisticRegression(object):
    def __init__(self):
        """This class implements the Logistic Regression
        Algorithm for the MNIST dataset using only numpy and scipy"""
        self.number_of_classes = 10
        self.learning_rate = 0.000008

    def softmax(self,z):
        e_and_z = np.exp(z - np.amax(z,axis=0,keepdims=True))
        sum = np.sum(e_and_z,axis=0,keepdims=True)
        denom = 1 + sum
        probabilities = np.divide(e_and_z, denom)
        return probabilities

    def calculate_gradient(self):
        gradient = (1/self.train_data_length) * np.dot(self.train_images_flattened, (self.one_hot_encoded_labels - self.probabilities.T))
        return gradient

    def create_one_hot_representation(self, array):
        shape = len(array)
        sparse_one_hot = sparse.csr_matrix((np.ones(shape),(np.arange(shape),array)), shape=(shape,self.number_of_classes))
        return sparse_one_hot.todense()

    def load_data(self):
        # load train data
        self.train_labels, train_images = dataset.read()
        self.train_data_length = len(self.train_labels)
        self.number_of_dimensions = train_images[0].flatten().shape[0]

        #load test data
        self.test_labels, test_images = dataset.read(dataset='testing')
        self.test_data_length = len(self.test_labels)


        """Much Better accuracy without normalization"""
        #train_images = train_images / 255
        #test_images = test_images / 255

        self.train_images_flattened = train_images.reshape(train_images.shape[0], -1).T
        self.test_images_flattened = test_images.reshape(test_images.shape[0], -1).T

        train_mean = np.nanmean(self.train_images_flattened)
        self.train_images_flattened = self.train_images_flattened - train_mean
        self.test_images_flattened = self.test_images_flattened - train_mean

        self.weight_vector = np.random.random((self.number_of_dimensions,self.number_of_classes)) * 0.000008
        self.one_hot_encoded_labels = self.create_one_hot_representation(self.train_labels)

        self.test_one_hot_encoded_labels = self.create_one_hot_representation(self.test_labels)

    def calculate_accuracy(self,labels):
        accuracy = np.sum(np.argmax(self.probabilities,axis=0) == labels) / len(labels)
        return accuracy

    def model_run(self):
        self.load_data()
        previous_accuracy = train_accuracy = 0
        #iterations = []
        iterations = range(100)

        accuracies = []
        for i in iterations:
            set_of_scores = np.dot(self.weight_vector.T, self.train_images_flattened)
            self.probabilities = self.softmax(set_of_scores)
            gradient = self.calculate_gradient()
            self.weight_vector += (self.learning_rate * gradient)
            #self.weight_vector += ((self.learning_rate * gradient) - self.weight_vector)
            train_accuracy = self.calculate_accuracy(self.train_labels)
            print("train_accuracy of",i,"th iteration::::",train_accuracy)
            set_of_scores = np.dot(self.weight_vector.T, self.test_images_flattened)
            self.probabilities = self.softmax(set_of_scores)
            test_accuracy = self.calculate_accuracy(self.test_labels)
            print("test_accuracy of", i, "th iteration::::", test_accuracy)
            accuracies.append(test_accuracy)
        print("final train accuracy::",train_accuracy)

        print("final test accuracy::", test_accuracy)

        plt.plot(iterations,accuracies)
        plt.title("Plot of Iteration count vs Test Accuracy")
        plt.xlabel("Number of Iterations")
        plt.ylabel("Test Accuracy")
        plt.show()

if __name__ == '__main__':
    logisticRegression = LogisticRegression()
    logisticRegression.model_run()
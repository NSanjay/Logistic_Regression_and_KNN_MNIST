# Logistic_Regression_and_KNN_MNIST

Implementation of the Logistic Regression and KNN algorithms for the MNIST dataset.
The dataset can be downloaded here: http://yann.lecun.com/exdb/mnist/

The distance metric used in KNN is the euclidean distance. KNN achieves an accuracy of 97.3 % for k = 3.
Logistic Regression achieves an accuracy of arounnd 85% and saturates after a 40 iterations.

KNN and implemented by using batching and the expanded form of (a - b)^2 = a^2 + b^2 + 2ab for fast execution. The dataset is centered and normalized before use.

The dataset is only centered and not normalized for Logistic Regression as it resulted in better acccuracy.

Both are vectorized implementations.


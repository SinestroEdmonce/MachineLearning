import numpy as np
from collections import Counter


############################################################################
# DO NOT MODIFY CODES ABOVE 
############################################################################

class KNN:
    def __init__(self, k, distance_function):
        """
        :param k: int
        :param distance_function
        """
        self.k = k
        self.distance_function = distance_function
        self.xs, self.ys = [], []

    # TODO: save features and lable to self
    def train(self, features, labels):
        """
        In this function, features is simply training data which is a 2D list with float values.
        For example, if the data looks like the following: Student 1 with features age 25, grade 3.8 and labeled as 0,
        Student 2 with features age 22, grade 3.0 and labeled as 1, then the feature data would be
        [ [25.0, 3.8], [22.0,3.0] ] and the corresponding label would be [0,1]

        For KNN, the training process is just loading of training data. Thus, all you need to do in this function
        is create some local variable in KNN class to store this data so you can use the data in later process.
        :param features: List[List[float]]
        :param labels: List[int]
        """

        self.xs = features
        self.ys = labels

    # TODO: find KNN of one point
    def get_k_neighbors(self, point):
        """
        This function takes one single data point and finds k-nearest neighbours in the training set.
        You already have your k value, distance function and you just stored all training data in KNN class with the
        train function. This function needs to return a list of labels of all k neighbours.
        :param point: List[float]
        :return:  List[int]
        """

        sorted_zipped = sorted(zip(self.xs, self.ys), key=lambda z: self.distance_function(z[0], point))
        _, sorted_ys = [list(z) for z in zip(*sorted_zipped)]

        return sorted_ys[0:self.k]

    # TODO: predict labels of a list of points
    def predict(self, features):
        """
        This function takes 2D list of test data points, similar to those from train function. Here, you need to process
        every test data point, reuse the get_k_neighbours function to find the nearest k neighbours for each test
        data point, find the majority of labels for these neighbours as the predicted label for that testing data point (you can assume that k is always a odd number).
        Thus, you will get N predicted label for N test data point.
        This function needs to return a list of predicted labels for all test data points.
        :param features: List[List[float]]
        :return: List[int]
        """

        predicted = []
        for i in range(features):
            neighbors = self.get_k_neighbors(features[i])
            y_predicted = self.__vote(neighbors)
            predicted.append(y_predicted)

        return predicted

    def __vote(self, votes):
        """
        This function takes a list of labels of all k neighbours and returns a label for prediction
        :param votes: List[int]
        :return: int
        """

        res = {}
        for vote in votes:
            res[vote] += 1
        # Get the majority of all k labels
        sorted_votes = sorted(res.items(), key=lambda item: item[1], reverse=True)

        return sorted_votes[0][0]


if __name__ == '__main__':
    print(np.__version__)

import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY CODES ABOVE 
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    # F1-score = 2 * (precision * recall) / (precision + recall) = tp / (tp + 1/2 * (fp + fn))
    tp, fp, fn = 0, 0, 0
    for i in range(len(real_labels)):
        # True positive
        if real_labels[i] == 1 and predicted_labels[i] == 1:
            tp += 1
        # False negative
        elif real_labels[i] == 1 and predicted_labels[i] == 0:
            fn += 1
        # False positive
        elif real_labels[i] == 0 and predicted_labels[i] == 1:
            fp += 1

    f1 = float(tp)/(tp+0.5*(fp+fn))
    return f1


class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        assert (len(point1) == len(point2))
        dist = 0.0
        for i in range(len(point1)):
            dist += pow(abs(point1[i] - point2[i]), 3)
        dist = pow(dist, 1.0/3)

        return dist

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        assert (len(point1) == len(point2))
        dist = 0.0
        for i in range(len(point1)):
            dist += pow(point1[i] - point2[i], 2)
        dist = pow(dist, 1.0/2)

        return dist

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        assert (len(point1) == len(point2))
        dist, p1_l2, p2_l2 = 1.0, 0.0, 0.0
        for i in range(len(point1)):
            p1_l2 += pow(point1[i], 2)
            p2_l2 += pow(point2[i], 2)

        # L2 normalization of p1 and p2
        p1_l2, p2_l2 = pow(p1_l2, 1.0/2), pow(p2_l2, 1.0/2)
        if p1_l2 == 0 or p2_l2 == 0:
            return dist

        dot_prod = 0.0
        for i in range(len(point1)):
            dot_prod += point1[i]*point2[i]
        # Cosine distance
        dist = dist-dot_prod/(p1_l2*p2_l2)

        return dist


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you need to try different distance functions you implemented in part 1.1 and different values of k (among 1, 3, 5, ... , 29),
        and find the best model with the highest f1-score on the given validation set.
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] training labels to train your KNN model
        :param x_val:  List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), and model (an instance of KNN) and assign them to self.best_k,
        self.best_distance_function, and self.best_model respectively.
        NOTE: self.best_scaler will be None.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check the distance function:  euclidean > Minkowski > cosine_dist 
		(this will also be the insertion order in "distance_funcs", to make things easier).
        For the same distance function, further break tie by prioritizing a smaller k.
        """
        best_f1 = 0.0
        # Iterate over all the distance functions
        for name, func in distance_funcs.items():
            # Iterate over K
            for k in range(1, 30, 2):
                model = KNN(k=k, distance_function=func)
                model.train(x_train, y_train)
                y_predicted = model.predict(x_val)
                f1 = f1_score(y_val, y_predicted)
                if f1 > best_f1:
                    best_f1 = f1
                    # Assign the best values to these variables
                    self.best_k = k
                    self.best_distance_function = name
                    self.best_model = model

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is the same as "tuning_without_scaling", except that you also need to try two different scalers implemented in Part 1.3. More specifically, before passing the training and validation data to KNN model, apply the scalers in scaling_classes to both of them. 
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param scaling_classes: dictionary of scalers (key is the scaler name, value is the scaler class) you need to try to normalize your data
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), scaler (its name), and model (an instance of KNN), and assign them to self.best_k, self.best_distance_function, best_scaler, and self.best_model respectively.
        
        NOTE: When there is a tie, choose the model based on the following priorities:
        First check scaler, prioritizing "min_max_scale" over "normalize" (which will also be the insertion order of scaling_classes). Then follow the same rule as in "tuning_without_scaling".
        """
        best_f1 = 0.0
        # Iterate over all the scalers
        for sname, scaling_class in scaling_classes.items():
            scaler = scaling_class()
            x_train_scaled = scaler(features=x_train)
            x_val_scaled = scaler(features=x_val)
            # Iterate over all the distance functions
            for fname, func in distance_funcs.items():
                # Iterate over K
                for k in range(1, 30, 2):
                    model = KNN(k=k, distance_function=func)
                    model.train(x_train_scaled, y_train)
                    y_predicted = model.predict(x_val_scaled)
                    f1 = f1_score(y_val, y_predicted)
                    if f1 > best_f1:
                        best_f1 = f1
                        # Assign the best values to these variables
                        self.best_k = k
                        self.best_distance_function = fname
                        self.best_scaler = sname
                        self.best_model = model


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        normalized = []
        for x in features:
            p_l2 = 0.0
            for i in range(len(x)):
                p_l2 += pow(x[i], 2)
            # L2 normalization
            p_l2 = pow(p_l2, 1.0/2)

            if p_l2 == 0:
                normalized.append(x)
            else:
                x_normalized = [f/p_l2 for f in x]
                normalized.append(x_normalized)

        return normalized


class MinMaxScaler:
    def __init__(self):
        pass

    # TODO: min-max normalize data
    def __call__(self, features):
        """
		For each feature, normalize it linearly so that its value is between 0 and 1 across all samples.
        For example, if the input features are [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]].
		This is because: take the first feature for example, which has values 2, -1, and 0 across the three samples.
		The minimum value of this feature is thus min=-1, while the maximum value is max=2.
		So the new feature value for each sample can be computed by: new_value = (old_value - min)/(max-min),
		leading to 1, 0, and 0.333333.
		If max happens to be same as min, set all new values to be zero for this feature.
		(For further reference, see https://en.wikipedia.org/wiki/Feature_scaling.)

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        # Fit data
        data = np.array(features)
        maxf = np.max(data, axis=0)
        minf = np.min(data, axis=0)

        # Min-max normalization
        normalized = []
        for x in features:
            x_normalized = []
            for i in range(len(x)):
                if maxf[i] == minf[i]:
                    x_normalized.append(0.0)
                else:
                    f_normalized = float(x[i]-minf[i]) / (maxf[i]-minf[i])
                    x_normalized.append(f_normalized)

            normalized.append(x_normalized)

        return normalized


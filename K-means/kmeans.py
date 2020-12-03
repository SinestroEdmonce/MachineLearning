import numpy as np


#################################
# DO NOT IMPORT OHTER LIBRARIES
#################################


def squared_euclidean_distances(x, y):
    """
        Compute pairwise (squared) Euclidean distances.
    :param x: (M, D), a numpy array
    :param y: (N, D), a numpy array
    :return: distances: (M, N), a numpy array
    """
    # Have a common dimension D
    assert x.shape[1] == y.shape[1]

    x_square = np.sum(x * x, axis=1, keepdims=True)
    y_square = np.sum(y * y, axis=1, keepdims=True).T
    distances = np.dot(x, y.T)

    # Use inplace operation to accelerate
    distances *= -2.0
    distances += x_square
    distances += y_square

    # result maybe less than 0 due to floating point rounding errors.
    np.maximum(distances, 0, distances)
    return distances


def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    """

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data - numpy array of points
    :param generator: random number generator. Use it in the same way as np.random.
            In grading, to obtain deterministic results, we will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being the *index* of a sample
             chosen as centroid.
    """

    # TODO: implement the Kmeans++ initialization

    centers = []
    # Pick the first center
    centers.append(generator.randint(0, n))
    centroids = np.array(x[centers[0]]).reshape(1, -1)

    # Pick the rest centers
    while len(centers) < n_cluster:
        # Obtain distances from any sample to all centroids
        distances = squared_euclidean_distances(x=x, y=centroids)
        chosen = np.min(distances, axis=1)
        cumulative, r, probs = 0.0, generator.rand(), chosen / np.sum(chosen)

        # Sampling
        for i in range(n):
            cumulative += probs[i]
            if cumulative > r:
                centers.append(i)
                break

        centroids = np.row_stack((centroids, x[centers[-1]]))

    # DO NOT CHANGE CODE BELOW THIS LINE
    return centers


# Vanilla initialization method for KMeans
def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)


class KMeans():
    """
        Class KMeans:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    """

    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator
        self.centers = []

    def fit(self, x, centroid_func=get_lloyd_k_means):

        """
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple in the following order:
                  - final centroids, a n_cluster X D numpy array,
                  - a length (N,) numpy array where cell i is the ith sample's assigned cluster's index (start from 0),
                  - number of times you update the assignment, an Int (at most self.max_iter)
        """
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"

        self.generator.seed(42)
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)

        # Obtain the centroid matrix
        centroids = np.zeros((self.n_cluster, D))
        for i in range(len(self.centers)):
            centroids[i] = x[self.centers[i]]

        # TODO: Update means and membership until convergence
        #   (i.e., average K-mean objective changes less than self.e)
        #   or until you have made self.max_iter updates.
        objective, itr, clustering = 0.0, 0, None
        while itr < self.max_iter:
            # Obtain all clusters based on the assignment
            distances = squared_euclidean_distances(x=x, y=centroids)
            clustering = np.argmin(distances, axis=1)

            # Generate one hot matrix
            assignment = np.eye(N, self.n_cluster)[clustering]

            # Calculate objective and update it
            loss = np.sum((x - np.dot(assignment, centroids)) ** 2)
            if np.abs(objective - loss) < self.e:
                break
            objective = loss

            # Update centroids
            counter = np.sum(assignment, axis=0)
            updated = np.dot(assignment.T, x) / counter[:, None]
            centroids = updated

            # Update iteration
            itr += 1

        # Finish maximum iteration or loss is tolerant
        return centroids, clustering, itr


class KMeansClassifier:
    """
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    """

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        """
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Store following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (numpy array of length n_cluster)
        """

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape

        # TODO:
        # - assign means to centroids (use KMeans class you implemented, 
        #      and "fit" with the given "centroid_func" function)
        # - assign labels to centroid_labels
        kmeans = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e, generator=self.generator)
        centroids, clustering, itr = kmeans.fit(x=x, centroid_func=centroid_func)

        # Voters for every cluster
        polling = []
        for _ in range(self.n_cluster):
            polling.append({})

        # Collect votes for every cluster
        for cluster, label in zip(clustering, y):
            if label not in polling[cluster]:
                polling[cluster][label] = 0

            polling[cluster][label] += 1

        # Vote
        centroid_labels = []
        for poll in polling:
            centroid_labels.append(max(poll, key=poll.get))
        centroid_labels = np.array(centroid_labels)

        # DO NOT CHANGE CODE BELOW THIS LINE
        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        """
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        """

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape

        # TODO:
        # - for each example in x, predict its label using 1-NN on the stored 
        #    dataset (self.centroids, self.centroid_labels)
        distances = squared_euclidean_distances(x=x, y=self.centroids)
        clustering = np.argmin(distances, axis=1)

        y_predict = self.centroid_labels[clustering]

        return np.array(y_predict)


def transform_image(image, code_vectors):
    """
        Quantize image using the code_vectors (aka centroids)

        Return a new image by replacing each RGB value in image with the nearest code vector
          (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    """

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    # TODO
    # - replace each pixel (a 3-dimensional point) by its nearest code vector
    N, M, D = image.shape
    flatten = image.reshape(N*M, D)

    # Assign every pixel to its nearest code vector
    distances = squared_euclidean_distances(x=flatten, y=code_vectors)
    assignment = np.argmin(distances, axis=1)
    compressed = code_vectors[assignment].reshape(N, M, D)

    return compressed


import math
import random
import numpy as np
import pandas
import pandas as pd
from sklearn.model_selection import train_test_split

Name = "Amir hosein Haji mohammad rezaie"
Student_Number = "99109252"


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        """
        Class for storing Decision Tree as a binary-tree
        Inputs:
        - feature: Name of the the feature based on which this node is split
        - threshold: The threshold used for splitting this subtree
        - left: left Child of this node
        - right: Right child of this node
        - value: Predicted value for this node (if it is a leaf node)
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        if self.left is None and self.right is None:
            return True
        else:
            return False


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        """
        Class for implementing Decision Tree
        Attributes:
        - max_depth: int
            The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until
            all leaves contain less than min_samples_split samples.
        - min_num_samples: int
            The minimum number of samples required to split an internal node
        - root: Node
            Root node of the tree; set after calling fit.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def is_splitting_finished(self, depth, num_class_labels, num_samples):
        """
        Criteria for continuing or finishing splitting a node
        Inputs:
        - depth: depth of the tree so far
        - num_class_labels: number of unique class labels in the node
        - num_samples: number of samples in the node
        :return: bool
        """
        if self.max_depth is None:
            if num_class_labels == num_samples:
                return True
            else:
                return False
        else:
            if self.max_depth == depth or num_class_labels == num_samples:
                return True
            else:
                return False

    def split(self, X, y, feature, threshold):
        """
        Splitting X and y based on value of feature with respect to threshold;
        i.e., if x_i[feature] <= threshold, x_i and y_i belong to X_left and y_left.
        Inputs:
        - X: Array of shape (N, D) (number of samples and number of features respectively), samples
        - y: Array of shape (N,), labels
        - feature: Name of the the feature based on which split is done
        - threshold: Threshold of splitting
        :return: X_left, X_right, y_left, y_right
        """
        X_left = X[X[feature] <= threshold]
        X_right = X[X[feature] > threshold]

        del X_left[feature]
        del X_right[feature]

        y_left = y[X[feature] <= threshold]
        y_right = y[X[feature] > threshold]

        return X_left, X_right, y_left, y_right

    def entropy(self, y):
        """
        Computing entropy of input vector
        - y: Array of shape (N,), labels
        :return: entropy of y
        """
        ones = y[y['target'] == 1]
        try:
            p_1 = len(ones)/len(y)
        except ZeroDivisionError:
            p_1 = 0

        entropy = 0
        if (0 < p_1 < 1):
            entropy -= p_1 * math.log(p_1, 2)
            entropy -= (1-p_1) * math.log(1-p_1, 2)
        else:
            entropy = 0

        return entropy

    def information_gain(self, X, y, feature, threshold):
        """
        Returns information gain of splitting data with feature and threshold.
        Hint! use entropy of y, y_left and y_right.
        """
        X_left, X_right, y_left, y_right = self.split(X, y, feature, threshold)

        e1 = self.entropy(y)
        p_left = len(X_left)/(len(X_left) + len(X_right))

        IG = p_left * self.entropy(y_left) + (1-p_left) * self.entropy(y_right)
        IG = e1 - IG
        return IG

    def best_split(self, X, y):
        """
        Used for finding best feature and best threshold for splitting
        Inputs:
        - X: Array of shape (N, D), samples
        - y: Array of shape (N,), labels
        :return:
        """
        features = list(X.columns)  # todo: You'd better use a random permutation of features 0 to D-1
        random.seed(0)
        random.shuffle(features)

        maximum = -math.inf
        selected_feature = None
        selected_threshold = 0
        for feature in features:
            thresholds = X[feature].unique()  # todo: use unique values in this feature as candidates for best threshold

            for threshold in thresholds:
                IG = self.information_gain(X, y, feature, threshold)
                if IG > maximum:
                    maximum = IG
                    selected_feature = feature
                    selected_threshold = threshold

        return selected_feature, selected_threshold

    def build_tree(self, X, y, depth=0):
        """
        Recursive function for building Decision Tree.
        - X: Array of shape (N, D), samples
        - y: Array of shape (N,), labels
        - depth: depth of tree so far
        :return: root node of subtree
        """

        node = None
        num_class_labels = max(len(y) - len(y[y['target'] == 1]), len(y[y['target'] == 1]))
        num_samples = len(X)

        if self.is_splitting_finished(depth, num_class_labels, num_samples):
            node = Node()
            if len(y[y['target'] == 1]) > 0:
                node.value = 1
            else:
                node.value = 0

        else:
            feature, threshold = self.best_split(X, y)
            X_left, X_right, y_left, y_right = self.split(X, y, feature, threshold)
            node = Node(feature, threshold)
            node.left = self.build_tree(X_left, y_left, depth + 1)
            node.right = self.build_tree(X_right, y_right, depth + 1)

        return node

    def fit(self, X, y):
        """
        Builds Decision Tree and sets root node
        - X: Array of shape (N, D), samples
        - y: Array of shape (N,), labels
        """

        self.root = self.build_tree(X, y)

    def predict(self, X):
        """
        Returns predicted labels for samples in X.
        :param X: Array of shape (N, D), samples
        :return: predicted labels
        """
        y_labels = {'target': []}
        for i in range(len(X)):
            node = self.root
            while not node.is_leaf():
                if X.iloc[i][node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            y_labels['target'].append(node.value)

        ans_df = pd.DataFrame(y_labels)
        return ans_df


# import data

df = pandas.read_csv('breast_cancer.csv')

y = pd.DataFrame(df['target'], columns=['target'])
X = df
del X['target']


# Split your data to train and validation sets

train_X, valid_X, train_y, valid_y = train_test_split(X, y, random_state=42, test_size=0.1)

# Tune your hyper-parameters using validation set

depth_candidates = [None] + list(range(5, 9))
min_candidates = list(range(4, 7))
depth = None
min_sample = None
accuracy = math.inf

c = []
for x in valid_y['target']:
    c.append(x)

c = np.array(c)

for a in depth_candidates:
    for b in min_candidates:
        tree = DecisionTree(a, b)
        tree.fit(valid_X, valid_y)
        ans_y = tree.predict(valid_X)
        
        d = []
        for x in ans_y['target']:
            d.append(x)

        d = np.array(d)

        e = c - d
        su = np.sum(np.abs(e))
        if su < accuracy:
            accuracy = su
            depth = a
            min_sample = b

# Train your model with hyper-parameters that works best on validation set

tree = DecisionTree(depth, min_sample)
tree.fit(train_X, train_y)

# Predict test set's labels

X = pd.read_csv('test.csv')
ans = tree.predict(X)
ans.to_csv('output.csv', index=False)

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

class Iris_data():

    def __init__(self):
        iris = datasets.load_iris()
        self.X = iris.data[:, [2, 3]]
        self.y = iris.target
        print('Class label: ', np.unique(self.y))

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=1, stratify=self.y)

        print('Number of labels in y data set:', np.bincount(self.y))
        print('Number of labels in y_train data set:', np.bincount(self.y_train))
        print('Number of labels in y_test data set:', np.bincount(self.y_test))

        sc = StandardScaler()
        sc.fit(self.X_train)
        self.X_train_std = sc.transform(self.X_train)
        self.X_test_std = sc.transform(self.X_test)

        self.X_combined_std = np.vstack((self.X_train_std, self.X_test_std))
        self.y_combined = np.hstack((self.y_train, self.y_test))

        self.X_train_01_subset = self.X_train[(self.y_train == 0) | (self.y_train == 1)]
        self.y_train_01_subset = self.y_train[(self.y_train == 0) | (self.y_train == 1)]

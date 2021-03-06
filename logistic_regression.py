import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def cost_1(z):
    return - np.log(sigmoid(z))

def cost_0(z):
    return - np.log(1 - sigmoid(z))

class LogisticRegressionGD(object):
    # ADAptive LInear NEuron - batch gradient descent

    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ =rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()

            cost = (-y.dot(np.log(output))) - ((1 - y).dot(np.log(1 - output)))
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250 )))

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)

        # same but different
        # return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


if __name__ == '__main__':
    if False:
        z = np.arange(-7, 7, 0.1)
        phi_z = sigmoid(z)
        plt.plot(z, phi_z)
        plt.axvline(0.0, color='k')
        plt.ylim(-0.1, 1.1)
        plt.xlabel('z')
        plt.ylabel('$\phi (z)$')

        plt.yticks([0.0, 0.5, 1.0])
        ax = plt.gca()
        ax.yaxis.grid(True)
        plt.show()

    if False:
        z = np.arange(-10, 10, 0.1)
        phi_z = sigmoid(z)
        c1 = [cost_1(x) for x in z]
        plt.plot(phi_z, c1, label='J(w) if y=1')
        c0 = [cost_0(x) for x in z]
        plt.plot(phi_z, c0, linestyle='--', label='J(w) if y=0')
        plt.ylim(0, 5.1)
        plt.xlim([0, 1])
        plt.xlabel('$\phi (z)$')
        plt.ylabel('J(w)')
        plt.legend(loc='upper center')
        plt.show()

    if True:
        from sklearn import datasets
        from sklearn.model_selection import train_test_split
        from decision_plot import plot_decision_regions
        from sklearn.preprocessing import StandardScaler

        iris = datasets.load_iris()
        X = iris.data[:, [2, 3]]
        y = iris.target
        print('Class label: ', np.unique(y))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

        # print(X.shape)
        # print(y.shape)

        print('Number of labels in y data set:', np.bincount(self.y))
        print('Number of labels in y_train data set:', np.bincount(self.y_train))
        print('Number of labels in y_test data set:', np.bincount(self.y_test)))

        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)

        X_combined_std = np.vstack((X_train_std, X_test_std))
        y_combined = np.hstack((y_train, y_test))

        X_train_01_subset = X_train[(y_train == 0) | (y_train == 1)]
        y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]

        lrgd = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
        lrgd.fit(X_train_01_subset, y_train_01_subset)
        # lrgd.fit(X, y)
        plot_decision_regions(X=X_train_01_subset, y=y_train_01_subset, classifier=lrgd)
        plt.xlabel('Standardized sepal length in cm')
        plt.ylabel('Standardized petal length in cm')
        plt.legend(loc='upper left')
        plt.show()

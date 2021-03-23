from sklearn.linear_model import LogisticRegression
from iris import Iris_data
from decision_plot import plot_decision_regions
import matplotlib.pyplot as plt

if __name__ == '__main__':
    iris = Iris_data()
    lr = LogisticRegression(C=1000, random_state=1)
    lr.fit(iris.X_train_std, iris.y_train)
    plot_decision_regions(iris.X_combined_std, iris.y_combined, classifier=lr, test_idx=range(105,150))
    plt.xlabel('Standardized sepal length in cm')
    plt.ylabel('Standardized petal length in cm')
    plt.legend(loc='upper left')
    # plt.show()

    print(lr.predict_proba(iris.X_test_std[:3, :]))
    print(lr.predict_proba(iris.X_test_std[:5, :]).sum(axis=1))
    print(lr.predict_proba(iris.X_test_std[:5, :]).argmax(axis=1))

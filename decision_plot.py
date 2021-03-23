from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

def versiontuple(v):
    return tuple(map(int, (v.split('.'))))

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    #konfiguracje generator znacznikó i mape kolorów
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    #rysowanie wykresu powierzchni
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    #rysowanie wszystkich próbek
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=colors[idx], marker=markers[idx], label=cl,
                    edgecolor='black')

    #zaznacz próbki testowe
    if test_idx:
        #rysuj wykres wszystkich ptóbek
        # X_test, y_test = X[list(test_idx), :], y[list(test_idx)] #sprawdzić czy tak tezdziala?
        X_test, y_test = X[test_idx, :], y[test_idx]

        # plt.scatter(X_test[:, 0], X_test[:, 1], c='', edgecolor='black', alpha=1.0, linewidth=1, marker='o', s=100, label='Zestaw testowy')
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', edgecolor='black', alpha=1.0, linewidths=1, marker='o', s=100, label='Zestaw testowy')



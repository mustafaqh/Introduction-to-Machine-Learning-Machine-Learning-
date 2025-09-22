import numpy as np
from matplotlib.colors import ListedColormap

def plotDecisionBoundary(X1, X2, y, model, title, ax):
    """
    Plots the decision boundary for a binary classification model along with the training data points.

    Parameters:
        X1 (array-like): Feature values for the first feature.
        X2 (array-like): Feature values for the second feature.
        y (array-like): Target labels.
        model (object): Trained binary classification model with a `predict` method.

    Returns:
        None
    """
    #--- Write your code here ---#
    h = 0.01
    x_min, x_max = X1.min() - 0.1, X1.max() + 0.1
    y_min, y_max = X2.min() - 0.1, X2.max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    XXe = model.predict(np.c_[xx.ravel(), yy.ravel()])
    XXe = XXe.reshape(xx.shape)

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

    ax.pcolormesh(xx, yy, XXe, cmap=cmap_light)
    ax.scatter(X1, X2, c=y, marker='.' ,cmap=cmap_bold)
    ax.set_title(title)
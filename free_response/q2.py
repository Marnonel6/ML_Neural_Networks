import datetime
import numpy as np

from sklearn.metrics import accuracy_score

from src import load_data, plt
from src import Model, FullyConnected, Regularizer
from src import SigmoidActivation, ReluActivation
from src import BinaryCrossEntropyLoss
from src import custom_transform

from free_response.visualize import plot_decision_regions


def visualize_spiral():
    """
    Helper function to help visualize the spiral dataset
    """
    X, y, _ = load_data("data/spiral.csv")
    axis = plt.subplot()
    axis.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()
    plt.close("all")


def visualize_transform():
    """
    Helper function to help visualize your data transformation
    """

    X, y, _ = load_data("data/spiral.csv")
    new_X = custom_transform(X)
    fig = plt.figure()

    # If 2D, plot 2D
    if new_X.shape[1] == 2:
        axis = plt.axes()
        axis.scatter(new_X[:, 0], new_X[:, 1], c=y)
    else:
        # Else plot 3D
        assert new_X.shape[1] == 3
        axis = fig.add_subplot(projection='3d')
        y1 = y == 1
        axis.scatter(new_X[y1, 0], new_X[y1, 1], new_X[y1, 2], c='r', depthshade=False)
        y0 = y == 0
        axis.scatter(new_X[y0, 0], new_X[y0, 1], new_X[y0, 2], marker='*', c='b', depthshade=False)

    plt.show()
    plt.close("all")


def main():
    """
    Load the spiral dataset and classify it with a MLP
    """

    # Try changing the layer structure and hyperparameters
    n_iters = 10000
    learning_rate = 1
    reg = Regularizer(alpha=0.0, penalty="l2")
    layers = [

        # Default
        FullyConnected(2, 32, regularizer=reg),
        SigmoidActivation(),
        FullyConnected(32, 1, regularizer=reg),
        SigmoidActivation()

        # # Changed the activation function
        # FullyConnected(2, 32, regularizer=reg),
        # ReluActivation(),
        # FullyConnected(32, 1, regularizer=reg),
        # ReluActivation()

        # # Change 1
        # FullyConnected(2, 32, regularizer=reg),
        # SigmoidActivation(),
        # FullyConnected(32, 64, regularizer=reg),
        # SigmoidActivation(),
        # FullyConnected(64, 128, regularizer=reg),
        # SigmoidActivation(),
        # FullyConnected(128, 32, regularizer=reg),
        # SigmoidActivation(),
        # FullyConnected(32, 1, regularizer=reg),
        # SigmoidActivation()
    ]

    # Don't change the following code
    X, y, _ = load_data("data/spiral.csv")
    model = Model(layers, BinaryCrossEntropyLoss(), learning_rate)
    model.fit(X, y, n_iters)
    acc = 100 * accuracy_score(model.predict(X) > 0.5, y)
    plot_decision_regions(X, y, model, title=f"{acc:.1f}% accuracy")
    fn = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.png")
    plt.savefig(f"free_response/q2_{fn}")
    plt.show()


if __name__ == "__main__":
    # visualize_spiral()
    # visualize_transform()
    main()

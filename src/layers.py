import numpy as np
import warnings


class SigmoidActivation:
    """
    The sigmoid activation function

    You should not need to edit this class
    """
    def sigmoid(self, x):
        """
        Helper function to compute sigmoid and avoid warnings
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return 1 / (1 + np.exp(-x))

    def forward(self, X):
        """
        First, save the input to self.input_ for gradient updates
        Then, return sigmoid(X)
        """
        self.input_ = X
        return self.sigmoid(X)

    def backward(self, grad, lr=None):
        """
        Using the saved inputs from the last `forward` call,
            compute the gradient of the sigmoid with respect to those inputs
            Multiply by `grad`, the gradient computed previously
        """
        new_grad = self.sigmoid(self.input_) * (1 - self.sigmoid(self.input_))
        return new_grad * grad


class ReluActivation:
    def forward(self, x):
        """
        First, save the input to self.input_ for gradient updates
        Then, return max(0, x) as a numpy calculation
        """

        # self.ones
        #  check previous

        self.input = x

        return np.maximum(0, x)

        # raise NotImplementedError

    def backward(self, grad, lr=None):
        """
        Using the saved inputs from the last `forward` call,
            compute the gradient of ReLU with respect to those inputs
            Multiply by `grad`, the gradient computed previously

        Notes:
          - The derivative of the ReLU is either 0
            if the previous input was <= 0, and otherwise 1.
          - The ReLU doesn't have any parameters to update,
            so you don't need to use `lr`
          - But do remember to include `grad` in your calculation!
        """
        # print(f"\n self.input = {self.input}")
        # print(f"\n grad = {grad}")

        Relu = np.zeros([self.input.shape[0],self.input.shape[1]])

        for i in range(self.input.shape[0]):
            for j in range(self.input.shape[1]):

                if self.input[i,j] <= 0:
                    Relu[i,j] = 0
                else:
                    Relu[i,j] = self.input[i,j]


        # print(f"\n Relu = {Relu}")
        # print(f"\n Relu*grad = {Relu*grad}")

        return Relu*grad


        # raise NotImplementedError


class FullyConnected:
    """
    A fully-connected layer 
    """
    def __init__(self, input_dim, output_dim, regularizer=None):
        """
        input_dim: the input dimension of the layer,
            *not* including the intercept that will be added
            If this is the first layer and the input is 2-dimensional,
            input_dim should be 2.
        output_dim: the output dimension of the layer
        regularizer: if not None, must implement `.grad(weights)`
            to be called in `self.backward()` to add regularization
            to this layer

        Note that self.weights will have shape [1 + input_dim, output_dim]
            because an intercept (or bias) term is added here and then
            an intercept column is to X whenever `self.forward(X)` is called
        """
        # A weight initialization strategy called "Xavier initialization"
        np.random.seed(1)
        self.weights = np.random.normal(
            0, np.sqrt(1 / input_dim), [1 + input_dim, output_dim])
        self.regularizer = regularizer

    def forward(self, X):
        """
        First, save the input to self.input_ for gradient updates
        Then compute W @ X
        """
        X = np.concatenate([np.ones([X.shape[0], 1]), X], axis=1)
        self.input_ = X
        return X.dot(self.weights)

    def backward(self, grad, lr=0.1):
        """
        Using saved inputs from the previous `forward` call, compute
            two gradients: d(W @ X)/dW and d(W @ X)/dX.
            The first is used to update self.weights, and the second
            is returned to be used by later `backward` calls in the network.

        Note: you will need to add code to handle how regularization
            affects the gradient update. Otherwise, you should not
            need to edit this function.
        """

        # Make sure the gradient that's been computed so far
        #   matches up to the shape of this layer
        input_dim, output_dim = self.weights.shape
        batch_size, output_dim2 = grad.shape
        assert output_dim == output_dim2, "Shape mismatch"

        update = np.zeros_like(self.weights)
        new_grad = np.zeros([batch_size, input_dim - 1])

        # Compute d(W @ X)/dW for each node in this layer
        for i in range(output_dim):
            update[:, i] = np.mean(grad[:, (i, )] * self.input_,
                                   axis=0, keepdims=True)

        # Compute d(W @ X)/dX for the next `backward` call
        #   and multiply it by `grad`
        new_grad = grad.dot(self.weights[1:, ].T)

        # Now, we can update our weights
        self.weights -= lr * update

        # If using regularization, perform an additional update to self.weights
        if self.regularizer is not None:
            grad = self.regularizer.grad(self.weights)
            self.weights -= grad*lr
            # raise NotImplementedError

        return new_grad

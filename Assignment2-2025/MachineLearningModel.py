from abc import ABC, abstractmethod
import numpy as np

class MachineLearningModel(ABC):
    """
    Abstract base class for machine learning models.
    """

    @abstractmethod
    def fit(self, X, y):
        """
        Train the model using the given training data.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        pass

    @abstractmethod
    def evaluate(self, X, y):
        """
        Evaluate the model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        score (float): Evaluation score.
        """
        pass

def _polynomial_features(self, X):
    """
        Generate polynomial features from the input features.
        Check the slides for hints on how to implement this one. 
        This method is used by the regression models and must work
        for any degree polynomial
        Parameters:
        X (array-like): Features of the data.

        Returns:
        X_poly (array-like): Polynomial features.
    """
    X_poly = np.ones((X.shape[0], 1))
    for d in range(1, self.degree + 1):
        for i in range(X.shape[1]):
            X_poly = np.c_[X_poly, X[:, i:i+1] ** d]
    return X_poly
    

class RegressionModelNormalEquation(MachineLearningModel):
    """
    Class for regression models using the Normal Equation for polynomial regression.
    """

    def __init__(self, degree):
        """
        Initialize the model with the specified polynomial degree.

        Parameters:
        degree (int): Degree of the polynomial features.
        """
        #--- Write your code here ---#
        self.degree = degree
        self.theta = np.zeros((degree + 1,))

    def fit(self, X, y):
        """
        Train the model using the given training data.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        #--- Write your code here ---#
        X_poly = _polynomial_features(self, X)
        self.theta = np.zeros(X_poly.shape[1])
        self.theta = np.linalg.inv(X_poly.T.dot(X_poly)).dot(X_poly.T).dot(y)
        self.cost = np.mean(np.square(X_poly.dot(self.theta) - y))

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        #--- Write your code here ---#
        X_poly = _polynomial_features(self, X)
        return X_poly.dot(self.theta)

    def evaluate(self, X, y):
        """
        Evaluate the model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        score (float): Evaluation score (MSE).
        """
        #--- Write your code here ---#
        predictions = self.predict(X)
        mse = np.mean((predictions - y) ** 2)
        return mse
    
class RegressionModelGradientDescent(MachineLearningModel):
    """
    Class for regression models using gradient descent optimization.
    """

    def __init__(self, degree, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the model with the specified parameters.

        Parameters:
        degree (int): Degree of the polynomial features.
        learning_rate (float): Learning rate for gradient descent.
        num_iterations (int): Number of iterations for gradient descent.
        """
        #--- Write your code here ---#
        self.degree = degree
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.theta = np.zeros((degree + 1,))
        self.cost_history = []

    def fit(self, X, y):
        """
        Train the model using the given training data.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """

        #--- Write your code here ---#
        X_poly = _polynomial_features(self, X)
        self.theta = np.zeros(X_poly.shape[1])

        for _ in range(self.num_iterations):
            self.theta -= self.learning_rate * (X_poly.T.dot(X_poly.dot(self.theta) - y) / len(y))
            self.cost_history.append(np.mean(np.square(X_poly.dot(self.theta) - y)))

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        #--- Write your code here ---#
        X_poly = _polynomial_features(self, X)
        return X_poly.dot(self.theta)
    
    def evaluate(self, X, y):
        """
        Evaluate the model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        score (float): Evaluation score (MSE).
        """
        #--- Write your code here ---#
        predictions = self.predict(X)
        mse = np.mean((predictions - y) ** 2)
        return mse

class LogisticRegression:
    """
    Logistic Regression model using gradient descent optimization.
    """

    def __init__(self, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the logistic regression model.

        Parameters:
        learning_rate (float): The learning rate for gradient descent.
        num_iterations (int): The number of iterations for gradient descent.
        """
        #--- Write your code here ---#
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.theta = np.zeros((1,))
        self.cost_history = []

    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        #--- Write your code here ---#
        # Bias term to features
        X_poly = np.c_[np.ones(X.shape[0]), X]
        
        # Initialize theta to zeros.
        self.theta = np.zeros(X_poly.shape[1])

        # Gradient descent optimization.
        for _ in range(self.num_iterations):
            predictions = self._sigmoid(X_poly.dot(self.theta))
            errors = predictions - y
            gradient = X_poly.T.dot(errors) / len(y)
            self.theta -= self.learning_rate * gradient
            self.cost_history.append(self._cost_function(predictions, y))

    def predict(self, X):
        """
        Make predictions using the trained logistic regression model.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted probabilities.
        """
        #--- Write your code here ---#
        X_poly = np.c_[np.ones(X.shape[0]), X]
        return self._sigmoid(np.dot(X_poly, self.theta))
    
    def evaluate(self, X, y):
        """
        Evaluate the logistic regression model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        score (float): Evaluation score (e.g., accuracy).
        """
        #--- Write your code here ---#
        return np.mean((self.predict(X) >= 0.5) == y)
    
    def _sigmoid(self, z):
        """
        Sigmoid function.

        Parameters:
        z (array-like): Input to the sigmoid function.

        Returns:
        result (array-like): Output of the sigmoid function.
        """
        #--- Write your code here ---#
        return 1 / (1 + np.exp(-z))
    
    def _cost_function(self, X, y):
        """
        Compute the logistic regression cost function.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        cost (float): The logistic regression cost.
        """
        #--- Write your code here ---#
        epsilon = 1e-15
        return -(1/len(y)) * (np.dot(y, np.log(X + epsilon)) + np.dot((1 - y), np.log(1 - X + epsilon))) #while X is the prediction
    
class NonLinearLogisticRegression:
    """
    Nonlinear Logistic Regression model using gradient descent optimization.
    It works for 2 features (when creating the variable interactions)
    """

    def __init__(self, degree=2, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the nonlinear logistic regression model.

        Parameters:
        degree (int): Degree of polynomial features.
        learning_rate (float): The learning rate for gradient descent.
        num_iterations (int): The number of iterations for gradient descent.
        """
        #--- Write your code here ---#
        self.degree = degree
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.theta = np.zeros((1,))
        self.cost_history = []

    def fit(self, X, y):
        """
        Train the nonlinear logistic regression model using gradient descent.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        #--- Write your code here ---#
        X_poly = self.mapFeature(X[:, 0], X[:, 1], self.degree)
        self.theta = np.zeros(X_poly.shape[1])
        for _ in range(self.num_iterations):
            self.theta -= self.learning_rate * (np.dot(X_poly.T, (self._sigmoid(np.dot(X_poly, self.theta))) - y) / len(y))
            self.cost_history.append(self._cost_function((self._sigmoid(np.dot(X_poly, self.theta))), y))

    def predict(self, X):
        """
        Make predictions using the trained nonlinear logistic regression model.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted probabilities.
        """
        #--- Write your code here ---#
        X_poly = self.mapFeature(X[:, 0], X[:, 1], self.degree)
        return self._sigmoid(np.dot(X_poly, self.theta))
    
    def evaluate(self, X, y):
        """
        Evaluate the nonlinear logistic regression model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        cost (float): The logistic regression cost.
        """
        #--- Write your code here ---#
        prediction = self.predict(X)
        return self._cost_function(prediction, y)

    def _sigmoid(self, z):
        """
        Sigmoid function.

        Parameters:
        z (array-like): Input to the sigmoid function.

        Returns:
        result (array-like): Output of the sigmoid function.
        """
        #--- Write your code here ---#
        return 1 / (1 + np.exp(-z))
    
    def mapFeature(self, X1, X2, D):
        """
        Map the features to a higher-dimensional space using polynomial features.
        Check the slides to have hints on how to implement this function.
        Parameters:
        X1 (array-like): Feature 1.
        X2 (array-like): Feature 2.
        D (int): Degree of polynomial features.

        Returns:
        X_poly (array-like): Polynomial features.
        """
        #--- Write your code here ---#
        one = np.ones([len(X1), 1])
        Xe = np.c_[one,X1,X2]
        for i in range(2,D+1):
            for j in range(0,i+1):
                Xnew = X1**(i - j)*X2**j
                Xnew = Xnew.reshape(-1,1)
                Xe = np.append(Xe, Xnew, 1)
        return Xe
    
    def _cost_function(self, X, y):
        """
        Compute the logistic regression cost function.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        cost (float): The logistic regression cost.
        """
        #--- Write your code here ---#
        epsilon = 1e-15
        return -(1/len(y)) * (np.dot(y, np.log(X+epsilon)) + np.dot((1 - y), np.log(1 - X+epsilon)))
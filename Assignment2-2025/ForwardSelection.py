import numpy as np
from ROCAnalysis import ROCAnalysis

class ForwardSelection:
    """
    A class for performing forward feature selection based on maximizing the F-score of a given model.

    Attributes:
        X (array-like): Feature matrix.
        y (array-like): Target labels.
        model (object): Machine learning model with `fit` and `predict` methods.
        selected_features (list): List of selected feature indices.
        best_cost (float): Best F-score achieved during feature selection.
    """

    def __init__(self, X, y, model):
        """
        Initializes the ForwardSelection object.

        Parameters:
            X (array-like): Feature matrix.
            y (array-like): Target labels.
            model (object): Machine learning model with `fit` and `predict` methods.
        """
        #--- Write your code here ---#
        self.X = X
        self.y = y
        self.model = model
        self.selected_features = []
        self.best_f_score = -np.inf

    def create_split(self, X, y, test_size=0.2):
        """
        Creates a train-test split of the data.

        Parameters:
            X (array-like): Feature matrix.
            y (array-like): Target labels.

        Returns:
            X_train (array-like): Features for training.
            X_test (array-like): Features for testing.
            y_train (array-like): Target labels for training.
            y_test (array-like): Target labels for testing.
        """
    def create_split(self, X, y, test_size=0.2):
        rng = np.random.default_rng(42)
        indices = rng.permutation(len(X))
        split_index = int((1 - test_size) * len(X))
        X_train, X_test = X[indices[:split_index]], X[indices[split_index:]]
        y_train, y_test = y[indices[:split_index]], y[indices[split_index:]]
        return X_train, X_test, y_train, y_test

    
    def train_model_with_features(self, features):
        """
        Trains the model using selected features and evaluates it using ROCAnalysis.

        Parameters:
            features (list): List of feature indices.

        Returns:
            float: F-score obtained by evaluating the model.
        """
        #--- Write your code here ---#
         # If no features are provided, return None.
        if not features:
            return None
        
        # Split the data into training and testing using provided features
        X_train, X_test, y_train, y_test = self.create_split(self.X[:, features], self.y)
        
        # Train the model on the training data.
        self.model.fit(X_train, y_train)
        
        # Predict the target values for the test data.
        y_pred = np.round(self.model.predict(X_test))
        
        # Evaluate the predictions using ROCAnalysis and return the F-score.
        return ROCAnalysis(y_pred, y_test).f_score()
    
    def forward_selection(self):
        """
        Performs forward feature selection based on maximizing the F-score.
        """
        #--- Write your code here ---#
        # Initialize list of available features with all feature indices.
        available_features = list(range(self.X.shape[1]))
                
        # Continue selection process until there are no more features.
        while available_features:
            feature_scores = []
                    
            # Evaluate each candidate feature.
            for f in available_features:
                # Combine currently selected features with the new feature
                candidate_features = self.selected_features + [f]
                        
                # Train the model with the current set of features.
                f_score = self.train_model_with_features(candidate_features)
                        
                # If an F-score is returned, add it to the list.
                if f_score is not None:
                    feature_scores.append((f, f_score))
                    
            # If no F-scores were added, exit.
            if not feature_scores:
                break
                    
            # Find the feature with the best (highest) F-score.
            best_feature, best_f_score = max(feature_scores, key=lambda x: x[1])      
            # Update the selected features, If the best F-score is better than the current best.
            if best_f_score > self.best_f_score:
                self.selected_features.append(best_feature)
                self.best_f_score = best_f_score
                available_features.remove(best_feature)
            else:
                break
                 
    def fit(self):
        """
        Fits the model using the selected features.
        """
        #--- Write your code here ---#
         # Select the features from the dataset indices.
        X_selected = self.X[:, self.selected_features]

        # Train the model using the selected features.
        self.model.fit(X_selected, self.y)

    def predict(self, X_test):
        """
        Predicts the target labels for the given test features.

        Parameters:
            X_test (array-like): Test features.

        Returns:
            array-like: Predicted target labels.
        """
        #--- Write your code here ---#
        # Select the features from the test dataset based on the selected features indices.
        X_test_selected = X_test[:, self.selected_features]

        # Predict the target labels using the trained model with the selected test features.
        return self.model.predict(X_test_selected)
    

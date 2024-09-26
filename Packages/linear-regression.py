import pandas as pd 
import numpy as np

class simplelinearregression():
    
    def __init__ (self):
        self.beta_0 = 0
        self.beta_1 = 0

    def fit(self,X, y):
        '''
        Fits model to data using least sqaure method

        Parameters:
        X : The independent variable (predictor).
        y : The dependent variable (response).
        
        Updates the instance variables:
        - self.beta_1: The slope of the regression line.
        - self.beta_0: The intercept of the regression line.
        '''
        
        X_mean = np.mean(X)
        y_mean = np.mean(y)

        numerator = np.sum((X - X_mean) * (y - y_mean))
        denominator = np.sum((X - X_mean) ** 2)

        self.beta_1 = numerator / denominator
        self.beta_0 = y_mean - self.beta_1 * X_mean

    
    def predict(self, X):
        '''
        Predict response values for given input X using least sqaure method.

        Parameters: 
        X : The independent variable (predictor).

        Returns:
        y_pred: The predicted y values based off the the predicted X inputs 
        '''
        
        return self.beta_0 + self.beta_1 * X

    def residuals(self, X, y):
        '''
        Return residuals.

        Parameters:
        X (array-like): The independent variable (predictor).
        y (array-like): The actual observed values of the dependent variable.
        
        Returns:
        array-like: Residuals (y - predicted y).
        '''
        
        y_pred = self.predict(X)

        return y - y_pred

    def rss(self, X, y):
        '''
        Calculate and return the sum of squared residuals (RSS).

        Parameters:
        X (array-like): The independent variable (predictor).
        y (array-like): The actual observed values of the dependent variable.
        
        Returns:
        float: The sum of squared residuals.
        '''
        
        residual = self.residuals(X,y)
        return np.sum(residual)

    def rse (self, X, y):
        '''
        Calculate and return the Residual Standard Error (RSE).

        Parameters:
        X (array-like): The independent variable (predictor).
        y (array-like): The actual observed values of the dependent variable.
        
        Returns:
        float: The Residual Standard Error (RSE).
        '''
        RSS = self.rss(X,y)
        return np.sqrt(RSS/(len(X)-2))

    def __str__(self):
        '''
        Return the linear regression equation as a string.

        Returns:
        str: A formatted string representing the linear regression equation.
        '''
        return f"SimpleLinearRegression Model: y = {self.beta_0:.2f} + {self.beta_1:.2f} * x"

        

    
    
        

        
        

        
        



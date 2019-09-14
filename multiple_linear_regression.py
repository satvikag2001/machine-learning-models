#Multiple Linear Regresssion

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import a dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[: , :-1].values
Y = dataset.iloc[: , 4].values

"""from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= 'NaN', strategy = 'mean')
imputer = imputer.fit_transform(X[:, 1:3])"""

#encoding categorical values
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X = LabelEncoder()
X[: , 3] = labelencoder_X.fit_transform(X[: , 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy variable Trap
X = X[: ,1:]

#Splitting the dataset into a Training set and a Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_opt, Y, test_size = 0.2, random_state = 0)

#Fitting Multiple Linear Regression to Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#Predictiong the Test set results
Y_pred = regressor.predict(X_test)

#Building the optimal model using Backward elimination
import statsmodels.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int),values = X, axis = 1)
X_opt = X[: , [0, 1, 2, 3, 4, 5]]
regressor_ols = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_ols.summary() 
#1
X_opt = X[: , [0, 1, 3, 4, 5]]
regressor_ols = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_ols.summary() 
#2
X_opt = X[: , [0, 3, 4, 5]]
regressor_ols = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_ols.summary() 
#3
X_opt = X[: , [0,  3,  5]]
regressor_ols = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_ols.summary() 
#4
X_opt = X[: , [0, 3]]
regressor_ols = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_ols.summary() 

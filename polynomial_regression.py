#Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

"""# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting Linear Regressor
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X , y)

#Fitting Polynomial Regressor
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly , y)

#Visualizing the Linear Regression model
plt.scatter(X , y, color = "red")
plt.plot(X, linreg.predict(X), color = "blue")
plt.title("Truth Or Bluff(Linear Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")

#Visualizing the Polynomial Regression model
X_grid  = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X , y, color = "red")
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = "blue")
plt.title("Truth Or Bluff(Polynomial Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")

#Predicting a new result with Linear regression
linreg.predict(np.array([6.5]).reshape(1 ,1))

#Prediction a new result with Polynomial regression
lin_reg2.predict(poly_reg.fit_transform(np.array([6.5]).reshape(1 ,1)))
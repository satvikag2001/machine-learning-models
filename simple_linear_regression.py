#Simple Linear Regresssion
#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import a dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[: , :-1].values
Y = dataset.iloc[: , 1].values

"""from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= 'NaN', strategy = 'mean')
imputer = imputer.fit_transform(X[:, 1:3])

#encoding categorical values
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X = LabelEncoder()
X[: , 0] = labelencoder_X.fit_transform(X[: , 0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)"""

#Splitting the dataset into a Training set and a Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

"""#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#Fitting Simple Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Predicting the Test set Observations
Y_pred = regressor.predict(X_test)

#Plotting a Training set graph 
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train) , color = 'blue')
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('salary')
plt.show()

#Plotting a Test set graph 
 plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train) , color = 'blue')
plt.title('Salary vs Experience(Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('salary')
plt.show()
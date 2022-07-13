import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("../data/Salary_Data.csv")
# select column X
X = dataset.iloc[:, :-1].values
# select column y
y = dataset.iloc[:, -1].values

# divide the training set into 2 parts with the train data = 80% and test data is 20% (test_size=0.2),
# random_state is how the function will randomly pick data for both training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_predicted = regressor.predict(X_test)

plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title('Salary vs Exp')
plt.show()

plt.scatter(X_test, y_test, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title('Salary vs Exp')
plt.show()






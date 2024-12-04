import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data

data = get_data()
# inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# TODO: calculate closed-form solution
theta_best = [0, 0]
ones = np.ones((x_train.shape[0]))
x_train_dot = np.vstack([np.ones(len(x_train)), x_train]).T
#print(x_pom)
theta_best = np.linalg.inv(x_train_dot.T @ x_train_dot) @ x_train_dot.T @ y_train.T
#print(theta_best)

# TODO: calculate error

#y_predicted_train = theta_best[0] + theta_best[1] * x_train

#MSE_train = np.mean((y_predicted_train - y_train) ** 2


y_predicted_test = theta_best[0] + theta_best[1] * x_test

MSE_test1 = np.mean((y_predicted_test - y_test) ** 2)
print(MSE_test1)



# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# TODO: standardization
x_standardized = (x_train - np.mean(x_train)) / np.std(x_train)
y_standardized = (y_train - np.mean(y_train)) / np.std(y_train)


# TODO: calculate theta using Batch Gradient Descent
learning_rate = 0.001
epsilon = 0.0000001
m = len(x_standardized)
theta = [[np.random.random()], [np.random.random()]]

ones = np.ones((x_standardized.shape[0]))
x_standardized_dot = np.vstack([np.ones(len(x_standardized)), x_standardized]).T
y_1 = y_standardized[:, np.newaxis]

while True:
    MSE_standardized1 = np.mean((x_standardized_dot@theta - y_train)**2)

    gradient = (2 / m) * x_standardized_dot.T @ (x_standardized_dot @ theta - y_1)
    theta = theta - learning_rate * gradient

    MSE_standardized2 = np.mean((x_standardized_dot@theta - y_train)**2)

    if abs(MSE_standardized1 - MSE_standardized2) < epsilon:
        break
    #print(theta)


scaled_theta = np.copy(theta)
scaled_theta[1] = scaled_theta[1] * np.std(y_train) / np.std(x_train)
scaled_theta[0] = np.mean(y_train) - scaled_theta[1] * np.mean(x_train)
scaled_theta = scaled_theta.reshape(-1)

#print(theta)

#print("theta: ", theta)
# TODO: calculate error
#(scaled_theta)
MSE_standardized = np.mean((x_train_dot @ scaled_theta - y_train) ** 2)
#print(MSE_standardized)

ones = np.ones((x_train.shape[0]))
x_test_dot = np.vstack([np.ones(len(x_test)), x_test]).T
MSE_test = np.mean((x_test_dot @ scaled_theta - y_test) ** 2)
print(MSE_test)

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data (replace with your own data)
# Independent variables (features) - 2D array (e.g., years of experience, age)
X = np.array([[1, 22], [2, 25], [3, 28], [4, 32], [5, 35]])
# Dependent variable (target) - 1D array (e.g., salary)
y = np.array([20, 30, 40, 50, 60])

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Make predictions (optional)
y_pred = model.predict(X)

# Print the coefficients
print(f"Intercept: {model.intercept_}")
print("Coefficients:", model.coef_)

# Example predictions
example_X = np.array([[6, 40], [7, 45]])
example_pred = model.predict(example_X)
print("Predictions for new data:", example_pred)

# Plot the results (only possible if you use 1 feature and 1 target)
plt.scatter(X[:, 0], y, color='blue')  # Actual data points for the first feature
plt.plot(X[:, 0], y_pred, color='red')  # Regression line
plt.xlabel('Feature 1')
plt.ylabel('Target')
plt.title('Multiple Linear Regression')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data (feel free to replace this with your own data)
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Independent variable
y = np.array([1, 3, 2, 5, 4])                # Dependent variable

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Print the coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")

# Plot the results
plt.scatter(X, y, color='blue')           # Actual data points
plt.plot(X, y_pred, color='red')          # Regression line
plt.xlabel('X')
plt.ylabel('y')
plt.title('Simple Linear Regression')
plt.show()

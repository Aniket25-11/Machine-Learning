import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the SVM classifier
svm = SVC(kernel='linear')  # You can change 'linear' to 'rbf' or 'poly' for different kernels
svm.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = svm.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Show Features, Actual and Predicted
results = pd.DataFrame(X_test, columns=iris.feature_names)
results['Actual'] = y_test
results['Predicted'] = y_pred
results['Actual'] = results['Actual'].map(lambda x: iris.target_names[x])
results['Predicted'] = results['Predicted'].map(lambda x: iris.target_names[x])
print(results)

# Example prediction for a new sample
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example features
new_prediction = svm.predict(new_sample)
print(f"Predicted class for the new sample: {iris.target_names[new_prediction][0]}")

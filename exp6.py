import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Sample weather data
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Windy': ['False', 'True', 'False', 'False', 'False', 'True', 'True', 'False', 'False', 'False', 'True', 'True', 'False', 'True'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert categorical data to numeric
df = pd.get_dummies(df, drop_first=True)

# Features and target variable
X = df.drop(columns=['Play_Yes'])
y = df['Play_Yes']

# Train the Decision Tree Classifier
clf = DecisionTreeClassifier(criterion='entropy')  # ID3 algorithm uses entropy
clf.fit(X, y)

# Plot the tree
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['No', 'Yes'])
plt.title("Decision Tree for Weather Detection (using ID3 Algorithm)")
plt.show()

# Classify a new sample
new_sample = pd.DataFrame({'Outlook_Sunny': [0], 'Outlook_Overcast': [0], 'Temperature_Mild': [1], 'Temperature_Hot': [0], 'Humidity_Normal': [1], 'Windy_True': [0]})
prediction = clf.predict(new_sample)
print(f"Predicted class for the new sample: {'Yes' if prediction[0] else 'No'}")


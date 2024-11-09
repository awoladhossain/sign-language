# Model training
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the dataset
data_dict = pickle.load(open("american.pickle", "rb"))

# Get the maximum length of the data points
max_length = max(len(x) for x in data_dict["data"])

# Pad the data points to the maximum length
data = np.asarray([np.pad(x, (0, max_length - len(x)), 'constant') for x in data_dict["data"]])
labels = np.asarray(data_dict["labels"])

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Create and train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions and evaluate the model
y_predict = model.predict(x_test)
accuracy = accuracy_score(y_predict, y_test)

# Print the accuracy
print(f"Accuracy: {accuracy:.2f}")

# Save the trained model to a pickle file
with open("americansign.p", "wb") as f:
    pickle.dump({"model": model}, f)

# Accuracy: 0.95
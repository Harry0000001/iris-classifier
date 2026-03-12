from sklearn.datasets import load_iris
iris=load_iris()
X=iris.data #shape(150,4)
y=iris.target #shape(150,)
print(iris.feature_names, iris.target_names)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Predictions:", y_pred[:5])
print("True labels:", y_test[:5])
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

# Assuming y_test and y_pred are already defined
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

# Create the ConfusionMatrixDisplay object with matching labels
# Make sure the number of labels matches the size of the confusion matrix
unique_labels = np.unique(np.concatenate([y_test, y_pred]))
cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix, 
    display_labels=unique_labels  # This ensures labels match the actual classes in your data
)
import os
os.makedirs("outputs",exist_ok=True)
# Plot and show
cm_display.plot()
plt.savefig("outputs/Confusion matrix.png")
plt.show()
import joblib
joblib.dump(model,"outputs/model.joblib")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Load dataset
# Example: simple dataset (you can replace with your own CSV)
data = pd.DataFrame({
    "feature1": [1, 2, 3, 4, 5, 6],
    "feature2": [2, 4, 6, 8, 10, 12],
    "label":    [0, 0, 0, 1, 1, 1]
})

# Step 2: Split data
X = data[["feature1", "feature2"]]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 4: Test model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy}")

# Step 5: Save model
joblib.dump(model, "model.pkl")

print("Model saved as model.pkl")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# 1. Load dataset
df = pd.read_csv("iris.csv")

# 2. Split features and target
# change 'species' if your column name is different
X = df.drop("species", axis=1)
y = df["species"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Train model (Version 1)
model = LogisticRegression(C=2.0)
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model v2 Accuracy:", accuracy)

# 7. Save model
joblib.dump(model, "model_v3.pkl")
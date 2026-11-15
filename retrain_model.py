import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("student_data/student-mat.txt", sep=";")

# Create target
df["Result"] = df["G3"].apply(
    lambda marks: "Pass" if marks >= 10 else "Fail"
)

# Select features
X = df[
    [
        "studytime",
        "failures",
        "absences",
        "G1",
        "G2",
        "age",
        "health",
        "freetime"
    ]
]

y = df["Result"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model trained successfully!")
print("Accuracy:", round(accuracy * 100, 2), "%")

# Save model
joblib.dump(model, "student_model.pkl")

print("New student_model.pkl saved successfully!")
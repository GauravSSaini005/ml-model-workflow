import os
from pandas import read_csv
from joblib import dump
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

workspace = os.getenv('GITHUB_WORKSPACE')
model_cleaning_dir = os.path.join(workspace, 'ModelCleaning')
csv_file_path = os.path.join(model_cleaning_dir, 'cleaned_data.csv')

# Check the file exists before loading
if os.path.exists(csv_file_path):
    print(f"File found: {csv_file_path}")
else:
    print(f"File not found at: {csv_file_path}")

df = read_csv(csv_file_path)
print(df.head())

# Age = input, Salary = what we predict
X = df["age"].values.reshape(-1, 1)
y = df["salary"]

# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
mind = LinearRegression()
mind.fit(X_train, y_train)

# Save the trained model
model_path = os.path.join(model_cleaning_dir, "AgeSalaryModel.pkl")
dump(mind, model_path)
print("Model saved as AgeSalaryModel.pkl")

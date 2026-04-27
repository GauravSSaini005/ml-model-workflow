import os
import pandas as pd

workspace = os.getenv('GITHUB_WORKSPACE')
model_cleaning_dir = os.path.join(workspace, 'ModelCleaning')
csv_path = os.path.join(model_cleaning_dir, 'data.csv')

df = pd.read_csv(csv_path)

# Show original data
print("Original Data:")
print(df.head())

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Remove leading/trailing whitespace from all string values
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

# Convert all strings to lowercase
df = df.map(lambda x: x.lower() if isinstance(x, str) else x)

# Drop rows with any missing values
df.dropna(inplace=True)

# Drop duplicate rows
df.drop_duplicates(inplace=True)

# Build output path and save cleaned file
output_path = os.path.join(model_cleaning_dir, 'cleaned_data.csv')
os.makedirs(model_cleaning_dir, exist_ok=True)
df.to_csv(output_path, index=False)

print(output_path)
print("Cleaned Data:")
print(df.head())
print("\nCleaned data saved to 'cleaned_data.csv'")

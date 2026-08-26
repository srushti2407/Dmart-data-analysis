import pandas as pd

data = pd.read_csv("landmarks.csv")

print("First 5 rows:")
print(data.head())

print("\nColumns:")
print(data.columns)

print("\nShape:")
print(data.shape)
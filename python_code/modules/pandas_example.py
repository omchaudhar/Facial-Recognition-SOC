
import pandas as pd
import numpy as np

# Creating a Series
s = pd.Series([1, 3, 5, np.nan, 6, 8])
print(f"Series:\n{s}")

# Creating a DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']
}
df = pd.DataFrame(data)
print(f"\nDataFrame:\n{df}")

# Reading data from a CSV file (example, assuming 'data.csv' exists)
# df_csv = pd.read_csv('data.csv')
# print(f"\nDataFrame from CSV:\n{df_csv}")

# Basic DataFrame operations
print(f"\nColumn 'Name':\n{df['Name']}")
print(f"\nFirst two rows:\n{df.head(2)}")
print(f"\nDescriptive statistics:\n{df.describe()}")

# Selecting data using loc and iloc
print(f"\nSelect row by label (index 0):\n{df.loc[0]}")
print(f"\nSelect row by integer position (index 1):\n{df.iloc[1]}")
print(f"\nSelect 'Name' and 'Age' for first two rows:\n{df.loc[0:1, ['Name', 'Age']]}")

# Filtering data
filtered_df = df[df['Age'] > 30]
print(f"\nFiltered DataFrame (Age > 30):\n{filtered_df}")

# Adding a new column
df['Salary'] = [50000, 60000, 75000, 90000]
print(f"\nDataFrame with new 'Salary' column:\n{df}")

# Grouping data
# Assuming we had more diverse data for grouping, e.g., by 'City'
# df.groupby('City')['Salary'].mean()

# Handling missing data (example with a Series)
s_with_nan = pd.Series([1, 2, np.nan, 4, 5])
print(f"\nSeries with NaN:\n{s_with_nan}")
print(f"Filled NaN with 0:\n{s_with_nan.fillna(0)}")
print(f"Dropped NaN:\n{s_with_nan.dropna()}")



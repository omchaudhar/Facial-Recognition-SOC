
import numpy as np

# Creating NumPy arrays
# From a list
a = np.array([1, 2, 3])
print(f"1D Array: {a}")

# From a list of lists (2D array)
b = np.array([[1, 2], [3, 4]])
print(f"2D Array:\n{b}")

# Creating arrays with placeholders
c = np.zeros((2, 3)) # Array of zeros
print(f"Array of zeros:\n{c}")
d = np.ones((2, 2))  # Array of ones
print(f"Array of ones:\n{d}")
e = np.full((2, 2), 7) # Array filled with a specific value
print(f"Array filled with 7:\n{e}")
f = np.arange(0, 10, 2) # Array with a range of values
print(f"Array with range: {f}")

# Array attributes
print(f"Shape of b: {b.shape}") # (rows, columns)
print(f"Number of dimensions of b: {b.ndim}")
print(f"Data type of b elements: {b.dtype}")

# Array indexing and slicing
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(f"Element at index 0: {arr[0]}")
print(f"Elements from index 2 to 5: {arr[2:6]}")

# 2D array indexing
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"Element at (0, 1): {matrix[0, 1]}") # Row 0, Column 1
print(f"First row: {matrix[0, :]}")
print(f"Second column: {matrix[:, 1]}")

# Basic array operations
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
print(f"Addition: {arr1 + arr2}")
print(f"Multiplication: {arr1 * arr2}")
print(f"Dot product: {arr1.dot(arr2)}")

# Universal functions (ufuncs)
print(f"Square root: {np.sqrt(arr1)}")
print(f"Exponential: {np.exp(arr1)}")

# Broadcasting example
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([10, 20, 30])
print(f"Broadcasting addition:\n{a + b}")



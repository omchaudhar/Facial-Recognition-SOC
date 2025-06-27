import math
from random import randint

# Example of variables and data types
my_integer = 10
my_float = 20.5
my_string = "Hello, Python!"
my_boolean = True
my_list = [1, 'two', 3.0]
my_tuple = (4, 'five', 6.0)
my_dictionary = {'key1': 'value1', 'key2': 123}

print(f"Integer: {my_integer}, Type: {type(my_integer)}")
print(f"Float: {my_float}, Type: {type(my_float)}")
print(f"String: {my_string}, Type: {type(my_string)}")
print(f"Boolean: {my_boolean}, Type: {type(my_boolean)}")
print(f"List: {my_list}, Type: {type(my_list)}")
print(f"Tuple: {my_tuple}, Type: {type(my_tuple)}")
print(f"Dictionary: {my_dictionary}, Type: {type(my_dictionary)}")

# Example of conditional statements
x = 10
if x > 0:
    print("x is positive")
elif x == 0:
    print("x is zero")
else:
    print("x is negative")

# Example of a for loop
for i in range(5):
    print(f"For loop iteration: {i}")

# Example of a while loop
count = 0
while count < 3:
    print(f"While loop iteration: {count}")
    count += 1

# Example of a function definition and call
def greet(name):
    return f"Hello, {name}!"

message = greet("Alice")
print(message)

# Example of importing a module
print(f"The value of pi is: {math.pi}")

# Example of importing a specific function from a module
print(f"Random integer between 1 and 10: {randint(1, 10)}")



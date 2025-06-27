
import os

# Get the current working directory
current_directory = os.getcwd()
print(f"Current working directory: {current_directory}")

# List files and directories in the current path
print("Files and directories in current path:")
for item in os.listdir(current_directory):
    print(item)

# Create a new directory
new_dir = "my_new_directory"
if not os.path.exists(new_dir):
    os.makedirs(new_dir)
    print(f"Directory \'{new_dir}\' created.")
else:
    print(f"Directory \'{new_dir}\' already exists.")

# Join path components intelligently
file_path = os.path.join(current_directory, new_dir, "my_file.txt")
print(f"Constructed file path: {file_path}")



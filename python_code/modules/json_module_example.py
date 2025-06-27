
import json

# Python dictionary
data = {
    "name": "John Doe",
    "age": 30,
    "isStudent": False,
    "courses": [{"title": "Math", "credits": 3}, {"title": "Science", "credits": 4}]
}

# Convert Python dictionary to JSON string
json_string = json.dumps(data, indent=4)
print("\nJSON string:")
print(json_string)

# Convert JSON string back to Python dictionary
parsed_data = json.loads(json_string)
print("\nParsed data (Python dictionary):")
print(parsed_data["name"])



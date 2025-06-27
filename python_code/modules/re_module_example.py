
import re

text = "The quick brown fox jumps over the lazy dog."

# Search for a pattern
match = re.search(r"fox", text)
if match:
    print(f"Found \'fox\' at index {match.start()}")

# Find all occurrences of a pattern
all_matches = re.findall(r"o.", text)
print(f"All matches for \'o.\': {all_matches}")

# Replace a pattern
new_text = re.sub(r"quick", "slow", text)
print(f"Text after replacement: {new_text}")




from datetime import datetime, timedelta

# Get current date and time
now = datetime.now()
print(f"Current date and time: {now}")

# Format date and time
formatted_date = now.strftime("%Y-%m-%d %H:%M:%S")
print(f"Formatted date and time: {formatted_date}")

# Create a specific date
specific_date = datetime(2024, 1, 1, 10, 30, 0)
print(f"Specific date: {specific_date}")

# Perform date arithmetic
tomorrow = now + timedelta(days=1)
print(f"Tomorrow: {tomorrow}")

# Calculate difference between dates
diff = now - specific_date
print(f"Time difference: {diff}")
print(f"Difference in days: {diff.days}")



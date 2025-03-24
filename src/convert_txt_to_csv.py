import csv

# Define the input and output file paths
input_file = 'C:\\Users\\thoma\\OneDrive\\Documents\\CS 529\\fighters_stats.txt'  # Replace with your .txt file name
output_file = 'fighter_stats.csv'  # Output .csv file name

# Initialize a list to hold each fighter's information
fighters_data = []

# Read the .txt file and parse the data
with open(input_file, 'r') as file:

    fighter = {}
    
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespace
        
        # Check if the line is not empty and contains ":"
        if line and ': ' in line:
            # Split into key and value, with a fallback for missing values
            key, value = line.split(': ', 1) if ': ' in line else (line, None)
            fighter[key] = value if value else None  # Set value to None if missing

        elif not line:  # Blank line signifies end of a fighter's data
            if fighter:  # If there's data in the dictionary
                fighters_data.append(fighter)
                fighter = {}  # Reset for the next fighter

    if fighter:  # Append the last fighter's data
        fighters_data.append(fighter)

# Get the headers from the first fighter's keys
headers = fighters_data[0].keys()

# Write the parsed data to a CSV file
with open(output_file, 'w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=headers)
    writer.writeheader()
    writer.writerows(fighters_data)

print(f"Data has been successfully written to {output_file}")

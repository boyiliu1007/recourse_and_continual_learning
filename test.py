import csv

# A simple 1D list
data = [1, 2, 3, 4, 5]

# Writing the list to a CSV file as a single column
with open('output.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for item in data:
        writer.writerow([item])  # Write each item in a new row
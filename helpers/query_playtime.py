import csv

# Count rows where total_playtime_min != 0
count = 0
total_rows = 0
field_name = 'total_playtime_mins'

with open('outputs/feature_matrix.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        total_rows += 1
        if row.get(field_name, '0') != '0':
            count += 1

print(f"Number of rows with {field_name} != 0: {count}")
print(f"Total rows: {total_rows}")

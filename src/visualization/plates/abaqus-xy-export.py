from abaqus import session
import csv

# inputs:
load_factor = 1400000
xy_data_name = 'xydata'

# Get the XY data from the session
xy_data = session.xyDataObjects[xy_data_name]

# Define the output file path
output_file = 'C:/Users/Hamed/projects/thesis/xydata.csv'

# Create and write to the CSV file
with open(output_file, 'wb') as csvfile:
    writer = csv.writer(csvfile)
    # Write headers
    writer.writerow(['X', 'Y'])
    # Write the data points
    for data_point in xy_data:
        if data_point:
            writer.writerow([data_point[0] * load_factor, abs(data_point[1])])

print('XY data has been successfully exported to {}'.format(output_file))

# Show the content of the saved file
with open(output_file, 'r') as file:
    content = file.read()

print(content)

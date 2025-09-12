from abaqus import session
import csv

# inputs:
load_factor = 1
data_names = ["sm1", "sm2", "sm3"]

def export_xy_data(data_name):
    xy_data = session.xyDataObjects[data_name]
    output_file = f'C:/Users/Hamed/projects/thesis/{data_name}.csv'
    with open(output_file, 'w', newline='') as csvfile:  # Added newline=''
        writer = csv.writer(csvfile)
        writer.writerow(['X', 'Y'])
        for data_point in xy_data:
            if data_point:
                writer.writerow([data_point[0] * load_factor, abs(data_point[1])])
    print('XY data has been successfully exported to {}'.format(output_file))

for data_name in data_names:
    export_xy_data(data_name)

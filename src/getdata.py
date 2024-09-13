import os
import pandas as pd


root_path = "C:\\Users\\Hassan\\Desktop\\load"
data_path = os.path.join(root_path, "data.xlsx")
step = 10
data_with_step_path = os.path.join(root_path, f"data-step-{step}.xlsx")

def read_excel_sheet(file_dir, sheet_name, usecols=None):
    return pd.read_excel(file_dir, sheet_name=sheet_name, usecols=usecols)


def get_column_headers(dataframe):
    headers = dataframe.columns.ravel()
    return headers


def get_column_data(dataframe, column_header):
    return dataframe[column_header].tolist()


def get_col_number_from_header(dataframe, header):
    return dataframe.columns.get_loc(header)


def update_cell(dataframe, column_header, row_num, value):
    dataframe.at[row_num, column_header] = value
    return dataframe


def create_excel_file(path, use_columns):
    writer = pd.ExcelWriter(path, engine='openpyxl')
    empty_dataframe = pd.DataFrame(columns=use_columns)
    empty_dataframe.to_excel(writer, sheet_name='Sheet1', index=False)
    writer.close()


dataframe = read_excel_sheet(data_path, sheet_name="Sheet1", usecols=["time", "load"])
time = get_column_data(dataframe=dataframe, column_header="time")
load = get_column_data(dataframe=dataframe, column_header="load")
total_data_count = len(load)


create_excel_file(data_with_step_path, ["time", "load"])
dataframe_with_step = read_excel_sheet(data_with_step_path, sheet_name="Sheet1", usecols=["time", "load"])

row_counter = 0
for i in range(0, total_data_count, step):
    dataframe_with_step = update_cell(
        dataframe=dataframe_with_step,
        column_header="time",
        row_num=row_counter,
        value=time[i],
    )
    dataframe_with_step = update_cell(
        dataframe=dataframe_with_step,
        column_header="load",
        row_num=row_counter,
        value=load[i],
    )
    dataframe_with_step.to_excel(data_with_step_path, sheet_name="Sheet1", index=False)
    row_counter += 1
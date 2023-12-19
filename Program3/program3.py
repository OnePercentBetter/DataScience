import pandas as pd
from sklearn.model_selection import train_test_split
def import_data(csv_file, names=None):
    """Imports and drops columns"""
    data_frame= pd.read_csv(csv_file)
    data_frame = data_frame.drop(columns=['DATE'])
    if names is not None:
        for prev_name, cur_name in names.items():
            data_frame = data_frame.rename(columns={prev_name : cur_name})
    for col in data_frame.columns:
        data_frame[col] = pd.to_numeric(data_frame[col], errors='coerce')
    data_frame = data_frame.dropna()
    return data_frame

def split_data(data_frame,xes_col_names, y_col_name,test_size = 0.33, random_state = 106):
    """Splits the Data"""
    independent = data_frame[xes_col_names]
    dependent = data_frame[y_col_name]
    xes_train, xes_test, dependent_train, dependent_test = train_test_split(independent, dependent,
        test_size=test_size, random_state=random_state)
    return xes_train, xes_test, dependent_train, dependent_test

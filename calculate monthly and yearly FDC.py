import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from joblib import Parallel, delayed

def clean_data(df):
    df['Streamflow (m3/s)'] = pd.to_numeric(df['Streamflow (m3/s)'], errors='coerce')
    df = df[df['Streamflow (m3/s)'] >= 0].dropna(subset=['Streamflow (m3/s)']).copy()
    return df

def calculate_fdc(flow_data):
    if len(flow_data) == 0:
        return ['No Data'] * 101  # Return 'No Data' if there is no data
    if np.all(flow_data == 0):
        # Replace one of the values with 0.01 to avoid an all-zero FDC especially in a arid climate
        flow_data[0] = 0.01
    sorted_flow = np.sort(flow_data)
    exceedance_probabilities = np.arange(1, len(sorted_flow) + 1) / (len(sorted_flow) + 1) * 100
    return np.interp(np.arange(0, 101), exceedance_probabilities, sorted_flow)

def process_station_file(file_path):
    try:
        # Preserve the decimal portion in station_id by removing only the .csv extension
        station_id = os.path.basename(file_path).rsplit('.', 1)[0]
        df = pd.read_csv(file_path, parse_dates=['Datetime'])
        df = clean_data(df)
        df['Month'] = df['Datetime'].dt.strftime('%b')

        overall_fdc = calculate_fdc(df['Streamflow (m3/s)'].values)

        monthly_fdc = {}
        for month, group in df.groupby('Month'):
            fdc = calculate_fdc(group['Streamflow (m3/s)'].values)
            monthly_fdc[month] = fdc

        return station_id, overall_fdc, monthly_fdc
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def create_fdc_dataframe_parallel(file_paths):
    columns = ['Station ID', 'Time'] + [f'{i}%' for i in range(101)]
    rows = []

    # Use joblib to parallelize the processing of files
    results = Parallel(n_jobs=-1)(delayed(process_station_file)(file_path) for file_path in tqdm(file_paths, desc='Processing files'))

    for result in results:
        if result is None:
            continue
        station_id, overall_fdc, monthly_fdc = result

        # Convert NumPy array to list for overall FDC
        overall_row = [station_id, 'Overall'] + overall_fdc.tolist()
        rows.append(overall_row)

        # Append monthly FDC rows
        all_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for month in all_months:
            if month in monthly_fdc:
                fdc_values = monthly_fdc[month]
            else:
                fdc_values = ['No Data'] * 101
            # Convert NumPy array to list for monthly FDC if it exists
            month_row = [station_id, month] + (fdc_values.tolist() if isinstance(fdc_values, np.ndarray) else fdc_values)
            rows.append(month_row)

    # Create the final DataFrame
    fdc_df = pd.DataFrame(rows, columns=columns)
    return fdc_df

if __name__ == '__main__':
    folder_path = 'path to your gauge data folder'
    file_paths = [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if file_name.endswith('.csv')]
    fdc_df = create_fdc_dataframe_parallel(file_paths)

    # Save the DataFrames to separate CSV files
    overall_output_file_path = 'folder path to save data/overall_fdc.csv'
    monthly_output_file_path = 'folder path to save data/monthly_fdc.csv'

    # Separate overall and monthly data
    overall_fdc_df = fdc_df[fdc_df['Time'] == 'Overall']
    monthly_fdc_df = fdc_df[fdc_df['Time'] != 'Overall']

    overall_fdc_df.to_csv(overall_output_file_path, index=False)
    monthly_fdc_df.to_csv(monthly_output_file_path, index=False)

    print(f"Overall FDC data has been successfully saved to {overall_output_file_path}")
    print(f"Monthly FDC data has been successfully saved to {monthly_output_file_path}")

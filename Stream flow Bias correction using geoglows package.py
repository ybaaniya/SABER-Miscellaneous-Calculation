import os
import pandas as pd
from geoglows.bias import correct_historical
from multiprocess import Pool, cpu_count, Manager



# Assuming the structure for the folders
simulated_data_folder = 'folder path to your simulated timeseries '
observed_data_folder = 'folder path to your observed timeseries'
output_folder = 'folder path for bias corrected timeseries'
# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load the CSV file
file_path = 'path that contains master csv that has information of corresponding simulated and onserved station'
data = pd.read_csv(file_path)


# Function to clean data
def clean_data(df, column_name):
    # Remove non-numeric values and convert to numeric for the relevant column
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    # Remove negative values
    df = df[df[column_name] >= 0]
    # Drop rows with NaN values in the relevant column
    df = df.dropna(subset=[column_name])
    return df

def process_row(row, skipped_rows, completed_rows):
    try:
        # Construct file paths
        simulated_file_path = os.path.join(simulated_data_folder, f"{int(row['linkno'])}.csv")  ##change the column heading that represent simulated id in master csv
        observed_file_path = os.path.join(observed_data_folder, f"{row['matching_column']}.csv")  ## change the column heading to represent obserbed station in master csv

        # Check if both files exist
        if not os.path.exists(simulated_file_path):
            skipped_rows.append(row['matching_column'])
            return None
        if not os.path.exists(observed_file_path):
            skipped_rows.append(row['matching_column'])
            return None

        # Load the simulated and observed data
        simulated_data = pd.read_csv(simulated_file_path, index_col=0, parse_dates=True)
        observed_data = pd.read_csv(observed_file_path, index_col=0, parse_dates=True)

        # Clean the data
        simulated_data = clean_data(simulated_data, 'Qout')
        observed_data = clean_data(observed_data, 'Streamflow (m3/s)')

        # Perform bias correction
        corrected_data = correct_historical(simulated_data, observed_data)

        # Save the resultant dataframe
        output_file_path = os.path.join(output_folder, f"{row['matching_column']}.csv")
        corrected_data.to_csv(output_file_path)
        
        # Append to completed rows
        completed_rows.append(row['matching_column'])

        # Print statement after each successful processing
        print(f"Bias correction completed for: {row['matching_column']}")
        
    except Exception as e:
        print(f"Error processing {row['matching_column']}: {e}")
        skipped_rows.append(row['matching_column'])

if __name__ == "__main__":
    num_processes = cpu_count()
    rows = [row for index, row in data.iterrows()]

    manager = Manager()  # Create Manager instance
    skipped_rows = manager.list()
    completed_rows = manager.list()

    # Use multiprocess Pool for parallel processing
    with Pool(num_processes) as pool:
        pool.starmap(process_row, [(row, skipped_rows, completed_rows) for row in rows])

    # Save skipped rows to a CSV file
    skipped_file_path = os.path.join(output_folder, 'skipped_rows.csv')
    pd.DataFrame(list(skipped_rows), columns=['matching_column']).to_csv(skipped_file_path, index=False)

    # Ensure the script exits after finishing
    print("All tasks completed.")
    print(f"Skipped rows: {list(skipped_rows)}")

    # Print completed rows
    for completed_file in completed_rows:
        print(f"Bias corrected: {completed_file}")

    import sys
    sys.exit()

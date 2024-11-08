#Download all necessary Package
import pandas as pd
import numpy
import zarr
import zarr
import os

#Read the monthly csv file obtained from the overall and monthly fdc for all gauge python files
df = pd.read_csv('/Users/yubinbaaniya/Documents/WORLD BIAS/saber workdir/tables/monthly_fdc.csv')

#The code calculates non probability of exceedence but we are concerned with probability of exceedence.
#Below code will change it by reversing the values

rename_dict = {f'{i}%': f'{100 - i}' for i in range(101) if f'{i}%' in df.columns}
# Rename the columns
df.rename(columns=rename_dict, inplace=True)

#Change the files in format to convert it to zarr
# Melt the DataFrame to convert percent exceedance columns into rows
melted_df = df.melt(id_vars=['Station ID', 'Time'], 
                    var_name='p_exceed', 
                    value_name='Value')

# Convert 'Time' and 'p_exceed' to ordered categorical types to control sorting
# Assuming the months are abbreviated as Jan, Feb, etc.
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
melted_df['Time'] = pd.Categorical(melted_df['Time'], categories=month_order, ordered=True)
melted_df['p_exceed'] = pd.to_numeric(melted_df['p_exceed'])  # Ensure p_exceed is numeric for sorting

# Sort by 'Station ID', 'Time', and 'p_exceed' (descending for p_exceed)
melted_df = melted_df.sort_values(by=['Station ID', 'Time', 'p_exceed'], ascending=[True, True, False]).reset_index(drop=True)

# Define a mapping from month abbreviations to numbers
month_mapping = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}

# Apply the mapping to the 'Time' column
melted_df['Time'] = melted_df['Time'].map(month_mapping)
melted_df.rename(columns={'Time': 'Month', 'Value': 'fdc', 'Station ID': 'gauge'}, inplace=True)

#Extra step to see if there is any NaN value in the data adn if present remove it
# Step 1: Ensure that the 'fdc' column contains only numeric values
melted_df['fdc'] = pd.to_numeric(melted_df['fdc'], errors='coerce')

# Step 2: Drop rows with NaN values in 'fdc' after conversion (indicating non-numeric entries)
initial_row_count = len(melted_df)
melted_df = melted_df.dropna(subset=['fdc'])
dropped_row_count = initial_row_count - len(melted_df)
print(f"Dropped {dropped_row_count} rows due to non-numeric 'fdc' values.")

#Prepare column for the Zarr file
melted_df['Month'] = melted_df['Month'].astype(int)
melted_df['p_exceed'] = melted_df['p_exceed'].astype(int)

# Pivot the DataFrame to have 'gauge' as the primary index, 'p_exceed' as rows, and 'Month' as columns
# This will align with having (gauge, p_exceed, month) dimensions for the Zarr file
pivot_df = melted_df.pivot(index='gauge', columns=['p_exceed', 'Month'], values='fdc')

# Get the unique values for coordinates
gauges = melted_df['gauge'].unique()
months = melted_df['Month'].unique()
p_exceed_values = melted_df['p_exceed'].unique()

#ZARR functions
def create_xarray_zarr(melted_df):
    """
    Convert melted DataFrame to xarray Dataset and save as Zarr
    
    Parameters:
    melted_df (pd.DataFrame): Melted DataFrame with columns 'gauge', 'Month', 'p_exceed', 'fdc'
    """
    # Ensure proper data types
    melted_df = melted_df.copy()
    melted_df['Month'] = melted_df['Month'].astype(int)
    melted_df['p_exceed'] = melted_df['p_exceed'].astype(int)
    
    # Sort values to ensure consistent ordering
    melted_df = melted_df.sort_values(['gauge', 'p_exceed', 'Month'])
    
    # Get unique values for dimensions
    gauges = sorted(melted_df['gauge'].unique())
    p_exceed_values = sorted(melted_df['p_exceed'].unique())
    months = sorted(melted_df['Month'].unique())
    
    # Create 3D array with proper shape
    shape = (len(gauges), len(p_exceed_values), len(months))
    data = np.full(shape, np.nan)
    
    # Create lookup dictionaries for faster indexing
    gauge_idx = {g: i for i, g in enumerate(gauges)}
    p_idx = {p: i for i, p in enumerate(p_exceed_values)}
    month_idx = {m: i for i, m in enumerate(months)}
    
    # Fill the 3D array
    for _, row in melted_df.iterrows():
        i = gauge_idx[row['gauge']]
        j = p_idx[row['p_exceed']]
        k = month_idx[row['Month']]
        data[i, j, k] = row['fdc']
    
    # Create xarray Dataset
    ds = xr.Dataset(
        {
            'fdc': (['gauge', 'p_exceed', 'month'], data)
        },
        coords={
            'gauge': gauges,
            'p_exceed': p_exceed_values,
            'month': months
        }
    )
    
    # Chunk the dataset - adjust chunk sizes based on your needs
    ds = ds.chunk({
        'gauge': min(50, len(gauges)),
        'p_exceed': min(101, len(p_exceed_values)),
        'month': min(12, len(months))
    })
    
    return ds

def save_to_zarr(ds, filename='/Users/yubinbaaniya/Downloads/fdc.zarr'):  #the file path is hardcoded here
    """
    Save xarray Dataset to Zarr format and return the absolute path
    
    Parameters:
    ds (xarray.Dataset): Dataset to save
    filename (str): Output filename with path
    
    Returns:
    str: Absolute path to the saved Zarr file
    """
    # Convert to absolute path
    abs_path = os.path.abspath(filename)
    
    # Save to Zarr format without compression
    ds.to_zarr(abs_path, mode='w')
    
    print(f"Zarr file saved to: {abs_path}")
    return abs_path


#Function Call to make a zarr file
ds = create_xarray_zarr(melted_df)
zarr_path = save_to_zarr(ds)
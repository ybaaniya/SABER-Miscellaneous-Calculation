import xarray as xr
import pandas as pd
import numpy as np
import os
import s3fs
from multiprocessing import Pool
import glob

retrospective_simulation_zarr_file = 's3://geoglows-v2-retrospective/retrospective.zarr'
zarr_variable = 'Qout'
earliest_date = '1950-01-01'
latest_date = '2019-12-31'
save_dir = 'path to save the z scaled fdc'
region_name = 'us-west-2'
s3 = s3fs.S3FileSystem(anon=True, client_kwargs=dict(region_name=region_name))
s3store = s3fs.S3Map(root=retrospective_simulation_zarr_file, s3=s3, check=False)


def hydrograph_to_fdc(river_id_list: np.array, file_name: str) -> None:
    # open the zarr file
    with xr.open_zarr(s3store) as ds:

        # select the hydrographs for the river_ids
        df = (
            ds
            [zarr_variable]
            .sel(rivid=river_id_list)
            .to_dataframe()
            [zarr_variable]
            .reset_index()
            .pivot(columns='rivid', values=zarr_variable, index='time')
        )

        # Filter the dataframe to intended range
        filtered_df = df.loc[earliest_date:latest_date]
        # Replace negative values with 0
        filtered_df = filtered_df.clip(lower=0)
        # DataFrame to a NumPy array
        data_array = filtered_df.values

        # Define the percentiles in a gap of 1
        percentiles = np.arange(100, -1, -1)

        # Calculate percentiles for each column/river_id
        river_percentiles = np.percentile(data_array, [100 - p for p in percentiles], axis=0)

        # Create a dictionary with river_id as keys and percentile values as values
        river_percentile_data = {
            river_id: {f'Q{p}': val for p, val in zip(percentiles, river_percentiles[:, i])}
            for i, river_id in enumerate(filtered_df.columns)
        }

        # Convert the dictionary to DataFrame and ensure float data type
        percentile_df = pd.DataFrame(river_percentile_data).T
        percentile_df = percentile_df.astype(float)
        # save the FDCs to parquet files
        annual_file_name = os.path.join(save_dir, f'fdc_group_{file_name}_annual.parquet')
        percentile_df.to_parquet(annual_file_name)

        # Add month column for grouping
        filtered_df['month'] = filtered_df.index.month

        # Function to calculate percentiles for each group
        def calculate_group_percentiles(group):
            river_percentiles = np.percentile(group, [100 - p for p in percentiles], axis=0)
            percentile_data = np.transpose(river_percentiles)
            return pd.DataFrame(percentile_data, index=group.columns, columns=[f'Q{p}' for p in percentiles])

        # Group by month and calculate percentiles
        grouped_percentiles = filtered_df.drop(columns='month').groupby(filtered_df['month']).apply(
            calculate_group_percentiles)

        # Save the monthly FDCs to parquet files
        monthly_file_name = os.path.join(save_dir, f'fdc_group_{file_name}_monthly.parquet')
        grouped_percentiles.to_parquet(monthly_file_name)

    return


if __name__ == '__main__':
    # make a list of river_ids and break it into chunks of rivid you want
    with xr.open_zarr(s3store) as ds:
        river_ids = ds.rivid.values
        river_ids = river_ids[-100:]
        river_id_chunks = np.array_split(river_ids, len(river_ids) // 10)

    jobs = [(chunk, i) for i, chunk in enumerate(river_id_chunks)]

    with Pool(8) as p:
        p.starmap(hydrograph_to_fdc, jobs)
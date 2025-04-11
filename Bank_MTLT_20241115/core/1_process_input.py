import os
import pandas as pd
from pathlib import Path
import numpy as np

# Function to load and process CSV files for a given date in 'mmdd' format
def load_and_process_csvs(path_base, mmdd, year='2024'):
    try:
        file_open = os.path.join(path_base, f"open_{mmdd}.csv")
        file_high = os.path.join(path_base, f"high_{mmdd}.csv")
        file_low = os.path.join(path_base, f"low_{mmdd}.csv")
        file_close = os.path.join(path_base, f"close_{mmdd}.csv")
        
        # Check if all necessary files exist before processing
        if not (os.path.exists(file_open) and os.path.exists(file_high) and os.path.exists(file_low) and os.path.exists(file_close)):
            print(f"Missing files for date {mmdd}, skipping...")
            return None

        # Load CSV files
        df_open = pd.read_csv(file_open)
        df_high = pd.read_csv(file_high)
        df_low = pd.read_csv(file_low)
        df_close = pd.read_csv(file_close)

        # Process each symbol column
        df_list = []
        for symbol in df_open.columns[1:]:
            df = pd.DataFrame({
                'open': df_open[symbol],
                'high': df_high[symbol],
                'low': df_low[symbol],
                'close': df_close[symbol],
                'symbol': symbol,
                'minutes': df_open.iloc[:, 0]
            })

            # Add 'datetime' and 'timestamp' columns
            date_str = f"{year}{mmdd}"
            df['datetime'] = pd.to_datetime(date_str + ' ' + df['minutes'], format='%Y%m%d %H:%M:%S')
            df['timestamp'] = df['datetime'].apply(lambda x: int(x.timestamp()))

            df_list.append(df)

        return pd.concat(df_list, ignore_index=True) if df_list else None

    except Exception as e:
        print(f"Error processing files for date {mmdd}: {e}")
        return None


# Function to resample data for different intervals
def resample_data(indexed_final_df, intervals):
    intervals_timeframe = [interval.replace('m', 'T').replace('h', 'H') for interval in intervals]
    resampled_dfs = {interval: [] for interval in intervals}
    
    for symbol, group_df in indexed_final_df.groupby('symbol'):
        for interval, interval_T in zip(intervals, intervals_timeframe):
            resampled_df = group_df.resample(interval_T).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'symbol': 'first',
                'datetime': 'last',
                'timestamp': 'last'
            }).dropna(subset=['open', 'close'])
            resampled_df['symbol'] = symbol
            resampled_dfs[interval].append(resampled_df)

    # Combine all resampled dataframes for each interval
    return {interval: pd.concat(resampled_list, ignore_index=True) for interval, resampled_list in resampled_dfs.items()}


# Function to save dataframes to Feather files partitioned by 'year_month'
def save_resampled_data(resampled_dfs, path_base):
    for interval, df_res in resampled_dfs.items():
        output_base_path = Path(path_base.replace("sample_data", f"df_res/{interval}"))
        os.makedirs(output_base_path, exist_ok=True)

        # Add 'year_month' column
        df_res['datetime'] = pd.to_datetime(df_res['datetime'])
        df_res['year_month'] = df_res['datetime'].dt.to_period('M')

        for year_month, partition_df in df_res.groupby('year_month'):
            year_month_str = year_month.strftime('%Y-%m')
            output_path = output_base_path / f"{interval}_{year_month_str}.ftr"
            partition_df.drop(columns=['year_month'], inplace=True)
            partition_df.reset_index(drop=True, inplace=True)
            partition_df.to_feather(output_path)
            print(f"Saved partition for {year_month_str} to {output_path}")


if __name__ == "__main__":

    
    # Main processing routine
    def main(path_base, start_date, end_date):
        mmdd_list = pd.date_range(start=start_date, end=end_date).strftime('%m%d').tolist()
        df_list = [load_and_process_csvs(path_base, mmdd) for mmdd in mmdd_list]
        final_df = pd.concat([df for df in df_list if df is not None], ignore_index=True)

        # Set 'datetime' as the index
        indexed_final_df = final_df.set_index('datetime', inplace=False, drop=False)

        # Define intervals and resample data
        intervals = ['15m', '30m', '1h', '2h', '4h']
        resampled_dfs = resample_data(indexed_final_df, intervals)

        # Save resampled dataframes to Feather files
        save_resampled_data(resampled_dfs, path_base)

    # Run main process
    path_base = r'D:\Project\SystemTrading\Project\MTLT\sample_data'
    start_date = '2024-04-01'
    end_date = '2024-12-10'
    main(path_base, start_date, end_date)

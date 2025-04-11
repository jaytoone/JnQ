import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import time
import traceback
from pathlib import Path
from joblib import Parallel, delayed


# Add project paths for importing necessary functions
path_project = "../"
path_funcs = "./include"
sys.path.append(path_project)
sys.path.append(path_funcs)

from idep import *  # Ensure all necessary functions are imported
from config import *


def load_df_res(interval, start_year_month, end_year_month, base_directory):
    """
    Load df_res data from Feather files for the specified symbol, interval, and date range.
    """
    start_date = pd.to_datetime(start_year_month, format='%Y-%m')
    end_date = pd.to_datetime(end_year_month, format='%Y-%m')
    
    data_dir = Path(base_directory) / interval
    dataframes = []
    
    for file in data_dir.glob(f"{interval}_*.ftr"):
        year_month_str = file.stem.split('_')[-1]
        
        if '-' in year_month_str and len(year_month_str) == 7:
            try:
                file_date = pd.to_datetime(year_month_str, format='%Y-%m')
                
                if start_date <= file_date <= end_date:
                    df = pd.read_feather(file)
                    if not df.empty:
                        dataframes.append(df)
                        print(f"Loaded and filtered {file.name}")
            except ValueError:
                continue
    
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        print("All relevant files have been loaded, filtered, and concatenated.")
        return combined_df
    else:
        print("No files found within the specified date range.")
        return pd.DataFrame()

def process_symbol(interval, configs, partition_start, partition_end, base_directory, mode='FULL'):
    """
    Process symbols for a given interval using the provided configurations.
    """
    try:
        df_res = load_df_res(interval, partition_start, partition_end, base_directory)
        
        if df_res.empty:
            print(f"No data found for any symbol at interval {interval}")
            return None
        
        unique_symbols = df_res['symbol'].unique()

        def process_symbol_for_config(symbol):
            try:
                df_res_symbol = df_res[df_res['symbol'] == symbol]
                results = []

                for config in configs:
                    try:
                        table_name = get_table_name(config)
                        print(f"[{time.strftime('%H:%M:%S')}] Starting processing for {interval}, {symbol} with config {table_name}")

                        df_res_local, point_bool_long, point_bool_short = get_point_bool(
                            df_res_symbol.copy(),
                            config["point_mode"],
                            config["point_indicator"],
                            config["point_value"],
                            interval
                        )

                        if config["zone_indicator"] == 'NULL':
                            zone_bool_long = point_bool_long
                            zone_bool_short = point_bool_short
                        else:
                            df_res_local, zone_bool_long, zone_bool_short = get_zone_bool(
                                df_res_local,
                                config["zone_indicator"],
                                config["zone_value"],
                                interval
                            )

                        point_index_long = np.argwhere(zone_bool_long & point_bool_long).ravel()
                        point_index_short = np.argwhere(zone_bool_short & point_bool_short).ravel()

                        table_trade_result = get_trade_result(
                            df_res_local,
                            symbol,
                            point_index_short,
                            point_index_long,
                            config['priceBox_indicator'],
                            config['priceBox_value'],
                            interval
                        )

                        if mode != 'FULL':
                            table_trade_result = table_trade_result[table_trade_result.status != 'DataEnd'].drop(columns=['is_valid'])

                        table_trade_result['priceBox_value'] = config["priceBox_value"]
                        table_trade_result['point_value'] = config["point_value"]
                        table_trade_result['zone_value'] = config["zone_value"]
                        table_trade_result['interval'] = interval

                        print(f"[{time.strftime('%H:%M:%S')}] Finished processing for {interval}, {symbol} with config {table_name}")

                        results.append((table_name, table_trade_result))

                    except Exception as e:
                        print(f"Error processing config for {symbol} at interval {interval}: {e}")
                        traceback.print_exc()
                        continue

                return results

            except Exception as e:
                print(f"Error processing {symbol} in interval {interval}: {e}")
                traceback.print_exc()
                return None

        all_results = Parallel(n_jobs=os.cpu_count() // 2)(delayed(process_symbol_for_config)(symbol) for symbol in unique_symbols)

        return [result for result_list in all_results if result_list for result in result_list if result is not None]

    except Exception as e:
        print(f"Error processing symbols in interval {interval}: {e}")
        traceback.print_exc()
        return None

def process_symbol_for_interval_and_symbol(interval, configs, partition_start, partition_end, base_directory):
    """
    Call process_symbol for each interval and return the results.
    """
    try:
        table_results = process_symbol(interval, configs, partition_start, partition_end, base_directory, mode='SIMPLE')
        return (interval, table_results)
    except Exception as e:
        print(f"Failed to process interval {interval}: {e}")
        traceback.print_exc()
        return None

def process_and_save_results(base_directory, table_name, results):
    """Process, merge, and save results for a given table."""
    combined_result = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    
    if combined_result.empty:
        print(f"No data for table {table_name}. Skipping...")
        return

    combined_result['datetime_entry'] = pd.to_datetime(combined_result['timestamp_entry'], unit='s', errors='coerce')
    combined_result['year_month'] = combined_result['datetime_entry'].dt.strftime('%Y-%m')

    path_save_trade_result = Path(base_directory) / table_name
    path_save_trade_result.mkdir(parents=True, exist_ok=True)

    unique_year_months = combined_result['year_month'].unique()
    for year_month in unique_year_months:
        path_file = path_save_trade_result / f"{table_name}_{year_month}.ftr"

        if path_file.exists():
            existing_result = pd.read_feather(path_file)
            existing_result['datetime_entry'] = pd.to_datetime(existing_result['timestamp_entry'], unit='s', errors='coerce')
            existing_result['year_month'] = existing_result['datetime_entry'].dt.strftime('%Y-%m')
            print(f"Existing file {path_file} loaded for merging.")
        else:
            existing_result = pd.DataFrame()

        filtered_result = combined_result[combined_result['year_month'] == year_month].reset_index(drop=True)
        merged_result = pd.concat([existing_result, filtered_result], ignore_index=True)
        merged_result = merged_result.drop_duplicates(subset=["symbol", "timestamp_entry", "priceBox_value", "point_value", "zone_value", "interval"])

        merged_result.to_feather(path_file)
        print(f"Table {table_name} for {year_month} has {len(merged_result)} rows.")
        print(f"{path_file} saved.")
        
    
def main(base_directory, results_parallel):
    """Process and save results for each table in results_parallel."""
    for table_name, results in results_parallel.items():
        process_and_save_results(base_directory, table_name, results)


if __name__ == "__main__":
    
    # Configuration
    base_directory = r'D:\Project\SystemTrading\Project\MTLT\df_res'
    base_directory = r'D:\Project\SystemTrading\Project\JnQ\anal\df_res'

    intervals = ['15m', '30m', '1h', '2h', '4h']
    partition_start = "2024-04" # end - 3 months
    partition_end = "2024-12"

    configs = generate_configs(base_config, price_range=(20, 60))
    

    # Final results storage
    results_parallel = {}

    with ThreadPoolExecutor(max_workers=max(1, os.cpu_count() // 2)) as executor:
        futures = {
            executor.submit(process_symbol_for_interval_and_symbol, interval, configs, partition_start, partition_end, base_directory): interval
            for interval in intervals
        }

        for future in as_completed(futures):
            interval = futures[future]
            try:
                result = future.result()
                if result and result[1]:
                    table_results = result[1]
                    for table_name, trade_result in table_results:
                        if table_name not in results_parallel:
                            results_parallel[table_name] = []
                        results_parallel[table_name].append(trade_result)
            except Exception as e:
                print(f"Error processing interval {interval}: {e}")
                traceback.print_exc()

    main(base_directory, results_parallel)

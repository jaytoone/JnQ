import os
import sys
import pandas as pd
import numpy as np
import time
from pathlib import Path
from joblib import Parallel, delayed
import copy


# Add project paths for importing necessary functions
path_project = "../"
path_funcs = "./include"
sys.path.append(path_project)
sys.path.append(path_funcs)

from idep import *  # Ensure all necessary functions are imported
from config import *


# Global : Generate other variables for processing
unit_RRratio_adj_fee = np.arange(0, 2, 0.1)
intervals = ['15m', '30m', '1h', '2h', '4h']
positions = ['LONG', 'SHORT']
RRratio_adj_fee_categories = pd.cut(
    unit_RRratio_adj_fee, bins=np.append(unit_RRratio_adj_fee, unit_RRratio_adj_fee[-1] + 0.1), right=True
).astype(str)
range_winRatio = np.arange(0.4, 1.0, 0.01)



# Function to process each position and RRratio category combination
def process_position_and_category(config, table_trade_result, position, RRratio_adj_fee_category, range_winRatio, symbol_whitelist):
    config['position'] = position
    config['RRratio_adj_fee_category'] = RRratio_adj_fee_category

    # Filter by position and RR ratio category
    table_trade_result_anchor = table_trade_result[
        (table_trade_result.position == config['position']) & 
        (table_trade_result.RRratio_adj_fee_category == config['RRratio_adj_fee_category'])
    ]

    # Filter train data and create pivot table
    table_trade_result_anchor_train = table_trade_result_anchor[table_trade_result_anchor.dataType == 'TRAIN']
    table_trade_result_anchor_train_pivot = pivot_table(table_trade_result_anchor_train)

    if table_trade_result_anchor_train_pivot is None:
        return []

    local_results = []
    for threshold_winRatio in range_winRatio:
        config['threshold_winRatio'] = threshold_winRatio
        
        msg = (
            f"{config['priceBox_indicator']}, {config['priceBox_value']}, "
            f"{config['point_mode']}, {config['point_indicator']}, "
            f"{config['point_value']}, {config['zone_indicator']}, {config['zone_value']}, "
            f"{config['interval']}, {config['partition_start']}, {position}, {RRratio_adj_fee_category}, {threshold_winRatio:.2f}"
        )
        print(msg, end='\r')

        symbol_extracted, symbol_extracted_len, symbol_extracted_len_pct = get_symbol_extracted(
            table_trade_result_anchor_train_pivot,
            config['threshold_winRatio'],
            config["threshold_frequency"],
            symbol_whitelist
        )

        table_trade_result_anchor_train_val = table_trade_result_anchor[
            table_trade_result_anchor['dataType'].isin(['TRAIN', 'VALID'])
        ]

        table_trade_result_agg = get_trade_result_agg(
            table_trade_result_anchor_train_val,
            symbol_extracted,
            config["mode_position"]
        )

        frequencyTotal = len(table_trade_result_agg)
        if frequencyTotal < config["threshold_frequencyTotal"] or table_trade_result_agg.empty:
            continue

        output = get_output_row(
            table_trade_result_agg, 
            symbol_extracted,
            symbol_extracted_len,  
            symbol_extracted_len_pct, 
            config
        )

        if output:
            row, title, profit_pct_cum, assets_over_time = output
            r2_val = row[-1]
            max_drawdown_scaled_val = row[-6]

            if r2_val > 0.90 and max_drawdown_scaled_val > -0.15:
                table_trade_result_agg = get_trade_result_agg(
                    table_trade_result_anchor,
                    symbol_extracted,
                    config["mode_position"]
                )
                
                output = get_output_row(
                    table_trade_result_agg, 
                    symbol_extracted,
                    symbol_extracted_len,  
                    symbol_extracted_len_pct, 
                    config
                )
                
                if output:
                    row, title, profit_pct_cum, assets_over_time = output
                    r2_test = row[-1]
                    r2_diff = r2_test - r2_val
                    local_results.append(row + [r2_val, r2_diff, config["partition_start"], config["partition_end"]])
                    break  # Prevent duplicates
                
    return local_results

# Main process function using Parallel processing
def process_partition(config, table_trade_result, positions, RRratio_adj_fee_categories, range_winRatio, leverage_brackets={}, symbol_whitelist=None):
    start_time = time.time()
    table_trade_result = set_quantity(
        table_trade_result,
        leverage_brackets,
        config["target_loss"],
        config["target_loss_pct"],
        config["target_leverage"],
        config["fee_market"],
        config["fee_market"],
        config["unit_RRratio_adj_fee"],
        config["leverage_rejection"],
    )
    print(f"Elapsed time for set_quantity: {time.time() - start_time:.4f}s")

    start_time = time.time()
    table_trade_result = add_datatype(table_trade_result, config["train_ratio"])
    print(f"Elapsed time for add_datatype: {time.time() - start_time:.4f}s")

    # Run parallel processing for each position and RRratio_adj_fee_category
    results = Parallel(n_jobs=os.cpu_count() // 2)(
        delayed(process_position_and_category)(config, table_trade_result, position, RRratio_adj_fee_category, range_winRatio, symbol_whitelist)
        for position in positions for RRratio_adj_fee_category in RRratio_adj_fee_categories
    )

    local_results = [res for sublist in results if sublist for res in sublist]
    return local_results



# Function to load Feather files within a date range
def load_feather_files(table_directory, table_name, partition_start, partition_end):
    feather_dataframes = []
    missing_dates = []
    partition_dates = pd.date_range(start=partition_start, end=partition_end, freq='MS').strftime("%Y-%m").tolist()

    for partition_date in partition_dates:
        path_file = table_directory / f"{table_name}_{partition_date}.ftr"
        if path_file.exists():
            df = pd.read_feather(path_file)
            feather_dataframes.append(df)
            print(f"Feather file {path_file} loaded.")
        else:
            print(f"Feather file {path_file} not found. Skipping...")
            missing_dates.append(partition_date)

    return feather_dataframes, missing_dates


# Function to asynchronously process a configuration for a specific interval and partition
def process_config_interval_partition(config_temp, config, table_trade_result_from_feather):
    # Update config_temp with values from the provided config
    config_temp['priceBox_indicator'] = config['priceBox_indicator']
    config_temp['priceBox_value'] = float(config['priceBox_value'])  # Type cast
    config_temp['point_mode'] = config['point_mode']
    config_temp['point_indicator'] = config['point_indicator']
    config_temp['point_value'] = float(config['point_value'])  # Type cast
    config_temp['zone_indicator'] = config['zone_indicator']
    config_temp['zone_value'] = float(config['zone_value'])  # Type cast

    # Filter data based on the given configuration
    filtered_data = table_trade_result_from_feather[
        (table_trade_result_from_feather['priceBox_value'] == config_temp['priceBox_value']) &
        (table_trade_result_from_feather['point_value'] == config_temp['point_value']) &
        (table_trade_result_from_feather['zone_value'] == config_temp['zone_value']) &
        (table_trade_result_from_feather['interval'] == config_temp['interval'])
    ]

    # Process the data only if it exists
    if not filtered_data.empty:
        # Pass a copy of config_temp to ensure safety within the process_partition function
        return process_partition(copy.deepcopy(config_temp), filtered_data, positions, RRratio_adj_fee_categories, range_winRatio)
    return []  # Return an empty list if no data is found



# Function to process Feather files and generate a combined DataFrame
def process_partition_data(table_directory, table_name, partition_start, partition_end):
    feather_dataframes, missing_dates = load_feather_files(table_directory, table_name, partition_start, partition_end)

    if missing_dates:
        print(f"Missing Feather files for {table_name} in dates: {', '.join(missing_dates)}. This may affect the final result.")
        return None

    if feather_dataframes:
        table_trade_result_from_feather = pd.concat(feather_dataframes, ignore_index=True)
        print(f"Concatenated DataFrame for {table_name} created with {len(table_trade_result_from_feather)} rows.")
        return table_trade_result_from_feather
    else:
        print(f"No Feather files found for {table_name} in the range from {partition_start} to {partition_end}.")
        return None


# Function to process data for each interval and update results
def process_intervals(config_temp, config, table_trade_result_from_feather, result_parallel):
    for interval in intervals:
        config_temp['interval'] = interval
        local_results = process_config_interval_partition(copy.deepcopy(config_temp), config, table_trade_result_from_feather)
        
        if local_results:
            result_parallel.extend(local_results)


# Main function to manage the workflow for each table configuration
def main(base_directory, config_temp, configs, partition_start, partition_end):
    partition_start_dates = [date.strftime("%Y-%m") for date in pd.date_range(start=partition_start, end=partition_end, freq='MS')]
    
    # Print initial log message
    print("Processing started with log level INFO.")
    result_parallel = []

    for config in configs:
        table_name = get_table_name(config)
        table_directory = Path(base_directory) / table_name
        
        for partition_start in partition_start_dates:
            config_temp['partition_start'] = partition_start
            config_temp['partition_end'] = partition_end
            
            # Process the data for the partition
            table_trade_result_from_feather = process_partition_data(table_directory, table_name, partition_start, partition_end)
            
            if table_trade_result_from_feather is not None:
                process_intervals(config_temp, config, table_trade_result_from_feather, result_parallel)

    # Print completion log message
    print("Processing completed for the specified partition_end and all partition_start_dates.")
    return result_parallel




if __name__ == "__main__":
    
    # Configuration
    base_directory = r'D:\Project\SystemTrading\Project\MTLT\table_trade_result'
    partition_start = "2024-04"
    partition_end = "2024-12"

    configs = generate_configs(base_config, price_range=(20, 60))

    # Generate output path and create directory
    path_dir_save_fig = create_directory_path(
        base_directory + "_anchor",
        config_temp
    )


    # Run the main function
    result_parallel = main(base_directory, config_temp, configs, partition_start, partition_end)
        
    table_trade_result_anchor =  pd.DataFrame(result_parallel, columns=columns) #.head()
    table_trade_result_anchor[columns_reorder].to_excel(os.path.join(path_dir_save_fig, "payload.xlsx"), index=0)


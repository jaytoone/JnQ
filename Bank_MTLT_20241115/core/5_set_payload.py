import sys
import pandas as pd
import time
import re
import ast
from pathlib import Path

# Add project paths for importing necessary functions
path_project = "../"
path_funcs = "./include"
sys.path.append(path_project)
sys.path.append(path_funcs)

from idep import *  # Ensure all necessary functions are imported
from config import *



def parse_payload_to_config(payload, config):
    """
    Parse payload to update configuration dictionary.
    
    v0.1.6:
    - Handle case when symbol_extracted is not present.
    - Modify to use groups with optional handling.
    """
    # Split main part and list part from payload
    match = re.match(r'^(.*?)(\[[^\]]*\])?$', payload)
    
    if not match:
        raise ValueError("Invalid payload format")
    
    main_part = match.group(1)
    list_part = match.group(2)
    
    # Split main_part to extract values
    main_parts = re.split(r'[\t_]', main_part)
    
    config_temp = config.copy()
    
    # Assign values to the config
    config_temp["priceBox_indicator"] = main_parts[0]
    config_temp["priceBox_value"] = float(main_parts[1])
    config_temp["point_mode"] = main_parts[2]
    config_temp["point_indicator"] = main_parts[3]
    config_temp["point_value"] = float(main_parts[4])
    config_temp["zone_indicator"] = main_parts[5]
    config_temp["zone_value"] = float(main_parts[6])
    config_temp["interval"] = main_parts[7]
    config_temp["partition_start"] = main_parts[8]
    config_temp["partition_end"] = main_parts[9]
    config_temp["target_loss"] = int(main_parts[10])
    config_temp["target_loss_pct"] = int(main_parts[11])
    config_temp["target_leverage"] = int(main_parts[12])
    config_temp["position"] = main_parts[13]
    config_temp["RRratio_adj_fee_category"] = main_parts[14]
    config_temp["threshold_winRatio"] = float(main_parts[15])
    config_temp["threshold_frequency"] = float(main_parts[16])
    config_temp["threshold_frequencyTotal"] = float(main_parts[17])
    config_temp["mode_position"] = main_parts[18]
    
    # Convert list_part to a Python list or set to an empty list
    if list_part:
        config_temp["symbol_extracted"] = ast.literal_eval(list_part)
    else:
        config_temp["symbol_extracted"] = []  # Default to an empty list
    
    return config_temp


def load_partition(config, start_year_month, end_year_month, base_directory):
    # 필터링 조건 설정
    priceBox_value = config.get('priceBox_value')
    point_value = config.get('point_value')
    zone_value = config.get('zone_value')
    interval = config.get('interval')
    
    # 테이블 이름 설정
    table_name = get_table_name(config)
    
    # 저장된 Feather 파일이 있는 디렉토리 경로 설정
    data_dir = Path(base_directory) / table_name
    
    # 시작 및 끝 연월을 datetime으로 변환하여 비교할 수 있게 함
    start_date = pd.to_datetime(start_year_month, format='%Y-%m')
    end_date = pd.to_datetime(end_year_month, format='%Y-%m')
    
    # 통합할 데이터프레임 리스트
    dataframes = []
    
    # 디렉토리 내의 모든 Feather 파일을 확인
    for file in data_dir.glob(f"{table_name}_*.ftr"):
        # 파일 이름에서 연-월 정보 추출, 'YYYY-MM' 형식에만 매칭
        year_month_str = file.stem.split('_')[-1]
        
        # 'YYYY-MM' 형식의 파일만 처리
        if '-' in year_month_str and len(year_month_str) == 7:
            try:
                file_date = pd.to_datetime(year_month_str, format='%Y-%m')
                
                # 시작 연월과 끝 연월 사이에 있는 파일만 로드
                if start_date <= file_date <= end_date:
                    df = pd.read_feather(file)
                    
                    # 조건에 맞는 행만 필터링
                    filtered_df = df[
                        (df["priceBox_value"] == priceBox_value) &
                        (df["point_value"] == point_value) &
                        (df["zone_value"] == zone_value) &
                        (df["interval"] == interval)
                    ]
                    
                    # 필터링된 데이터프레임을 리스트에 추가
                    dataframes.append(filtered_df)
                    # print(f"Loaded and filtered {file.name}")
            except ValueError:
                # 파일 이름 형식이 잘못된 경우 건너뜁니다.
                continue
    
    # 모든 파일을 하나의 데이터프레임으로 통합
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        print("All relevant files have been loaded, filtered, and concatenated.")
        return combined_df
    else:
        print("No files found within the specified date range.")
        return pd.DataFrame()  # 빈 데이터프레임 반환
    


def process_trade_result(config_temp, base_directory, path_dir_save_fig, leverage_limits, symbol_whitelist, show_figure=False, save_figure=False):
    """
    v0.3.4
        - use load_partition
        - log year_month.
    
    20241112 1724.
    """
    
    # Initial setup
    print("------------------------------------------------")
    start_time = time.time()
    
    table_trade_result = load_partition(config_temp, config_temp['partition_start'], config_temp['partition_end'], base_directory)
    print(f"table_trade_result.year_month.unique() : {table_trade_result.year_month.unique()}")

    print("IDEP : elapsed time, load table_trade_result : {:.4f}s".format(time.time() - start_time))
    print("------------------------------------------------")

    if len(table_trade_result):
        print("------------------------------------------------")
        start_time = time.time()
        
        table_trade_result = set_quantity(
            table_trade_result,
            leverage_limits,
            config_temp["target_loss"],
            config_temp["target_loss_pct"],
            config_temp["target_leverage"],
            config_temp["fee_market"],
            config_temp["fee_market"],
            config_temp["unit_RRratio_adj_fee"],
            config_temp["leverage_rejection"],
        )
        
        print("IDEP : elapsed time, set_quantity : {:.4f}s".format(time.time() - start_time))
        print("------------------------------------------------")
        
        print("------------------------------------------------")
        start_time = time.time()
        
        table_trade_result = add_datatype(table_trade_result, config_temp["train_ratio"])
        
        print("IDEP : elapsed time, add_datatype : {:.4f}s".format(time.time() - start_time))
        print("------------------------------------------------")
        
        print("------------------------------------------------")
        start_time = time.time()
        
        table_trade_result_anchor = table_trade_result[(table_trade_result.position == config_temp["position"]) & (table_trade_result.RRratio_adj_fee_category == config_temp["RRratio_adj_fee_category"])].copy()
        
        print("IDEP : elapsed time, get table_trade_result_anchor : {:.4f}s".format(time.time() - start_time))
        print("------------------------------------------------")
        
        if table_trade_result_anchor is not None:
            symbol_extracted_len = len(config_temp["symbol_extracted"])
            symbol_extracted_len_pct = (symbol_extracted_len / table_trade_result_anchor.symbol.nunique()) * 100
            
            print("------------------------------------------------")
            start_time = time.time()
            
            # adj. on full dataTypes.
            table_trade_result_agg = get_trade_result_agg(
                table_trade_result_anchor,
                config_temp["symbol_extracted"],
                config_temp["mode_position"]
            )
            
            print("IDEP : elapsed time, get_table_trade_result_agg : {:.4f}s".format(time.time() - start_time))
            print("------------------------------------------------")
            
            frequencyTotal = len(table_trade_result_agg)
            if frequencyTotal > 0:
                print("------------------------------------------------")
                start_time = time.time()
                
                output = get_output_row(
                    table_trade_result_agg, 
                    config_temp["symbol_extracted"],
                    symbol_extracted_len,  
                    symbol_extracted_len_pct, 
                    config_temp,
                )
                
                print("IDEP : elapsed time, get_output_row: {:.4f}s".format(time.time() - start_time))
                print("------------------------------------------------")
                
                if output:
                    print("------------------------------------------------")
                    start_time = time.time()
                    
                    row, title, profit_pct_cum, assets_over_time = output
                    
                    set_figure(
                        table_trade_result_agg, 
                        profit_pct_cum, 
                        assets_over_time,
                        config_temp, 
                        path_dir_save_fig, 
                        title, 
                        show_figure, 
                        save_figure
                    )
                    
                    print("IDEP : elapsed time, set_figure: {:.4f}s".format(time.time() - start_time))
                    print("------------------------------------------------")
                
                return table_trade_result_agg




if __name__ == "__main__":
        
    show_figure = 1
    save_figure = 0
    
    # Configuration
    base_directory = r'D:\Project\SystemTrading\Project\MTLT\table_trade_result'

    configs = generate_configs(base_config, price_range=(20, 60))

    # Generate output path and create directory
    path_dir_save_fig = create_directory_path(
        base_directory + "_anchor",
        config_temp
    )    
    
    payload_df_test = pd.read_excel(Path(path_dir_save_fig) / f"payload_condition.xlsx")
    payload_df_test = payload_df_test.sort_values(by='profit_pct_monthly_min', ascending=False).reset_index(drop=True)

    # Convert each row into a string format and create a list called payloads
    payloads = payload_df_test.apply(
        lambda row: '_'.join(
            'NULL' if pd.isna(x) else (f'{x:.2f}' if isinstance(x, float) else str(x))
            for x in row['priceBox_indicator':'symbol_extracted']
        ),
        axis=1
    ).tolist()
    
    # print(payloads)

    for payload_idx, payload in enumerate(payloads):
        
        print(f"payload {payload_idx}: {payload}", end='\r')
        
        parsed_config = parse_payload_to_config(payload, config_temp)    
        # parsed_config["partition_end"] = partition_end
        print(f"parsed_config: {parsed_config}")
        
        result = process_trade_result(parsed_config, base_directory, path_dir_save_fig, {}, None, show_figure=show_figure, save_figure=save_figure)
        
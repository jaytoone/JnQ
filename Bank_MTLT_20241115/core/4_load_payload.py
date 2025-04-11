import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


# Add project paths for importing necessary functions
path_project = "../"
path_funcs = "./include"
sys.path.append(path_project)
sys.path.append(path_funcs)

from idep import *  # Ensure all necessary functions are imported
from config import *


# Step 1: Configuration 설정 함수
def set_configurations():
    return {
        'threshold_r2_val': 0.97,
        'threshold_frequencyTotal': 10,
        'threshold_r2_diff': -0.02,
        'threshold_consistency': 0.7,
        'threshold_consistency_count': 4,
    }

# Step 2: 데이터 필터링 함수
def filter_payload_data(payload_df, config):
    filtered_df = payload_df[
        (payload_df['r2_val'] > config['threshold_r2_val'])
        & (payload_df['frequencyTotal'] > config['threshold_frequencyTotal'])
    ]
    filtered_df['r2_valid'] = filtered_df['r2_diff'] > config['threshold_r2_diff']
    return filtered_df

# Step 3: 피벗 테이블 생성 함수
def create_pivot_table(filtered_df):
    pivot_df = filtered_df.pivot_table(
        index=['position', 'RRratio_adj_fee_category', 'partition_start'],
        columns=['r2_valid'],
        values='r2_val',
        aggfunc='count',
        fill_value=0
    )
    pivot_df['count'] = pivot_df[True] + pivot_df[False]
    pivot_df['consistency'] = pivot_df[True] / pivot_df['count']
    pivot_df['consistency'] = pivot_df['consistency'].fillna(0)
    return pivot_df

# Step 4: 일관성 기준으로 필터링하는 함수
def filter_by_consistency(pivot_df, config):
    return pivot_df[pivot_df['consistency'] > config['threshold_consistency']].reset_index()

# Step 5: 멀티 인덱스 집계 및 조건 필터링 함수
def aggregate_and_filter(pivot_df_reset, config):
    pivot_agg = pivot_df_reset.pivot_table(
        index=['position', 'RRratio_adj_fee_category'],
        values=['count', 'consistency'],
        aggfunc={'count': 'sum', 'consistency': 'mean'}
    )
    pivot_agg = pivot_agg[['count', 'consistency']].sort_values(by='count', ascending=False)
    positions_rrratios = pivot_agg[pivot_agg['count'] >= config['threshold_consistency_count']].index
    
    filtered_df = pivot_df_reset[
        pivot_df_reset.set_index(['position', 'RRratio_adj_fee_category']).index.isin(positions_rrratios)
    ]
    return filtered_df[filtered_df['RRratio_adj_fee_category'].apply(lambda x: float(x.split(',')[0].strip('()')) >= 1.0)]

# Step 6: 테스트 데이터셋 준비 함수
def prepare_test_dataset(filtered_df, filtered_payload_df):
    test_list = []
    for _, row in filtered_df.iterrows():
        position, partition_start, RRratio_adj_fee_category = row[['position', 'partition_start', 'RRratio_adj_fee_category']]
        test_df = filtered_payload_df[
            (filtered_payload_df['position'] == position) & 
            (filtered_payload_df['partition_start'] == partition_start) & 
            (filtered_payload_df['RRratio_adj_fee_category'] == RRratio_adj_fee_category) &
            (filtered_payload_df['r2_valid'] == True)
        ]
        test_list.append(test_df)
    return pd.concat(test_list).sort_values(by='profit_pct_monthly_min', ascending=False).reset_index(drop=True)


def plot_winratio_distribution_with_counts(data, x_var, y_var='consistency', rotation=45, box_alpha=0.5, median_linewidth=3, save_path=None):
    """
    v0.2.1
        - add save mode 20241116.
    """
    
    fig, ax = plt.subplots(2, 1, figsize=(6, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    # Boxplot for Consistency distribution
    sns.boxplot(data=data, x=x_var, y=y_var, ax=ax[0], boxprops=dict(alpha=box_alpha))
    ax[0].set_title(f"Consistency Distribution by {x_var}")
    ax[0].tick_params(axis='x', rotation=rotation)
    
    # Adjust the width of the median line
    for line in ax[0].lines:
        if line.get_linestyle() == '-':  # Only adjust solid lines
            line.set_linewidth(median_linewidth)
    
    # Countplot for showing the number of records per category
    sns.countplot(data=data, x=x_var, ax=ax[1])
    ax[1].set_title(f"Count of Records by {x_var}")
    ax[1].tick_params(axis='x', rotation=rotation)
    
    plt.tight_layout()

    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()
    

if __name__ == "__main__":
    
    # Configuration
    base_directory = r'D:\Project\SystemTrading\Project\MTLT\table_trade_result'

    configs = generate_configs(base_config, price_range=(20, 60))
    
    # Generate output path and create directory
    path_dir_save_fig = create_directory_path(
        base_directory + "_anchor",
        config_temp
    )

    path_payload = Path(rf"{path_dir_save_fig}\payload.xlsx")
    payload_df = pd.read_excel(path_payload).fillna('NULL')


    # Main Code Execution
    config = set_configurations()
    payload_df_thresh = filter_payload_data(payload_df, config)
    pivot_df = create_pivot_table(payload_df_thresh)
    
    # for idx, x_val in enumerate(pivot_df.index.names):
    #     plot_winratio_distribution_with_counts(pivot_df, x_val, save_path=Path(path_dir_save_fig) / f"{x_val}.png")
    
    pivot_df_reset = filter_by_consistency(pivot_df, config)
    final_filtered_df = aggregate_and_filter(pivot_df_reset, config)
    payload_df_test = prepare_test_dataset(final_filtered_df, payload_df_thresh)

    # Save the result as a CSV file with '_condition' added to the filename
    path_payload_condition = Path(path_dir_save_fig) / f"payload_condition.xlsx"
    payload_df_test.to_excel(path_payload_condition, index=False)
    
    # Display result
    print(payload_df_test)
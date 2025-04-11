import numpy as np


partition_start = "2024-04"
partition_end = "2024-12"

# Base configuration template
base_config = {
    "priceBox_indicator": "DC",
    "priceBox_value": 20,
    "point_mode": "CROSS",
    "point_indicator": "SROC",
    "point_value": 13,
    "zone_indicator": "NULL",
    "zone_value": -1,
}

# Config for other template settings
config_temp = base_config.copy()
config_temp.update({
    "leverage_rejection": True,
    "train_ratio": 0.7,
    "fee_limit": 0.0002,
    "fee_market": 0.0005,
    "unit_RRratio_adj_fee": np.arange(0, 2, 0.1),
    "mode_profit": 'SIMPLE',
    "target_loss": 15,
    "target_loss_pct": 100,
    "target_leverage": 1,
    "threshold_frequency": 0,
    "threshold_frequencyTotal": 0,
    "mode_position": 'MULTIPLE',
})


columns = [
    'priceBox_indicator',
    'priceBox_value',
    'point_mode',
    'point_indicator',
    'point_value',
    'zone_indicator',
    'zone_value',
    
    'interval',
    'target_loss',
    'target_loss_pct', 
    'target_leverage',
    'position',
    'RRratio_adj_fee_category',
    'threshold_winRatio',
    'threshold_frequency',
    'threshold_frequencyTotal',
    'mode_position',
    
    'symbol_extracted',
    'symbol_extracted_len',
    'symbol_extracted_len_pct',
    
    'frequencyTotal',
    'frequencyMean',
    'winRatio',
    
    'assets_min_required_const',
    'assets_max_drawdown_const',
    'profit_pct_final',
    'profit_pct_daily', 
    'profit_pct_monthly',
    'profit_pct_yearly',
    'profit_pct_final_min',
    'profit_pct_daily_min', 
    'profit_pct_monthly_min',
    'profit_pct_yearly_min',
    'profit_pct_final_max',
    'profit_pct_daily_max', 
    'profit_pct_monthly_max',
    'profit_pct_yearly_max',
    
    'mean_return',
    'std_return',
    'max_drawdown',
    'max_drawdown_scaled',
    'sharpe_ratio',
    'sortino_ratio',
    'profit_factor',
    'mse',
    'r2',
    
    'r2_val',
    'r2_diff',
    'partition_start',
    'partition_end',
]

columns_reorder = [
    'r2',
    'r2_val',
    'r2_diff',
    'max_drawdown_scaled',
    
    'frequencyTotal',
    'frequencyMean',
    'symbol_extracted_len_pct',
    'profit_pct_monthly_min',
    'profit_pct_monthly_max',
    'winRatio',
    
    'priceBox_indicator',
    'priceBox_value',
    'point_mode',
    'point_indicator',
    'point_value',
    'zone_indicator',
    'zone_value',
    'interval',    
    'partition_start',
    'partition_end',
    
    'target_loss',
    'target_loss_pct', 
    'target_leverage',
    'position',
    'RRratio_adj_fee_category',
    'threshold_winRatio',
    'threshold_frequency',
    'threshold_frequencyTotal',    
    'mode_position',
    
    'symbol_extracted',
    'symbol_extracted_len',
    
    'assets_min_required_const',
    'assets_max_drawdown_const',    
    'profit_pct_final',
    'profit_pct_daily', 
    'profit_pct_monthly',
    'profit_pct_yearly',
    'profit_pct_final_min',
    'profit_pct_daily_min', 
    'profit_pct_yearly_min',
    'profit_pct_final_max',
    'profit_pct_daily_max', 
    'profit_pct_yearly_max',
    
    'mean_return',
    'std_return',
    'max_drawdown',
    'sharpe_ratio',
    'sortino_ratio',
    'profit_factor',
    'mse',
]


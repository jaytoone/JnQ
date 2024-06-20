import os, sys

path_project = r"D:\Project\SystemTrading\Project\JnQ"
path_funcs = r"D:\Project\SystemTrading\Project\JnQ\funcs"
path_package = r"D:\Project\SystemTrading\Project\JnQ\Bank"
sys.path.append(path_project)
sys.path.append(path_funcs)
sys.path.append(path_package)


from bank import * 
from bank_loop import * 



"""
v1.0
    replace to bank.concat_candlestick. 
    replace Projects --> Project

last confirmed at, 20240610 2126.
"""

bank = Bank(path_config=r"D:\Project\SystemTrading\Project\JnQ\Bank\config.json", 
            path_api=r"D:\Project\SystemTrading\Project\JnQ\keys\binance_mademerich.pkl", # 'r' preserve away 't, a' other vague letters.
            save_path_log=r"D:\Project\SystemTrading\Project\JnQ\Bank\logs\{}.log".format(datetime.now().strftime('%Y%m%d%H%M%S')),
            chat_id="5320962614",
            token="6206784409:AAE-xpIQELzgBfzbcFxhQE2Cjb6lZt6zn4s",
            receive_limit_ms=1000*3600)

bank.path_dir_df_res = r"D:\Project\SystemTrading\Project\JnQ\Bank\data\df_res"
bank.path_table_condition = r"D:\Project\SystemTrading\Project\JnQ\Bank\data\table\table_condition.xlsx"
bank.path_table_trade = r"D:\Project\SystemTrading\Project\JnQ\Bank\data\table\table_trade.pkl"
bank.path_table_log = r"D:\Project\SystemTrading\Project\JnQ\Bank\data\table\table_log.xlsx"



"""
v1.0
    add concat_candlestick. 

last confirmed at, 20240610 2120.
"""

# BankSetting
# bank.echo = echo
# bank.get_messenger_bot = get_messenger_bot
bank.push_msg = push_msg

bank.get_tables = get_tables
bank.set_tables = set_tables

# LoopMessenger
bank.get_tickers = get_tickers 


# LoopTableCondition
bank.get_streamer = get_streamer 
bank.concat_candlestick = concat_candlestick 
bank.get_df_new_by_streamer = get_df_new_by_streamer
bank.get_df_new = get_df_new

bank.get_df_mtf = get_df_mtf
bank.get_wave_info = get_wave_info

bank.set_price_and_open_signal = set_price_and_open_signal
bank.set_price_box = set_price_box
bank.set_price = set_price
bank.get_open_res = get_open_res
bank.adj_wave_point = adj_wave_point
bank.adj_wave_update_hl_rejection = adj_wave_update_hl_rejection
bank.validate_price = validate_price
bank.get_reward_risk_ratio = get_reward_risk_ratio

bank.get_side_open = get_side_open
bank.entry_point_location = entry_point_location
bank.entry_point2_location = entry_point2_location


# InitTableTrade
bank.get_balance_available = get_balance_available
bank.get_balance_info = get_balance_info
bank.get_price = get_price
bank.get_price_liquidation = get_price_liquidation
bank.get_price_expiration = get_price_expiration
bank.get_precision = get_precision

bank.get_leverage_limit = get_leverage_limit
bank.get_leverage = get_leverage
bank.set_leverage = set_leverage
bank.set_position_mode = set_position_mode
bank.set_margin_type = set_margin_type


# LoopTableTrade
bank.get_order_info = get_order_info
bank.get_price_realtime = get_price_realtime
bank.get_quantity_unexecuted = get_quantity_unexecuted
bank.check_stop_loss = check_stop_loss
bank.check_stop_loss_onbarclose = check_stop_loss_onbarclose
bank.check_stop_loss_by_signal = check_stop_loss_by_signal

bank.order_limit = order_limit
bank.order_market = order_market
bank.order_cancel = order_cancel

bank.get_income_info = get_income_info



if __name__=="__main__":


    """
    v2.0
        trade without messenger.
    v2.1
        consider, timestampLastUsed
        
    last confirmed at, 20240613 1347.
    """
    
    # parameter : term.
    term_loop_messenger = 1
    
    time_start = time.time()
    time_start_messenger = time_start
    
    # main    
    bank.get_tables(bank)
    
    while 1:
    
        time_real = time.time()
        if time_real - time_start_messenger > term_loop_messenger:
            # save backup file before run pipleine.
            bank.sys_log.debug("------------------------------------------------")
            start_time = time.time()
            bank.table_log.reset_index(drop=True, inplace=True) # for index row by 'at'    
            bank.set_tables(bank, mode='LOG')
            bank.sys_log.debug("Main : elasped time, set_tables : {:.2f}s".format(time.time() - start_time))
            bank.sys_log.debug("------------------------------------------------")
    
            time_start_messenger = time_real
    
        loop_table_condition(bank)
        loop_table_trade(bank)
        
        clear_output(wait=True)
        time.sleep(0.2) # time.sleep for sleep processor  


        

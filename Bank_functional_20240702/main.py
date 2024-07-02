import os, sys

path_project = "../"
path_funcs = "../funcs"
sys.path.append(path_project)
sys.path.append(path_funcs)


from bank_loop import * 



"""
v1.0
    replace to bank.concat_candlestick. 
    replace Projects --> Project
v2.0
    remove function assign phase.
        use relative path.
            considering dynamic path_dir_pkg.

last confirmed at, 20240702 1301.
"""


bank = Bank(path_config="./config.json", 
            path_api="../keys/binance_mademerich.pkl", # 'r' preserve away 't, a' other vague letters.
            path_dir_df_res =  "./data/df_res",
            path_table_account = "./data/table/table_account.xlsx",            
            path_table_condition =  "./data/table/table_condition.xlsx",
            path_table_trade =  "./data/table/table_trade.xlsx",
            path_table_log =  "./data/table/table_log.xlsx",
            path_save_log="./logs/{}.log".format(datetime.now().strftime('%Y%m%d%H%M%S')),
            chat_id="5320962614",
            token="7156517117:AAF_Qsz3pPTHSHWY-ZKb8LSKP6RyHSMTupo",
            receive_limit_ms=1000*3600)


if __name__=="__main__":
    
    """
    v2.0
        trade without messenger.
    v2.1
        consider, timestampLastUsed
    v2.2
        remove get_tables.
        
    last confirmed at, 20240702 1308.
    """
    
    while 1:
        # loop_table_condition(bank) # debugging.
        loop_table_trade(bank)
        
        clear_output(wait=True) # if run in ipynb.
        # time.sleep(0.2) # time.sleep for sleep processor  


        

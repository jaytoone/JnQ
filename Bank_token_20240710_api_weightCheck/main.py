import os, sys

path_project = "../"
path_funcs = "../funcs"
sys.path.append(path_project)
sys.path.append(path_funcs)


from bank_loop import * 



bank = Bank(path_config="./config.json", 
            path_api="../keys/binance_mademerich.pkl", # 'r' preserve away 't, a' other vague letters.
            path_dir_df_res =  "./data/df_res",
            path_save_log="./logs/{}.log".format(datetime.now().strftime('%Y%m%d%H%M%S')),
            api_rate_limit = 2400, # per minute. (set margin -10)
            table_account_name = 'table_account',
            table_condition_name = 'table_condition',
            table_trade_name = 'table_trade',
            table_log_name = 'table_log',
            chat_id="5320962614",
            token="7156517117:AAF_Qsz3pPTHSHWY-ZKb8LSKP6RyHSMTupo", # TheBank
            # token="6961110608:AAEH4_vSxmRxJ2OnGQCk0NocfmH-iujHyNs", # BotBinance (for .ipynb)
            # token="6206784409:AAGbiGMNPBFn1SzJo_dqekGmFHTJfp0llVM", # BinanceMessenger (Tester)
            receive_limit_ms=1000*3600)


if __name__=="__main__":
    
    while 1:
        loop_table_condition(bank) # debugging.
        loop_table_trade(bank)
        
        # bank.sys_log.debug(f"tokens : {bank.token_bucket.tokens}")
        # clear_output(wait=True) # if run in ipynb.


        

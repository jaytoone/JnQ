import os
import pickle


# if(os.path.exists("binance_f/privateconfig.py")):
#     from binance_f.privateconfig import *
#     g_api_key = p_api_key
#     g_secret_key = p_secret_key
# else:

# ------ Todo, check key_abspath ------ #
# key_abspath = r"D:\Projects\System_Trading\JnQ\api_keys\binance_mademerich.pkl"  # static define
key_abspath = os.path.abspath("../api_keys/binance_mademerich.pkl")  # dynamic define
# key_abspath = os.path.abspath("../api_keys/binance_JnQ.pkl")   # JnQ : restricted IP adjusted

with open(key_abspath, 'rb') as f:
    api_list = pickle.load(f)
g_api_key = api_list[0]
g_secret_key = api_list[1]
# print(g_api_key)

# g_account_id = 12345678



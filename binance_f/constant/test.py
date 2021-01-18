import os
import pickle

with open('binance_key.p', 'rb') as f:
    api_list = pickle.load(f)

if(os.path.exists("binance_f/privateconfig.py")):
    from binance_f.privateconfig import *
    g_api_key = p_api_key
    g_secret_key = p_secret_key

else:
    # with open('binance_key.p', 'rb') as f:
    #     api_key = pickle.load(f)
    g_api_key = api_list[0]
    g_secret_key = api_list[1]


# g_account_id = 12345678
g_account_id = 'toomuch2281@gmail.com'



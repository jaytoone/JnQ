import os
import pickle

# print("os.getcwd() in test.py :", os.getcwd())

key_abspath = os.path.abspath("../private_info/mademerich.pkl")
# key_abspath = os.path.abspath("../private_info/JnQ.pkl")
# key_abspath = os.path.abspath(__file__)
# print("key_abspath :", key_abspath)

# key_abspath = os.path.realpath(__file__)
# print("key_realpath :", key_abspath)

with open(key_abspath, 'rb') as f:
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

    # print(g_api_key)

# g_account_id = 12345678



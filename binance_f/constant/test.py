import os
import pickle

# ------ Todo, check key_abspath ------ #
key_abspath = r"D:\Projects\System_Trading\JnQ\private_info\mademerich.pkl"  # ../ 는 상대적이니까 차라리 고정시킴.
# key_abspath = os.path.abspath("../private_info/mademerich.pkl")
# key_abspath = os.path.abspath("../private_info/JnQ.pkl")   # restricted IP access
# key_abspath = os.path.abspath(__file__)
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



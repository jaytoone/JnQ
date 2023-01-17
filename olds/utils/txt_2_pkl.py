import pickle
import os

target_file_path = "./api_keys/JnQ2.txt"
target_file_path = "./ticker_list/binance_futures_20211207.pkl"
key_abspath = os.path.abspath(target_file_path)

if target_file_path.endswith("txt"):
    with open(key_abspath, 'r') as f:
        # api_list = f.readlines()
        api_list = f.read().splitlines()
        print("api_list :", api_list)

    #       save        #
    with open(key_abspath.replace("txt", "pkl"), 'wb') as f:
        pickle.dump(api_list, f)
        print('api_list saved !')

else:
    #       validation      #
    with open(key_abspath.replace("txt", "pkl"), 'rb') as f:
        api_list = pickle.load(f)
        print("api_list :", api_list)
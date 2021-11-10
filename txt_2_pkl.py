import pickle
import os


key_abspath = os.path.abspath("./private_info/JnQ2.txt")

save_mode = 0

if save_mode:
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
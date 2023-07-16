import os
import pickle

data_dir_path = r"D:\Projects\System_Trading\JnQ\Bank\data\wave_cci_wrr32_spread_wave_length4"
post_fix = "_dict.pkl"
data_dir_path = r"D:\Projects\System_Trading\JnQ\Bank\data\wave_cci_wrr32_spread_wave_length4\replication"
post_fix = "1688202990.pkl"
file_list = os.listdir(data_dir_path)

remove_list = []

for file in file_list:
    if file.endswith(post_fix):
        file_path = os.path.join(data_dir_path, file)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            print("{} loaded.".format(file_path))
            for code, data_ in data.copy().items():

                # 1. pop data.
                # if data_["name"] in remove_list:
                #     data.pop(code)
                #     print("{} removed.".format(data_["name"]))
                #     continue
                for key in data_.keys():
                    # if key == "order_no_list":
                    #     data[code]["order_no_list"] = [""]
                    if "df" not in key:
                        print("{} : {}".format(key, data_[key]))
                # if data_["trade_log_dict"] == 0:
                #     data_["trade_log_dict"] = {}
                print("\n")

        # 2. edit.
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
            print("{} edited.".format(file_path))
        print("\n")

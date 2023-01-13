import importlib


# ------- input params ------- #
strat_pkg = 'banking'
frame_ver = "0204_candlegame"
ID_list = ['3_1']

#       trader       #
trader_name = "{}.traders.{}_trader".format(strat_pkg, frame_ver)
trader_lib = importlib.import_module(trader_name)

#       for enlist rtc & tr     #
utils_public_name = "{}.utils.{}_utils_public".format(strat_pkg, frame_ver)
utils_public_lib = importlib.import_module(utils_public_name)

u_name_list = ["{}.utils.{}_utils_{}".format(strat_pkg, frame_ver, id_) for id_ in ID_list]
utils_list = [importlib.import_module(u_name) for u_name in u_name_list]

#       config       #
config_list = ["{}_config_{}.json".format(frame_ver, id_) for id_ in ID_list]


if __name__ == '__main__':

    trader = trader_lib.Trader(utils_public=utils_public_lib, utils_list=utils_list, config_list=config_list)
    trader.run()

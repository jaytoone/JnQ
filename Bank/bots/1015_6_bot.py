import importlib


# ------- input params ------- #
strat_pkg = 'Bank'
frame_ver = "1015_6"
ID_list = ['hybrid15T', 'hybrid30T']

# ------- trader ------- #
trader_name = "{}.traders.{}_trader".format(strat_pkg, frame_ver)    # partial_suffix change
trader_lib = importlib.import_module(trader_name)

# ------- utils_ ------- #
utils_public_name = "{}.utils.{}_utils_public".format(strat_pkg, frame_ver)
utils_public_lib = importlib.import_module(utils_public_name)

u_name_list = ["{}.utils.{}_utils_{}".format(strat_pkg, frame_ver, id_) for id_ in ID_list]
utils_list = [importlib.import_module(u_name) for u_name in u_name_list]

# ------- config ------- #  Todo, backtrader bot 이 달라야할 곳은 config 밖에 없음
config_name_list = ["{}_config_{}.json".format(frame_ver, id_) for id_ in ID_list]


if __name__ == '__main__':

    trader = trader_lib.Trader(utils_public=utils_public_lib, utils_list=utils_list, config_name_list=config_name_list)
    trader.run()

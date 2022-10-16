import pandas as pd


if __name__ == '__main__':

    res_df_ = pd.read_feather(
        r"C:\Users\Lenovo\PycharmProjects\System_Trading\JnQ\candlestick_concated\database_bn\2022-07-19\2022-07-19 ETHUSDT_1m.ftr",
        columns=None, use_threads=True).set_index("index")
    print(res_df_.tail())
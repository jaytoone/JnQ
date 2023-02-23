
"""
0. Trader version 관리를 위해, traders/ 하위에 배치시킴. 
1. list 형식의 utils & configuration 을 다루기 위해 importlib 형식을 도입함. (in Trader)
    a. public 이라는 공통된 paper 에 utils, configuration 기입하는 구조
"""
from Bank.traders.trader_v1_1 import *

# ------- input params ------- #
paper_name = "wave_cci_wrr32_wave_length"
id_list = [1]


if __name__ == '__main__':

    trader = Trader(paper_name=paper_name, id_list=id_list, backtrade=True)
    trader.run()

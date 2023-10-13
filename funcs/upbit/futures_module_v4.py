# import pyupbit
# from pyupbit import Upbit
# from pyupbit import WebSocketClient
from pyupbit import *
# from constant import *

import threading
# import multiprocessing as mp

# from binance.um_futures import UMFutures
# from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient
# # from binance.lib.utils import config_logging
# from binance.error import ClientError

import math
import pandas as pd
import logging
import time


class UpbitModule(Upbit):

    """
    binance_futures 와 연관된 개별적인 method 의 집합 Class
    v3_1 -> v4
        1. allow server_time.
    """

    def __init__(self, config, **kwargs):

        Upbit.__init__(self, access=kwargs['api_key'], secret=kwargs['secret_key'])

        # self.websocket_queue = mp.Queue()
        # self.websocket_client = mp.Process(
        #     target=pyupbit.WebSocketClient,
        #     args=('ticker', [symbol], self.websocket_queue),
        #     daemon=True
        # )
        # self.websocket_client.start()

        self.thread = threading.Thread(
            target=self.agg_trade_message_handler,
            args=()
        )
        self.thread.start()

        # self.websocket_client = UMFuturesWebsocketClient()
        # self.websocket_client.start()

        self.market_price = {}

        self.config = config
        self.sys_log = logging.getLogger()

    # def get_limit_leverage(self, code):
    #     server_time = self.time()['serverTime']
    #     response = self.leverage_brackets(symbol=code, recvWindow=6000, timestamp=server_time)
    #     return response[0]['brackets'][0]['initialLeverage']

    """
    거래소별, 수정 필요 methods
    """
    def get_available_balance(self, asset_type='KRW'):
        while 1:
            try:
                response = self.get_balances()

                # server_time = self.time()['serverTime']
                # response = self.balance(recvWindow=6000, timestamp=server_time)
            except Exception as e:
                msg = "error in get_balance() : {}".format(e)
                self.sys_log.error(msg)
                self.msg_bot.sendMessage(chat_id=self.chat_id, text=msg)

                time.sleep(self.config.trader_set.api_term)
            else:
                available_asset = float([res['balance'] for res in response if res['currency'] == asset_type][0])
                return self.calc_with_precision(available_asset, 0)

    def agg_trade_message_handler(self):
        """
        1. websocket streaming method 를 이용한 get_market_price method.
            a. try --> received data = None 일 경우를 대비한다.
        """
        while 1:
            try:
                self.market_price = get_current_price(list(self.market_price.keys()))
                time.sleep(0.15)
            except Exception as e:
                # print("e : {}".format(e))
                pass

    def get_market_price_v4(self, code):

        """
        v3 --> v4
            1. assert return value is not None.
        """
        while 1:
            try:
                res = self.market_price[code]
                if res is not None:
                    return res
            except Exception as e:
                msg = "error in get_market_price : {}".format(e)
                self.sys_log.error(msg)
                # self.msg_bot.sendMessage(chat_id=self.chat_id, text=msg)

                time.sleep(self.config.trader_set.api_term)

    # def get_market_price_v2(self, code):
    #     """
    #     1. agg_trades is faster than ticker_price
    #         a. subscribe method 가 24시간 이상 유지되지 않는다고해, restful api method 사용함.
    #     """
    #     response = self.agg_trades(code)
    #     return float(response[-1]['p'])

    def get_order_info(self, code, order_id):
        while 1:
            try:
                # server_time = self.time()['serverTime']
                # order_info_res = self.query_order(symbol=code, orderId=order_id, recvWindow=2000, timestamp=server_time)

                order_info_res = self.get_order(order_id)
                # {'uuid': '97cbed9b-e8c5-442f-ac3e-402a0d4396ec', 'side': 'bid', 'ord_type': 'limit', 'price': '613', 'state': 'cancel', 'market': 'KRW-XRP',
                # 'created_at': '2023-07-15T10:06:32+09:00', 'volume': '10', 'remaining_volume': '10', 'reserved_fee': '3.065', 'remaining_fee': '3.065', 'paid_fee': '0',
                # 'locked': '6133.065', 'executed_volume': '0', 'trades_count': 0, 'trades': []}

            except Exception as e:
                msg = "error in get_order_info : {}".format(e)
                self.sys_log.error(msg)
                self.msg_bot.sendMessage(chat_id=self.chat_id, text=msg)

                time.sleep(self.config.trader_set.api_term)   # for prevent max retries error
            else:
                return order_info_res

    # def get_precision(self, code):
    #     while 1:
    #         try:
    #             response = self.exchange_info()
    #         except Exception as e:
    #             msg = "error in get_precision : {}".format(e)
    #             self.sys_log.error(msg)
    #             self.msg_bot.sendMessage(chat_id=self.chat_id, text=msg)
    #
    #             time.sleep(self.config.trader_set.api_term)
    #         else:
    #             price_precision, quantity_precision = [[data['pricePrecision'], data['quantityPrecision']]
    #                                                    for data in response['symbols'] if data['symbol'] == code][0]
    #             self.sys_log.info('price_precision : {}'.format(price_precision))
    #             self.sys_log.info('quantity_precision : {}'.format(quantity_precision))
    #             return price_precision, quantity_precision

    def get_tick_size(self, price, method="floor"):
        """원화마켓 주문 가격 단위

        Args:
            price (float]): 주문 가격
            method (str, optional): 주문 가격 계산 방식. Defaults to "floor".

        Returns:
            float: 업비트 원화 마켓 주문 가격 단위로 조정된 가격
        """

        if method == "floor":
            func = math.floor
        elif method == "round":
            func = round
        else:
            func = math.ceil

        if price >= 2000000:
            tick_size = func(price / 1000) * 1000
        elif price >= 1000000:
            tick_size = func(price / 500) * 500
        elif price >= 500000:
            tick_size = func(price / 100) * 100
        elif price >= 100000:
            tick_size = func(price / 50) * 50
        elif price >= 10000:
            tick_size = func(price / 10) * 10
        elif price >= 1000:
            tick_size = func(price / 5) * 5
        elif price >= 100:
            tick_size = func(price / 1) * 1
        elif price >= 10:
            tick_size = func(price / 0.1) / 10
        elif price >= 1:
            tick_size = func(price / 0.01) / 100
        elif price >= 0.1:
            tick_size = func(price / 0.001) / 1000
        else:
            tick_size = func(price / 0.0001) / 10000

        return tick_size

    @staticmethod
    def get_exec_price(order_info):
        
        # 'bid_types': ['limit', 'price'], 'ask_types': ['limit', 'market']
        if order_info['ord_type'] == 'limit':
            exec_price = float(order_info['price'])
        else:
            exec_price = float(order_info['trades'][0]['price'])

        return exec_price

    @staticmethod
    def get_exec_qty(order_info):
        return float(order_info['executed_volume'])

    @staticmethod
    def get_exec_ratio(order_info):
        return float(order_info['executed_volume']) / float(order_info['volume'])

    @staticmethod
    def check_execution(order_info, qty_precision):
        return (float(order_info['volume']) - float(order_info['executed_volume'])) < 1 / (10 ** qty_precision)

    """
    거래소별, 수정 불필요 methods
    """
    @staticmethod
    def get_precision_by_price(price):
        try:
            precision = len(str(price).split('.')[1])
        except Exception as e:
            precision = 0
        return precision

    @staticmethod
    def calc_with_precision(data, data_precision, def_type='floor'):

        if not pd.isna(data):
            if data_precision > 0:
                if def_type == 'floor':
                    data = math.floor(data * (10 ** data_precision)) / (10 ** data_precision)
                elif def_type == 'round':
                    data = float(round(data, data_precision))
                else:
                    data = math.ceil(data * (10 ** data_precision)) / (10 ** data_precision)
            else:
                data = int(data)

        return data


if __name__ == '__main__':

    """
    test_vars. should be named differ, by from _ import *    
    """
    t_tp_list = [91.8, 91.7, 91.65]
    t_partial_qty_divider = 1.5
    t_quantity_precision = 1
    # t_symbol = 'XRPUSDT'
    t_symbol = 'ETHUSDT'
    t_symbol = 'BNBUSDT'
    # quantity =
    # symbol = 'ADAUSDT'

    from easydict import EasyDict
    import pickle
    import json

    # key_abspath = r"D:\Projects\System_Trading\JnQ\Bank\api_keys\binance_mademerich.pkl"
    key_abspath = r"D:\Projects\System_Trading\JnQ\Bank_Upbit\api_keys\upbit_key.pkl"
    with open(key_abspath, 'rb') as f:
        api_key, secret_key = pickle.load(f)

    json_file_path = r"D:\Projects\System_Trading\JnQ\Bank\papers\config\wave_cci_wrr32_spread_wave_length_4.json"
    with open(json_file_path, 'r') as f:
        upbit_module = UpbitModule(EasyDict(json.load(f)), api_key=api_key, secret_key=secret_key)

    # config_logging(logging, logging.DEBUG)

    # um_futures_client = UMFutures(key=api_key, secret=secret_key)

    print(len(get_tickers()))
    quit()

    # start_time = time.time()
    # upbit_module.websocket_client.agg_trade(
    #     symbol='ethusdt',
    #     id=1,
    #     callback=upbit_module.agg_trade_message_handler,
    # )
    # upbit_module.websocket_client.agg_trade(
    #     symbol='ethusdt',
    #     id=1,
    #     callback=upbit_module.agg_trade_message_handler,
    # )


    # def add_socket(code):
    #     # code_exist = list(upbit_module.market_price.keys())
    #     # code_exist.append(code)
    #     upbit_module.market_price[code] = None
    #     # print(code_exist)
    #     queue = mp.Queue()
    #     proc = mp.Process(
    #         target=WebSocketClient,
    #         args=('ticker', list(upbit_module.market_price.keys()), queue),
    #         daemon=True
    #     )
    #     proc.start()
    #     return queue


    try:
        # queue = mp.Queue()
        # proc = mp.Process(
        #     target=upbit_module.agg_trade_message_handler,
        #     args=(),
        #     daemon=True
        # )
        # proc.start()

        # thread_ = threading.Thread(
        #     target=upbit_module.agg_trade_message_handler,
        #     args=()
        # )
        # thread_.start()
        # queue = add_socket('KRW-BTC')
        # queue = add_socket('KRW-XRP')
        # queue = add_socket('KRW-MBL')
        rq_list = ['KRW-BTC', 'KRW-XRP', 'KRW-MBL']
        upbit_module.market_price['KRW-BTC'] = None
        upbit_module.market_price['KRW-XRP'] = None
        upbit_module.market_price['KRW-MBL'] = None
        # def

        while True:
            # data = queue.get()
            # # print(data)
            # upbit_module.market_price[data['code']] = data['trade_price']
            # print(upbit_module.market_price)
            # upbit_module.agg_trade_message_handler(rq_list)

            # print(upbit_module.market_price)
            print(upbit_module.get_market_price_v4('KRW-BTC'))
            # print(upbit_module.get_current_price(['KRW-BTC']))
            # print(get_current_price(['KRW-BTC']))
            # print()
            time.sleep(1)

        # upbit_module.get_limit_leverage()
        # while 1:
        #     # response = upbit_module.get_market_price_v2()
        #     # print(response)
        #     try:
        #         print(upbit_module.market_price)
        #         # print(upbit_module.websocket_client._conns.keys())
        #         # print(type(upbit_module.market_price['ETHUSDT']))
        #     except:
        #         pass
        #     time.sleep(1)

        #     if time.time() - start_time > 3:
        #         # upbit_module.websocket_client.agg_trade(
        #         #             symbol='ethusdt',
        #         #             id=2,
        #         #             callback=upbit_module.agg_trade_message_handler,
        #         #         )
        #         # print("add successed.")
        #         upbit_module.websocket_client.stop_socket('ethusdt@aggTrade')
        #         print("stop_socket successed.")
        #         break
        #
        # upbit_module.websocket_client.agg_trade(
        #     symbol='btcusdt',
        #     id=1,
        #     callback=upbit_module.agg_trade_message_handler,
        # )
        # upbit_module.websocket_client.agg_trade(
        #     symbol='ethusdt',
        #     id=1,
        #     callback=upbit_module.agg_trade_message_handler,
        # )
        #
        # while 1:
        #     # try:
        #     #     print(upbit_module.market_price)
        #     # except:
        #     #     pass
        #     time.sleep(0.1)

        # response = upbit_module.get_order_info(t_symbol, 46216493671)
        # response = upbit_module.get_limit_leverage()
        # response = upbit_module.get_available_balance()

        # uuid = '97cbed9b-e8c5-442f-ac3e-402a0d4396ec'
        # order_info = upbit_module.get_order_info(code=None, order_id=uuid)
        # # print(type(response))
        # # print((response['executed_volume']))
        # # response = upbit_module.get_exec_qty(response)
        # # print(float(order_info['executed_volume']) / float(order_info['volume']))
        # response = upbit_module.get_exec_ratio(order_info)
        # response = upbit_module.get_precision_by_price(100.34)

        # response = upbit_module.get_precision()
        # response = um_futures_client.new_order(timeInForce=TimeInForce.GTC,
        #                                        symbol=t_symbol,
        #                                        side='BUY',
        #                                        positionSide='LONG',
        #                                        type=OrderType.LIMIT,
        #                                        quantity=str(0.01),
        #                                        price=str(1700))

        # response = um_futures_client.agg_trades(t_symbol)

        # print(type(response[0]))
        # print(response)
        # {'orderId': 8389765589627969208, 'symbol': 'ETHUSDT', 'status': 'NEW', 'clientOrderId': '5sLtSYvidsGdLf6Zyo0c2s', 'price': '1700', 'avgPrice': '0.00000', 'origQty': '0.010',
        # 'executedQty': '0', 'cumQty': '0', 'cumQuote': '0', 'timeInForce': 'GTC', 'type': 'LIMIT', 'reduceOnly': False, 'closePosition': False, 'side': 'BUY', 'positionSide': 'LONG',
        # 'stopPrice': '0', 'workingType': 'CONTRACT_PRICE', 'priceProtect': False, 'origType': 'LIMIT', 'updateTime': 1680420492874}

    except Exception as error:
        logging.error(
            "Found error. status: {}, error code: {}, error message: {}".format(
                error.status_code, error.error_code, error.error_message
            )
        )

    # class Test:
    #
    #     def __init__(self):
    #         price_ = None
    #
    # test = Test

    # from database_insert import create_table
    # from base_sql import Session

    # realtime_output = {'price': []}


    # def message_handler(message):
    #     print(message)
    #     # for key, item in message.items():
    #     #     if key == 'p':
    #     #         print(key, item)
    #     #         realtime_output['price'].append(item)
    #     # print("data added.\n")
    #     # pass
    #     # return message

    # response = um_futures_client.new_listen_key()
    # print(response)
    #
    # my_client = UMFuturesWebsocketClient()
    # my_client.start()
    # # my_client.start()
    #
    # my_client.agg_trade(
    #     symbol="ethusdt",
    #     id=1,
    #     callback=message_handler,
    # )
    # # my_client.agg_trade(
    # #     symbol="btcusdt",
    # #     id=2,
    # #     callback=message_handler,
    # # )
    # # time.sleep(10)
    # start_time = time.time()
    # while 1:
    #     # if len(my_client._conns) > 0:
    #     #     print(dir(my_client._conns['ethusdt@aggTrade']))#.recv())
    #         # print(my_client._conns)
    #     if time.time() - start_time > 6:
    #         break
    #     # time.sleep(1)
    # # my_client.stop()

    # print(realtime_output['price'])

    # print(upbit_module.get_order_info(8389765574377187710))
    # float(order_info['origQty']) - float(order_info['executedQty']))
    # order_info_ = upbit_module.get_order_info(8389765584857223813) # 8389765584857223813 8389765584857223351
    # print(order_info_.origQty, order_info_.executedQty)
    # print(upbit_module.get_exec_price(order_info_))
    # PrintMix.print_data(order_info_)
    # import numpy as np
    # # print(np.random.randint(1, 10))
    # while 1:
    #     upbit_module.change_initial_leverage(symbol=t_symbol, leverage=np.random.randint(1, 10))
    #     time.sleep(0.1)
    # for data in upbit_module.get_exchange_information().symbols:
    #     print(dir(data)) #.symbol)

    # print(upbit_module.check_execution(, 3))
    # PrintMix.print_data(upbit_module.get_order_info(8389765574377187710))
    # print(type(upbit_module.get_order_info(8389765574377187710).status))
    # print(get_position_info(t_symbol))
    # PrintMix.print_data(get_position_info(t_symbol))
    # while 1:
    #     print(get_position_info(t_symbol).positionAmt) #unrealizedProfit)
    #     time.sleep(1)
    # PrintMix.print_data(self.get_adl_quantile())
    # print(json.dumps(self.get_adl_quantile()[0]))
    # PrintBasic.print_obj(self.get_adl_quantile())
    # sub_client.subscribe_aggregate_trade_event(t_symbol.lower(), callback, error)

    # listen_key = self.start_user_data_stream()
    # print("listenKey: ", listen_key)
    #
    # # Keep user data stream
    # result = self.keep_user_data_stream()
    # print("Result: ", result)
    #
    # sub_client.subscribe_user_data_event(listen_key, callback_user, error)
    # result = self.get_account_trades(t_symbol)
    # # result = self.get_account_trades(t_symbol, fromId=1422647525)
    # PrintMix.print_data(result)
    # result = self.change_initial_leverage(symbol=t_symbol, leverage=5)
    # print(result)

    # print(total_income(t_symbol, 1644589260000, 1644589376739))
    # # print(result.symbols)   # type = obj
    # print(get_precision(t_symbol))
    # print([[data.pricePrecision, data.quantityPrecision] for data in result.symbols if data.symbol == t_symbol][0])
    # for data in result.symbols:
    #     if data.symbol == symbol_:
    #         print([data.pricePrecision, data.quantityPrecision])

    # open_side = OrderSide.BUY
    # # close_side = OrderSide.SELL
    # # print(dir(get_order_info(t_symbol, 8389765520388500621)))
    # # print(total_income(t_symbol, 1650933600000, 1650981660000))
    # # sub_client.unsubscribe_all()
    # start_0 = time.time()

    # print(upbit_module.__dict__)
    # print(dir(upbit_module))
    # print(UpbitModule.get_available_balance())

    # sub_client.subscribe_aggregate_trade_event(t_symbol.lower(), callback, error)
    # while 1:
    #     print(upbit_module.get_market_price_v2())
    #     time.sleep(0.5)

    # # sub_client.subscribe_candlestick_event(t_symbol.lower(), '1m', callback, error)
    # # print(dir(sub_client.subscribe_aggregate_trade_event))
    # # quit()
    # while 1:
    #     # print(dir(sub_client.connections[0]))
    #     if 3 > time.time() - start_0 > 2:
    #         sub_client.connections[0].on_failure(error)
    #         # sub_client.connections[0].close()
    #         # sub_client.connections[0].re_connect()
    #     # if sub_client.connections[0].price is not None:
    #     #     print(time.time() - start_0)
    #     #     break
    #     print("sub_client.connections[0].price :", sub_client.connections[0].price)
    #
    #     time.sleep(1)

    # print(dir(sub_client.connections[0].request))
    # print((sub_client.connections[0].request.json_parser.price))
    # print("dir(sub_client.connections[1]) :", dir(sub_client.connections[1]))
    # print(sub_client.connections[0].price)
    # print((sub_client.connections.price))

    # print(get_precision_by_price(91.823))
    # result = self.post_order(timeInForce=TimeInForce.GTC, symbol=t_symbol,
    #                           side='BUY',
    #                           positionSide="LONG",
    #                           ordertype=OrderType.LIMIT,
    #                           quantity=str(15.0), price=str(0.6415),
    #                           # reduceOnly=False
    #                                    )
    # result = self.post_order(symbol=t_symbol,
    #                           side='BUY',
    #                           positionSide="LONG",
    #                           ordertype=OrderType.MARKET,
    #                           quantity=str(15.0)
    #                           # reduceOnly=False
    #                                    )
    # result = self.post_order(symbol=t_symbol,
    #                                    side='SELL',
    #                                    positionSide="LONG",
    #                                    ordertype=OrderType.MARKET,
    #                                    quantity=str(15.0)
    #                                    # reduceOnly=False
    #                                    )
    # #{"orderId":19419988889,"symbol":"XRPUSDT","status":"NEW","clientOrderId":"8NaTskf7GFXnwUyD3Z4BME",
    # # "price":"0.6515","avgPrice":"0.00000","origQty":"15","executedQty":"0","cumQty":"0","cumQuote":"0",
    # # "timeInForce":"GTC","type":"LIMIT","reduceOnly":false,"closePosition":false,"side":"BUY","positionSide":"BOTH",
    # # "stopPrice":"0","workingType":"CONTRACT_PRICE","priceProtect":false,"origType":"LIMIT","updateTime":1644024833731}
    # res_obj = self.post_order(symbol=t_symbol, side=OrderSide.SELL,
    #                                     ordertype=OrderType.MARKET,
    #                                     quantity=str(15.0),
    #                                     reduceOnly=False)
    # {"orderId": 19420890060, "symbol": "XRPUSDT", "status": "NEW", "clientOrderId": "cSkFxDVuYPyViD1O1pcypT", "price": "0", "avgPrice": "0.00000",
    #  "origQty": "15", "executedQty": "0", "cumQty": "0", "cumQuote": "0", "timeInForce": "GTC", "type": "MARKET", "reduceOnly": false,
    #  "closePosition": false, "side": "BUY", "positionSide": "BOTH", "stopPrice": "0", "workingType": "CONTRACT_PRICE", "priceProtect": false,
    #  "origType": "MARKET", "updateTime": 1644032979553}
    # print(result.orderId)

    # result = self.get_order(t_symbol, orderId=19420890060)
    # # # print(float(result.origQty) - float(result.executedQty))
    # PrintBasic.print_obj(result)

    # result = self.get_open_orders()
    # PrintMix.print_data(result)

    # import time
    # time.sleep(5)
    # try:
    #     result = self.cancel_order(symbol=symbol, orderId=19420814937)
    # except Exception as e:
    #     print(str(e))
    #     if '-2011' not in str(e):
    #         print(1)
    # print(result)
    # #{"orderId":19420814937,"symbol":"XRPUSDT","status":"CANCELED","clientOrderId":"O1xL7JhPxLw9Z4RSZyJr0F",
    # # "price":"0.6515","avgPrice":"0.00000","origQty":"15","executedQty":"0","cumQty":"0","cumQuote":"0",
    # # "timeInForce":"GTC","type":"LIMIT","reduceOnly":false,"closePosition":false,"side":"BUY","positionSide":"BOTH",
    # # "stopPrice":"0","workingType":"CONTRACT_PRICE","priceProtect":false,"origType":"LIMIT","updateTime":1644032291286}
    # PrintBasic.print_obj(result)
    # quit()
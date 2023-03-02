from binance_f import RequestClient
from binance_f import SubscriptionClient
from binance_f.model import *
from binance_f.base.printobject import *
from binance_f.exception.binanceapiexception import BinanceApiException

import math
import pandas as pd
import logging
import time


class FuturesModule(RequestClient, SubscriptionClient):

    """
    binance_futures 와 연관된 개별적인 method 의 집합 Class
    """

    def __init__(self, config, **kwargs):
        RequestClient.__init__(self, **kwargs)
        SubscriptionClient.__init__(self, **kwargs)

        self.config = config
        self.sys_log = logging.getLogger()

        SubscriptionClient.subscribe_aggregate_trade_event(self, self.config.trader_set.symbol.lower(), self.callback, self.error)    # init 에 static method 사용가능함.

    def get_limit_leverage(self):
        return [res.brackets[0].initialLeverage for res in self.get_leverage_bracket() if res.symbol == self.config.trader_set.symbol][0]

    def get_available_balance(self, asset_type='USDT'):
        while 1:
            try:
                results = self.get_balance_v2()
            except Exception as e:
                self.sys_log.error("error in get_balance_v2() :", e)
            else:
                available_asset = [res.availableBalance for res in results if res.asset == asset_type][0]
                return self.calc_with_precision(available_asset, 2)

    def get_market_price(self):
        result = self.get_symbol_price_ticker(symbol=self.config.trader_set.symbol)

        return result[0].price

    def get_market_price_v2(self):
        while 1:
            res = self.connections[0].price
            if res is not None:
                return res

    def total_income(self, startTime=None, endTime=None):
        results = self.get_income_history(symbol=self.config.trader_set.symbol, startTime=startTime, endTime=endTime)

        total_income_ = 0.0
        for result in results:
            total_income_ += result.income
            print(result.income)

        return total_income_

    def get_order_info(self, orderId):
        while 1:
            try:
                order_info_res = self.get_order(self.config.trader_set.symbol, orderId=orderId)
            except Exception as e:
                self.sys_log.error("error in get_order_info : {}".format(e))
                time.sleep(1)   # for prevent max retries error
                continue
            else:
                return order_info_res

    def get_precision(self):
        while 1:
            try:
                results = self.get_exchange_information()
            except Exception as e:
                self.sys_log.error("error in get_precision :", e)
            else:
                price_precision, quantity_precision = [[data.pricePrecision, data.quantityPrecision] for data in results.symbols if data.symbol == self.config.trader_set.symbol][0]
                self.sys_log.info('price_precision : {}'.format(price_precision))
                self.sys_log.info('quantity_precision : {}'.format(quantity_precision))
                return price_precision, quantity_precision

    def get_trade_history_info(self):    # pnl != percentage
        result = self.get_account_trades(symbol=self.config.trader_set.symbol)

        return result[-1].price

    @staticmethod
    def get_exec_price(order_info):
        if order_info.origType == OrderType.LIMIT:
            exec_price = float(order_info.price)
        else:
            exec_price = float(order_info.avgPrice)

        return exec_price

    @staticmethod
    def get_exec_qty(order_info):
        return float(order_info.executedQty)

    @staticmethod
    def get_exec_ratio(order_info):
        return float(order_info.executedQty) / float(order_info.origQty)

    @staticmethod
    def check_execution(order_info, qty_precision):
        return (float(order_info.origQty) - float(order_info.executedQty)) < 1 / (10 ** qty_precision)

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

    @staticmethod
    def callback(data_type: 'SubscribeMessageType', event: 'any', show_detail=False):
        if data_type == SubscribeMessageType.RESPONSE:
            if show_detail:
                print("Event ID: ", event)
        elif data_type == SubscribeMessageType.PAYLOAD:
            if show_detail:
                PrintBasic.print_obj(event)
            # sub_client.unsubscribe_all()
        else:
            print("Unknown Data:")

        if show_detail:
            print()

    @staticmethod
    def callback_user(data_type: 'SubscribeMessageType', event: 'any'):
        if data_type == SubscribeMessageType.RESPONSE:
            print("Event ID: ", event)
        elif data_type == SubscribeMessageType.PAYLOAD:
            if(event.eventType == "ACCOUNT_UPDATE"):
                print("Event Type: ", event.eventType)
                print("Event time: ", event.eventTime)
                print("Transaction time: ", event.transactionTime)
                print("=== Balances ===")
                PrintMix.print_data(event.balances)
                print("================")
                print("=== Positions ===")
                PrintMix.print_data(event.positions)
                print("================")
            elif(event.eventType == "ORDER_TRADE_UPDATE"):
                print("Event Type: ", event.eventType)
                print("Event time: ", event.eventTime)
                print("Transaction Time: ", event.transactionTime)
                print("Symbol: ", event.symbol)
                print("Client Order Id: ", event.clientOrderId)
                print("Side: ", event.side)
                print("Order Type: ", event.type)
                print("Time in Force: ", event.timeInForce)
                print("Original Quantity: ", event.origQty)
                print("Position Side: ", event.positionSide)
                print("Price: ", event.price)
                print("Average Price: ", event.avgPrice)
                print("Stop Price: ", event.stopPrice)
                print("Execution Type: ", event.executionType)
                print("Order Status: ", event.orderStatus)
                print("Order Id: ", event.orderId)
                print("Order Last Filled Quantity: ", event.lastFilledQty)
                print("Order Filled Accumulated Quantity: ", event.cumulativeFilledQty)
                print("Last Filled Price: ", event.lastFilledPrice)
                print("Commission Asset: ", event.commissionAsset)
                print("Commissions: ", event.commissionAmount)
                print("Order Trade Time: ", event.orderTradeTime)
                print("Trade Id: ", event.tradeID)
                print("Bids Notional: ", event.bidsNotional)
                print("Ask Notional: ", event.asksNotional)
                print("Is this trade the maker side?: ", event.isMarkerSide)
                print("Is this reduce only: ", event.isReduceOnly)
                print("stop price working type: ", event.workingType)
                print("Is this Close-All: ", event.isClosePosition)
                if not event.activationPrice is None:
                    print("Activation Price for Trailing Stop: ", event.activationPrice)
                if not event.callbackRate is None:
                    print("Callback Rate for Trailing Stop: ", event.callbackRate)
            elif(event.eventType == "listenKeyExpired"):
                print("Event: ", event.eventType)
                print("Event time: ", event.eventTime)
                print("CAUTION: YOUR LISTEN-KEY HAS BEEN EXPIRED!!!")
                print("CAUTION: YOUR LISTEN-KEY HAS BEEN EXPIRED!!!")
                print("CAUTION: YOUR LISTEN-KEY HAS BEEN EXPIRED!!!")
        else:
            print("Unknown Data:")
        print()

    @staticmethod
    def error(e: 'BinanceApiException'):
        print(e.error_code + e.error_message)


if __name__ == '__main__':

    #   Todo - test_vars. should be named differ, by from _ import *
    t_tp_list = [91.8, 91.7, 91.65]
    t_partial_qty_divider = 1.5
    t_quantity_precision = 1
    # t_symbol = 'XRPUSDT'
    t_symbol = 'ETHUSDT'
    # quantity =
    # symbol = 'ADAUSDT'

    from easydict import EasyDict
    import pickle
    import json

    key_abspath = r"D:\Projects\System_Trading\JnQ\Bank\api_keys\binance_mademerich.pkl"
    with open(key_abspath, 'rb') as f:
        api_key, secret_key = pickle.load(f)

    json_file_path = r"D:\Projects\System_Trading\JnQ\Bank\papers\config\wave_cci_wrr32_wave_length_1.json"
    with open(json_file_path, 'r') as f:
        futures_module = FuturesModule(EasyDict(json.load(f)), api_key=api_key, secret_key=secret_key)

    # print(futures_module.get_order_info(8389765574377187710))
    # float(order_info.origQty) - float(order_info.executedQty))
    # order_info_ = futures_module.get_order_info(8389765579886005149)
    # print(order_info_.origQty, order_info_.executedQty)
    # print(futures_module.get_exec_price(order_info_))
    # PrintMix.print_data(order_info_)
    import numpy as np
    # print(np.random.randint(1, 10))
    while 1:
        futures_module.change_initial_leverage(symbol=t_symbol, leverage=np.random.randint(1, 10))
        time.sleep(0.1)
    # for data in futures_module.get_exchange_information().symbols:
    #     print(dir(data)) #.symbol)

    # print(futures_module.check_execution(, 3))
    # PrintMix.print_data(futures_module.get_order_info(8389765574377187710))
    # print(type(futures_module.get_order_info(8389765574377187710).status))
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

    # print(futures_module.__dict__)
    # print(dir(futures_module))
    # print(FuturesModule.get_available_balance())

    # sub_client.subscribe_aggregate_trade_event(t_symbol.lower(), callback, error)
    # while 1:
    #     print(futures_module.get_market_price_v2())
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
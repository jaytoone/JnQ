from funcs_binance.binance_futures_bot_config import *
import math
import pandas as pd
import logging
# import time

sys_log = logging.getLogger()


def get_limit_leverage(symbol_):
    results = request_client.get_leverage_bracket()
    # for result in results:
    #     if result.symbol == symbol_:
    #         return result.brackets[0].initialLeverage
    #
    # return None
    return [res.brackets[0].initialLeverage for res in results if res.symbol == symbol_][0]


def get_availableBalance(asset_type='USDT'):
    while 1:
        try:
            results = request_client.get_balance_v2()
        except Exception as e:
            sys_log.error("error in get_balance_v2() :", e)
        else:
            available_asset = [res.availableBalance for res in results if res.asset == asset_type][0]
            # for result in results:
            #     # --- check asset : USDT --- #
            #     if result.asset == asset_type:
            #     # print(result.availableBalance)
            #         available_asset = result.availableBalance
            #         break
            return calc_with_precision(available_asset, 2)


# result = request_client.get_balance_v2()
# PrintMix.print_data(result)


def get_market_price(symbol):
    result = request_client.get_symbol_price_ticker(symbol=symbol)
    # PrintMix.print_data(result)

    return result[0].price


#           get realtime price using websocket           #
def get_market_price_v2(sub_client_):
    while 1:
        res = sub_client_.connections[0].price
        if res is not None:
            return res


#           income history per pair         #
def total_income(symbol_, startTime=None, endTime=None):
    results = request_client.get_income_history(symbol=symbol_, startTime=startTime, endTime=endTime)
    # PrintMix.print_data(result)
    total_income_ = 0.0
    for result in results:
        total_income_ += result.income
        # print(result[i].income)

    return total_income_


def get_order_info(symbol_, orderId):
    while 1:
        try:
            order_info_res = request_client.get_order(symbol_, orderId=orderId)
        except Exception as e:
            sys_log.error("error in get_order_info : {}".format(e))
            continue
        else:
            return order_info_res


def get_execPrice(order_info):
    if order_info.origType == OrderType.LIMIT:
        executedPrice = float(order_info.price)
    else:
        executedPrice = float(order_info.avgPrice)

    return executedPrice


def get_execQty(order_info):
    return float(order_info.executedQty)


def check_execution(order_info, qty_precision):
    return (float(order_info.origQty) - float(order_info.executedQty)) < 1 / (10 ** qty_precision)


# old
def check_exec_by_order_info(symbol_, post_order_res, qty_precision, return_price=False, order_type=OrderType.LIMIT):
    order_info = get_order_info(symbol_, post_order_res.orderId)
    if return_price:
        if order_type == OrderType.LIMIT:
            executedPrice = float(order_info.price)
        else:
            executedPrice = float(order_info.avgPrice)
        return (float(order_info.origQty) - float(order_info.executedQty)) < 1 / (10 ** qty_precision), executedPrice
    return (float(order_info.origQty) - float(order_info.executedQty)) < 1 / (10 ** qty_precision)


# old
def remaining_order_check(symbol_):
    result = request_client.get_open_orders(symbol_)
    if len(result) == 0:
        return None
    else:
        # print(result[0].orderId) #type = int
        return [result[i].orderId for i in range(len(result))]


# old
def remaining_order_info(symbol_):
    # info_col = ['activatePrice', 'avgPrice', 'clientOrderId', 'closePosition', 'cumQuote', 'executedQty',
    #   'json_parse', 'orderId', 'origQty', 'origType', 'positionSide', 'price', 'priceRate', 'reduceOnly',
    #   'side', 'status', 'stopPrice', 'symbol', 'timeInForce', 'type', 'updateTime', 'workingType']

    result = request_client.get_open_orders(symbol_)
    if len(result) == 0:
        return None
    else:
        # print(result[0].orderId) #type = int
        #       Todo        #
        #        index set 을 0 로 설정함으로서 발생할 수 있는 이슈, 추후 고려
        return result[0]


# old
def get_remaining_quantity(symbol_):

    results = request_client.get_position_v2()
    # r_quantity_dict = dict()
    for result in results:
        if result.symbol == symbol_:
            return abs(result.positionAmt)

    # PrintMix.print_data(result)
    # print(r_quantity_dict)

    return None


# old
def get_position_info(symbol_):

    # info_col = ['entryPrice', 'isAutoAddMargin', 'isolatedMargin', 'json_parse', 'leverage', 'liquidationPrice'
    #     , 'marginType', 'markPrice', 'maxNotionalValue', 'positionAmt', 'positionSide', 'symbol', 'unrealizedProfit']

    results = request_client.get_position_v2()
    # r_quantity_dict = dict()
    for result in results:
        if result.symbol == symbol_:
            return result

    # PrintMix.print_data(result)
    # print(r_quantity_dict)

    return None


def get_precision(symbol_):
    while 1:
        try:
            results = request_client.get_exchange_information()
        except Exception as e:
            sys_log.error("error in get_precision :", e)
        else:
            price_precision, quantity_precision = [[data.pricePrecision, data.quantityPrecision] for data in results.symbols if data.symbol == symbol_][0]
            sys_log.info('price_precision : {}'.format(price_precision))
            sys_log.info('quantity_precision : {}'.format(quantity_precision))
            return price_precision, quantity_precision


def get_precision_by_price(price):
    try:
        precision = len(str(price).split('.')[1])
    except Exception as e:
        precision = 0
    return precision


# def calc_precision(number):
#
#     precision_ = 0
#     while 1:
#         if number * (10 ** precision_) == int(number * (10 ** precision_)):
#             break
#         else:
#             precision_ += 1
#
#     return precision_


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


def error(e: 'BinanceApiException'):
    print(e.error_code + e.error_message)


def get_trade_history_info(symbol_):    # pnl != percentage
    result = request_client.get_account_trades(symbol=symbol_)
    # for i in range(len(result)):
    #     print(result[i].price, result[i].realizedPnl)

    return result[-1].price


if __name__ == '__main__':

    #   Todo - test_vars. should be named differ, by from _ import *
    t_tp_list = [91.8, 91.7, 91.65]
    t_partial_qty_divider = 1.5
    t_quantity_precision = 1
    t_symbol = 'XRPUSDT'
    # t_symbol = 'ETHUSDT'
    # quantity =
    # symbol = 'ADAUSDT'

    # result = request_client.get_account_trades(t_symbol)
    # # result = request_client.get_account_trades(t_symbol, fromId=1422647525)
    # PrintMix.print_data(result)

    # print(total_income(t_symbol, 1644589260000, 1644589376739))
    # result = request_client.get_exchange_information()
    # # print(result.symbols)   # type = obj
    # print(get_precision(t_symbol))
    # print([[data.pricePrecision, data.quantityPrecision] for data in result.symbols if data.symbol == t_symbol][0])
    # for data in result.symbols:
    #     if data.symbol == symbol_:
    #         print([data.pricePrecision, data.quantityPrecision])

    open_side = OrderSide.BUY
    # close_side = OrderSide.SELL

    # print(get_precision_by_price(91.823))
    # result = request_client.post_order(timeInForce=TimeInForce.GTC, symbol=t_symbol,
    #                           side='BUY',
    #                           positionSide="LONG",
    #                           ordertype=OrderType.LIMIT,
    #                           quantity=str(15.0), price=str(0.6415),
    #                           # reduceOnly=False
    #                                    )
    # result = request_client.post_order(symbol=t_symbol,
    #                           side='BUY',
    #                           positionSide="LONG",
    #                           ordertype=OrderType.MARKET,
    #                           quantity=str(15.0)
    #                           # reduceOnly=False
    #                                    )
    result = request_client.post_order(symbol=t_symbol,
                                       side='SELL',
                                       positionSide="LONG",
                                       ordertype=OrderType.MARKET,
                                       quantity=str(15.0)
                                       # reduceOnly=False
                                       )
    # #{"orderId":19419988889,"symbol":"XRPUSDT","status":"NEW","clientOrderId":"8NaTskf7GFXnwUyD3Z4BME",
    # # "price":"0.6515","avgPrice":"0.00000","origQty":"15","executedQty":"0","cumQty":"0","cumQuote":"0",
    # # "timeInForce":"GTC","type":"LIMIT","reduceOnly":false,"closePosition":false,"side":"BUY","positionSide":"BOTH",
    # # "stopPrice":"0","workingType":"CONTRACT_PRICE","priceProtect":false,"origType":"LIMIT","updateTime":1644024833731}
    # res_obj = request_client.post_order(symbol=t_symbol, side=OrderSide.SELL,
    #                                     ordertype=OrderType.MARKET,
    #                                     quantity=str(15.0),
    #                                     reduceOnly=False)
    # {"orderId": 19420890060, "symbol": "XRPUSDT", "status": "NEW", "clientOrderId": "cSkFxDVuYPyViD1O1pcypT", "price": "0", "avgPrice": "0.00000",
    #  "origQty": "15", "executedQty": "0", "cumQty": "0", "cumQuote": "0", "timeInForce": "GTC", "type": "MARKET", "reduceOnly": false,
    #  "closePosition": false, "side": "BUY", "positionSide": "BOTH", "stopPrice": "0", "workingType": "CONTRACT_PRICE", "priceProtect": false,
    #  "origType": "MARKET", "updateTime": 1644032979553}
    # print(result.orderId)

    # result = request_client.get_order(t_symbol, orderId=19420890060)
    # # # print(float(result.origQty) - float(result.executedQty))
    # PrintBasic.print_obj(result)

    # result = request_client.get_open_orders()
    # PrintMix.print_data(result)

    # import time
    # time.sleep(5)
    # try:
    #     result = request_client.cancel_order(symbol=symbol, orderId=19420814937)
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
    quit()



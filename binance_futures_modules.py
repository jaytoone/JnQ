from binance_futures_bot_config import *
import math
import pandas as pd


# import time


def get_limit_leverage(symbol_):
    
    result = request_client.get_leverage_bracket()
    leverage_dict = dict()
    for i in range(len(result)):
        dict_symbol = result[i].symbol
        limit_leverage = result[i].brackets[0].initialLeverage
        leverage_dict[dict_symbol] = limit_leverage
    # print(leverage_dict)

    return leverage_dict[symbol_]


#               Get Balance V2              #
def get_availableBalance(asset_type='USDT'):

    results = request_client.get_balance_v2()

    # print(dir(result[0]))
    for result in results:

        # --- check asset : USDT --- #
        if result.asset == asset_type:
        # print(result.availableBalance)
            available_asset = result.availableBalance

    return calc_with_precision(available_asset, 2)


# result = request_client.get_balance_v2()
# PrintMix.print_data(result)


#           Get Market Price            #
def get_market_price(symbol):
    result = request_client.get_symbol_price_ticker(symbol=symbol)
    # PrintMix.print_data(result)

    return result[0].price


#           Get Market Price with websocket           #
def get_market_price_v2(sub_client):

    while 1:
        res = sub_client.connections[0].price
        if res is not None:
            return res


#           Income History per Pair         #
def total_income(symbol, startTime=None, endTime=None):
    result = request_client.get_income_history(symbol=symbol, startTime=startTime, endTime=endTime)
    # PrintMix.print_data(result)
    total_income_ = 0.0
    for i in range(len(result)):
        total_income_ += result[i].income
        # print(result[i].income)

    return total_income_


#           Remaining Order Check           #
def remaining_order_check(symbol_):
    #           Get Open Orders         #
    result = request_client.get_open_orders(symbol_)
    if len(result) == 0:
        return None
    else:
        # print(result[0].orderId) #type = int
        return [result[i].orderId for i in range(len(result))]


def get_remaining_quantity(symbol_):
    result = request_client.get_position_v2()
    r_quantity_dict = dict()
    for i in range(len(result)):
        dict_symbol = result[i].symbol
        r_quantity = result[i].positionAmt
        r_quantity_dict[dict_symbol] = r_quantity
    # PrintMix.print_data(result)
    # print(r_quantity_dict)

    return abs(r_quantity_dict[symbol_])


#           Exchange Information            #
def get_precision(symbol_):
    result = request_client.get_exchange_information()
    # members = [attr for attr in dir(result) if not callable(attr) and not attr.startswith("__")]
    # print(members)
    symbols_data = result.symbols
    precision_dict = dict()
    for i in range(len(symbols_data)):
        # print(result.symbols[i].symbol)
        # print(result.symbols[i].pricePrecision)
        # print(result.symbols[i].quantityPrecision)
        # print(symbols_data[i].symbol, '->', symbols_data[i].pricePrecision, symbols_data[i].quantityPrecision)
        precision_dict[symbols_data[i].symbol] = [symbols_data[i].pricePrecision, symbols_data[i].quantityPrecision]
        # print(precision_dict['BTCUSDT'])

    return precision_dict[symbol_]


def calc_precision(number):

    precision_ = 0
    while 1:
        if number * (10 ** precision_) == int(number * (10 ** precision_)):
            break
        else:
            precision_ += 1

    return precision_


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


def get_partial_tplist(ep, tp, open_side, partial_num, price_precision_):

    #       Todo : ultimate tp 설정       #
    #              calc_precision 씌워야함       #
    tp_list_ = []
    if open_side == OrderSide.BUY:

        for part_i in range(partial_num, 0, -1):
            partial_tp = ep + abs(tp - ep) * (part_i / partial_num)
            tp_list_.append(partial_tp)

        #   Todo : 이부분 왜 있지 ?       #
        # if order.entry_type == OrderType.MARKET:
        #     if ep <= sl_level:
        #         print('ep <= level : %s <= %s' % (ep, sl_level))
        #         continue  # while loop 가 위에 형성되면서 차후 사용을 위해선 break 으로 바뀌어야할 것

    else:

        #      Todo     #
        #       이부분이 문제네, dynamic tp 라, ep 보다 tp 가 커질 수 있음     #
        for part_i in range(partial_num, 0, -1):
            partial_tp = ep - abs(tp - ep) * (part_i / partial_num)
            tp_list_.append(partial_tp)

        # if order.entry_type == OrderType.MARKET:
        #     if ep >= sl_level:
        #         print('ep >= level : %s >= %s' % (ep, sl_level))
        #         continue

    tp_list_ = list(map(lambda x: calc_with_precision(x, price_precision_), tp_list_))

    return tp_list_


def get_partial_tplist_v2(ep, tp, partial_num, price_precision_):

    #       Todo : ultimate tp 설정       #
    #              calc_precision 씌워야함       #
    tp_list_ = []
    if tp > ep:

        for part_i in range(partial_num, 0, -1):
            partial_tp = ep + abs(tp - ep) * (part_i / partial_num)
            tp_list_.append(partial_tp)

        #   Todo : 이부분 왜 있지 ?       #
        # if order.entry_type == OrderType.MARKET:
        #     if ep <= sl_level:
        #         print('ep <= level : %s <= %s' % (ep, sl_level))
        #         continue  # while loop 가 위에 형성되면서 차후 사용을 위해선 break 으로 바뀌어야할 것

    else:

        #      Todo     #
        #       이부분이 문제네, dynamic tp 라, ep 보다 tp 가 커질 수 있음     #
        for part_i in range(partial_num, 0, -1):
            partial_tp = ep - abs(tp - ep) * (part_i / partial_num)
            tp_list_.append(partial_tp)

        # if order.entry_type == OrderType.MARKET:
        #     if ep >= sl_level:
        #         print('ep >= level : %s >= %s' % (ep, sl_level))
        #         continue

    tp_list_ = list(map(lambda x: calc_with_precision(x, price_precision_), tp_list_))

    return tp_list_


def partial_limit(symbol, tp_list_, close_side, quantity_precision, partial_qty_divider):

    tp_count = 0
    tp_type = OrderType.LIMIT
    retry_cnt = 0
    while 1:  # loop for partial tp

        #          get remaining quantity         #
        if tp_count == 0:
            try:
                remain_qty = get_remaining_quantity(symbol)
            except Exception as e:
                print('error in get_remaining_quantity :', e)
                continue

        #           partial tp_level        #
        tp_level = tp_list_[tp_count]

        if tp_count == len(tp_list_) - 1:
            quantity = calc_with_precision(remain_qty, quantity_precision)
            # quantity = calc_with_precision(remain_qty, quantity_precision, def_type='ceil')

        else:
            quantity = remain_qty / partial_qty_divider
            quantity = calc_with_precision(quantity, quantity_precision)

        #       1. 남을 qty 가 최소 주분 qty 보다 작고
        #       2. 반올림 주문 가능한 양이라면, remain_qty 로 order 진행    #
        if 9 / (10 ** (quantity_precision + 1)) < remain_qty - quantity < 1 / (10 ** quantity_precision):
            print('remain_qty, quantity (in qty < 1 / (10 ** quantity_precision) phase) :', remain_qty, quantity)

            #       Todo        #
            #        1. calc_with_precision 은 내림 상태라 r_qty 를 온전히 반영하지 못함      #
            #        2. r_qty - qty < 1 / (10 ** quantity_precision) --> 따라서, ceil 로 반영함
            # quantity = calc_with_precision(remain_qty, quantity_precision)
            quantity = calc_with_precision(remain_qty, quantity_precision, def_type='ceil')

        print('remain_qty, quantity :', remain_qty, quantity)

        #   Todo    #
        #    1. 남은 qty 가 최소 주문 qty_precision 보다 작다면, tp order 하지말고 return       #
        if quantity < 1 / (10 ** quantity_precision):
            return

        #           partial tp             #
        try:
            #           limit order             #
            request_client.post_order(timeInForce=TimeInForce.GTC, symbol=symbol, side=close_side,
                                      ordertype=tp_type,
                                      quantity=str(quantity), price=str(tp_level),
                                      reduceOnly=True)
        except Exception as e:
            print('error in partial tp :', e)

            # quit()

            #        1. Quantity error occurs, tp_list_[0] & remain_qty 로 close_order 진행       #
            #        2. 기존 주문에 추가로 주문이 가능하지 cancel order 진행하지 않고 일단, 진행         #
            #        3. e 를 str 로 변환해주지 않으면, argument of type 'BinanceApiException' is not iterable error 가 발생함
            if "zero" in str(e):
                tp_count = 0
                tp_list_ = [tp_list_[0]]
                continue

            retry_cnt += 1
            if retry_cnt >= 10:
                return "maximum_retry"
            continue
        else:
            tp_count += 1
            if tp_count >= len(tp_list_):
                break
            remain_qty -= quantity
            if remain_qty < 1 / (10 ** quantity_precision):  # --> means remain_qty = 0.0
                break

    return


def get_trade_history_info(symbol_):    # pnl != percentage
    result = request_client.get_account_trades(symbol=symbol_)
    # for i in range(len(result)):
    #     print(result[i].price, result[i].realizedPnl)

    return result[-1].price


if __name__ == '__main__':
    tp_list = [91.8, 91.7, 91.65]
    partial_qty_divider = 1.5
    quantity_precision = 3
    symbol = 'ETHUSDT'
    close_side = OrderSide.SELL

    # result = request_client.get_income_history(symbol=symbol)
    # PrintMix.print_data(result)

    # result = get_trade_history_info(symbol)
    # PrintMix.print_data(result)

    # print(result)

    # result = request_client.get_balance_v2()
    # PrintMix.print_data(result)
    print(get_availableBalance())
    # for r in result:
    #     print(r.availableBalance)
    quit()

    # print(get_precision('ADAUSDT'))
    # print(calc_precision(2.2321, ))
    # print(calc_with_precision(0.23299999999999993, 3, def_type='ceil'))
    print(9 / (10 ** (3 + 1)) < 0.23289999999999993 - 0.232 < 1 / (10 ** (3)))
    print(9 / (10 ** (3 + 1)))
    quit()

    # remained_orderId = remaining_order_check(symbol)
    # print(remained_orderId)

    # result = request_client.get_open_orders(symbol)
    # print([dir(result[i]) for i in range(len(result))])

    # print(get_remaining_quantity(symbol))

    # result = request_client.get_leverage_bracket()
    # leverage_dict = dict()
    # symbol_list = []
    # for i in range(len(result)):
    #     dict_symbol = result[i].symbol
    #     print("dict_symbol :", dict_symbol)
    #
    #     symbol_list.append(dict_symbol)
    #
    # with open("ticker_in_futures.pkl", "wb") as f:
    #     pickle.dump(symbol_list, f)
    #     print("symbol_list saved !")

        # limit_leverage = result[i].brackets[0].initialLeverage
        # leverage_dict[dict_symbol] = limit_leverage


    #           Custom Partial Limit        #
    # partial_limit(symbol, tp_list, close_side, quantity_precision, partial_qty_divider)

    # logger = logging.getLogger("binance-futures")
    # logger.setLevel(level=logging.INFO)
    # handler = logging.StreamHandler()
    # handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    # logger.addHandler(handler)
    #
    # sub_client = SubscriptionClient(api_key=g_api_key, secret_key=g_secret_key)
    # while 1:
    #     result = sub_client.subscribe_symbol_ticker_event("btcusdt", callback, error)
    #     # print(result.lastPrice)
    #     time.sleep(1)

    #               Margin Type for Leverage                #
    # try:
    #     result = request_client.change_margin_type(symbol="BTCUSDT", marginType=FuturesMarginType.ISOLATED)
    #     PrintBasic.print_obj(result)
    # except Exception as e:
    #     print('Error in margin_type :', e)

    #           Set Leverage            #
    # result = request_client.change_initial_leverage(symbol="SXPUSDT", leverage=51)
    # PrintBasic.print_obj(result)

    #           Get Order per Pair & OrderId          #
    # result = request_client.get_order(symbol="ALGOUSDT", orderId=1268453192)
    # PrintBasic.print_obj(result)

    #               Get Position in all Pairs            #
    # result = request_client.get_position()
    # PrintMix.print_data(result)

    #               Get Market Order              #
    # try:
    #     realtime_price = get_market_price("DASHUSDT")
    # except Exception as e:
    #     print('Error in get_market_price :', e)

    # import time
    # sub_client.subscribe_aggregate_trade_event(symbol.lower(), callback, error)
    # start = time.time()
    # while 1:
    #     # print(len(sub_client_2.connections))
    #     # print((sub_client.connections[0].price))
    #     print(get_market_price_v2(sub_client))
    #     time.sleep(.5)
    #     if time.time() - start > 3:
    #         break
    # print('trading done')
    # start = time.time()
    # while 1:
    #     # print(len(sub_client_2.connections))
    #     # print((sub_client.connections[0].price))
    #     print(get_market_price_v2(sub_client))
    #     time.sleep(.5)
    #     if time.time() - start > 3:
    #         break
    # print('trading done')
    # quit()
    # print(obj)
    # members = [attr for attr in dir(obj) if not callable(attr) and not attr.startswith("__")]
    # print(members)
    # price = realtime_price
    price = 87.29
    quantity = 0.002
    price_precision = 2
    qty_precision = 3

    trigger_price = price - 1 * 10 ** -price_precision
    # trigger_price = calc_with_precision(trigger_price, price_precision)
    # print(price, trigger_price)

    #
    # tp_list = [price, price, price]
    #
    # # price = str(calc_with_precision(price, 2))
    # quantity = str(calc_with_precision(quantity, qty_precision))
    #
    # tp_list = list(map(calc_with_precision(x, price_precision), tp_list))
    # print(tp_list)

    # print(calc_with_precision(0.0009999, 3))
    # result = request_client.cancel_all_orders(symbol='DOTUSDT')

    # print('price :', price)
    # print('amt_str :', amt_str)

    # #           Stop Order             #
    # result = request_client.post_order(timeInForce=TimeInForce.GTC, symbol="DASHUSDT", side=OrderSide.BUY,
    #                                    ordertype=OrderType.TAKE_PROFIT, stopPrice=str(trigger_price),
    #                                    quantity=str(quantity), price=str(price), reduceOnly=False)

    #           Take Profit Order           #
    # if close_side == OrderSide.SELL:
    #     stop_price = calc_with_precision(realtime_price + 5 * 10 ** -price_precision,
    #                                      price_precision)
    # else:
    #     stop_price = calc_with_precision(realtime_price - 5 * 10 ** -price_precision,
    #                                      price_precision)

    # request_client.post_order(timeInForce=TimeInForce.GTC, symbol=fundamental.symbol,
    #                           side=close_side,
    #                           ordertype=order.sl_type, stopPrice=str(stop_price),
    #                           quantity=str(quantity), price=str(realtime_price),
    #                           reduceOnly=True)

    #           Limit Order             #
    # request_client.post_order(timeInForce=TimeInForce.GTC, symbol=symbol, side=close_side,
    #                           ordertype=tp_type,
    #                           quantity=str(quantity), price=str(tp_level),
    #                           reduceOnly=True)

    #           Market Order            #
    # result = request_client.post_order(symbol="DOTUSDT", side=OrderSide.SELL, ordertype=OrderType.MARKET,
    #                                    quantity=str(quantity), reduceOnly=True)
    # PrintBasic.print_obj(result)
    # print(get_market_price("COMPUSDT"))

    #                  Cancel Order                     #
    # result = request_client.cancel_order(symbol="FLMUSDT", orderId=103234560)
    # PrintBasic.print_obj(result)

    #               Account Information V2 per Pair --> position 에 대한 amt, 상세 정보를 제외시켰다.                 #
    # result = request_client.get_account_information_v2()
    # print("canDeposit: ", result.canDeposit)
    # print("canWithdraw: ", result.canWithdraw)
    # print("feeTier: ", result.feeTier)
    # print("maxWithdrawAmount: ", result.maxWithdrawAmount)
    # print("totalInitialMargin: ", result.totalInitialMargin)
    # print("totalMaintMargin: ", result.totalMaintMargin)
    # print("totalMarginBalance: ", result.totalMarginBalance)
    # print("totalOpenOrderInitialMargin: ", result.totalOpenOrderInitialMargin)
    # print("totalPositionInitialMargin: ", result.totalPositionInitialMargin)
    # print("totalUnrealizedProfit: ", result.totalUnrealizedProfit)
    # print("totalWalletBalance: ", result.totalWalletBalance)
    # print("totalCrossWalletBalance: ", result.totalCrossWalletBalance)
    # print("totalCrossUnPnl: ", result.totalCrossUnPnl)
    # print("availableBalance: ", result.availableBalance)
    # print("maxWithdrawAmount: ", result.maxWithdrawAmount)
    # print("updateTime: ", result.updateTime)
    # print("=== Assets ===")
    # PrintMix.print_data(result.assets)
    # print("==============")
    # print("=== Positions ===")
    # PrintMix.print_data(result.positions)
    # print("==============")

    #               Account trades per coin History             #
    # result = request_client.get_account_trades(symbol="ALGOUSDT")
    # PrintMix.print_data(result)
    # price, pnl =
    # print(get_trade_history_info("ALGOUSDT"))

    #           All order History per Pair         #
    # result = request_client.get_all_orders(symbol="ALGOUSDT")
    # PrintMix.print_data(result)

    #               Adl quantile --> 어느 symbol 에 어떠한 position 이 존재하는지 알려준다.                #
    #       Example Below
    #       data number 0 :
    # adlQuantile:{'LONG': 0, 'SHORT': 0, 'BOTH': 2}
    # json_parse:<function AdlQuantile.json_parse at 0x00000198D1B4C158>
    # symbol:SXPUSDT

    # result = request_client.get_adl_quantile()
    # PrintMix.print_data(result)

    #           API Trading stats           #
    #           Example Below ---> 유효한 position 이 출력되지 않는다.           #
    # indicators:<object object at 0x000001E94203A710>
    # json_parse:<function ApiTradingStatus.json_parse at 0x000001E93116B378>
    # positions:[]
    # result = request_client.get_api_trading_stats()
    # PrintMix.print_data(result)

    #   --------------------- ETC ---------------------  #
    # print()
    # import time
    # print(get_precision('DOTUSDT'))
    # print(total_income('ALGOUSDT', startTime=None, endTime=None))
    # result = request_client.get_income_history(symbol='ALGOUSDT', startTime=None, endTime=None)
    # print(result[0])

    # result = remaining_order_check('ALGOUSDT')
    # result = request_client.get_open_orders()
    # PrintMix.print_data(result)

    # # print(int(time.time()))
    # # print(calc_with_precision(6.28 / 108.07 * 7, 3))
    # # print(calc_with_precision(6.2895214 / 108.07 * 7, 3))
    # # remaining_order_check()
    # # print('%.2f %%' % (-0.013 / 6.28 * 100))
    # print((get_remaining_quantity('DOTUSDT')))
    # result = request_client.get_position()
    # print(result[0])

    # print(get_limit_leverage('DOTUSDT'))
    # print(get_availableBalance())
    # print(get_precision('BTCUSDT'))

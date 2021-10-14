import os

# switch_path = "C:/Users/Lenovo/PycharmProjects/Project_System_Trading/Rapid_Ascend"

dir_path = os.path.abspath('./../')
print("dir_path :", dir_path)

# os.chdir(switch_path)
os.chdir("../../")
# print("os.getcwd() :", os.getcwd())
# quit()

from binance_futures_modules import *
# from funcs.funcs_for_trade import *
from binance_futures_concat_candlestick import concat_candlestick
from easydict import EasyDict
from SE.utils.v4_1_noninv import *
import pickle

#       Todo        #
#        1. check upper dir
#        2. check utils


class Trader:

    def __init__(self, initial_asset):

        #           static platform variables        #
        self.initial_asset = initial_asset

        self.over_balance = None
        self.min_balance = 5.0  # USDT

        self.trading_fee = 0.0008
        self.accumulated_income = 0.0
        self.calc_accumulated_profit = 1.0
        self.accumulated_profit = 1.0
        self.accumulated_back_profit = 1.0

        self.sub_client = None

    def run(self):

        print('# ----------- Trader v1 ----------- #')

        #        1. save open / close data to dict     #
        #        2. dict name = str(datetime.now().timestamp()).split(".")[0]      #
        #        3. key = str(timeindex), value = [ep, ordertype] | [tp (=exit_price)]
        #        4. logger_name is the trade start time index        #
        # logger_name = "%s.pkl" % str(datetime.now()).split(".")[0]  # <-- invalid arguments
        logger_name = "%s.pkl" % str(datetime.now().timestamp()).split(".")[0]
        trade_log = {}

        initial_run = 1
        while 1:

            #       load config (refreshed by every trade)       #
            with open(dir_path + '/config/v4_1_noninv.json', 'r') as cfg:
                config = EasyDict(json.load(cfg))

            # df_path = dir_path + '/%s.xlsx' % config.init_set.symbol

            if config.init_set.symbol_changed:
                initial_run = 1

                #      Todo     #
                #       1. rewrite symbol_changed to false
                config.init_set.symbol_changed = False

                with open(dir_path + '/config_v2.json', 'w') as cfg:
                    json.dump(config, cfg, indent=2)

            if initial_run:

                while 1:    # for completion
                    # --- 1. leverage type => "isolated" --- #
                    try:
                        request_client.change_margin_type(symbol=config.init_set.symbol, marginType=FuturesMarginType.ISOLATED)
                    except Exception as e:
                        print('error in change_margin_type :', e)
                    else:
                        print('leverage type --> isolated')

                    # --- 2. confirm limit leverage --- #
                    try:
                        limit_leverage = get_limit_leverage(symbol_=config.init_set.symbol)
                    except Exception as e:
                        print('error in get limit_leverage :', e)
                        continue
                    else:
                        print('limit_leverage :', limit_leverage)

                    # --- 3. sub_client --- #
                    try:
                        sub_client.subscribe_aggregate_trade_event(config.init_set.symbol.lower(), callback, error)
                        self.sub_client = sub_client
                    except Exception as e:
                        print('error in get sub_client :', e)
                        continue

                    break

                initial_run = 0
                print()

            #       check run        #
            if not config.init_set.run:
                continue

            #       init order side      #
            open_side = None
            # close_side = None

            #       get startTime     #
            start_timestamp = int(time.time() * 1000)
            entry_const_check = True
            # load_new_df = False   # entry_const_check 과 겹침

            while 1:

                if entry_const_check:

                    #       log last trading time     #
                    #       미체결을 고려해, entry_const_check 마다 log 수행       #
                    trade_log["last_trading_time"] = str(datetime.now())
                    with open(dir_path + "/trade_log/" + logger_name, "wb") as dict_f:
                        pickle.dump(trade_log, dict_f)

                    temp_time = time.time()

                    # if config.init_set.stacked_df_exist:

                    try:

                        new_df_, _ = concat_candlestick(config.init_set.symbol, config.init_set.interval, days=1, limit=config.init_set.use_rows,
                                                        timesleep=0.2, show_process=False)
                        #       Todo        #
                        #        1. 모든 inveral 의 마지막 time_index 은 현재시간보다 커야함 (최신이라면)        #
                        if datetime.timestamp(new_df_.index[-1]) < datetime.now().timestamp():
                            # load_new_df = True
                            continue

                        new_df2_, _ = concat_candlestick(config.init_set.symbol, config.init_set.interval2, days=1, limit=config.init_set.use_rows2,
                                                         timesleep=0.2)
                        if datetime.timestamp(new_df2_.index[-1]) < datetime.now().timestamp():
                            continue

                        new_df3_, _ = concat_candlestick(config.init_set.symbol, config.init_set.interval3, days=1, limit=config.init_set.use_rows3,
                                                         timesleep=0.2)
                        if datetime.timestamp(new_df3_.index[-1]) < datetime.now().timestamp():
                            continue

                        new_df = new_df_.iloc[-config.init_set.use_rows:].copy()
                        new_df2 = new_df2_.iloc[-config.init_set.use_rows2:].copy()
                        new_df3 = new_df3_.iloc[-config.init_set.use_rows3:].copy()
                        print("complete load_df execution time :", datetime.now())

                    except Exception as e:
                        print("error in load new_dfs :", e)
                        continue

                    # if self.save_stacked_df:
                    #
                    #     if config.init_set.stacked_df_exist:
                    #
                    #         try:
                    #             stacked_df = pd.read_excel(df_path, index_col=0)
                    #
                    #             #    use rows   #
                    #             #       add ep columns for append      #
                    #             # new_df['ep'] = np.nan
                    #             prev_complete_df = stacked_df.iloc[:-1, :]
                    #             stacked_df = prev_complete_df.append(new_df)
                    #
                    #             #       Todo        #
                    #             #        1. keep='last' 로 변경해놓음, 추후 사용시 recheck
                    #             stacked_df = stacked_df[~stacked_df.index.duplicated(keep='last')].iloc[
                    #                          -config.init_set.use_rows:, :]
                    #
                    #         except Exception as e:
                    #             print("error in making stacked_df :", e)
                    #             continue
                    #
                    #     else:
                    #
                    #         try:
                    #             # if not config.init_set.stacked_df_exist:
                    #             days = calc_train_days(interval=config.init_set.interval, use_rows=config.init_set.use_rows)
                    #             stacked_df, _ = concat_candlestick(config.init_set.symbol, config.init_set.interval, days=days, timesleep=0.2)
                    #             stacked_df = stacked_df.iloc[-config.init_set.use_rows:, :]
                    #             config.init_set.stacked_df_exist = True
                    #
                    #             stacked_df.to_excel(df_path)
                    #
                    #         except Exception as e:
                    #             print("error in load new stacked_df :", e)
                    #             config.init_set.stacked_df_exist = False
                    #             continue

                    print("~ load_new_df time : %.2f" % (time.time() - temp_time))
                    # temp_time = time.time()

                    try:
                        #       latency 를 줄이기 위해서는, input df length 를 조절해야함     #
                        res_df = sync_check(new_df, new_df2, new_df3)

                        print('~ sync_check time : %.5f' % (time.time() - temp_time))

                    except Exception as e:
                        print("error in sync_check :", e)
                        continue

                    #        Todo       #
                    #         ------------ ep / out / tp ------------ #
                    #           1. 추후 limit_entry 경우, ep 추가 고려
                    #           2. 현재, 완성된 close 기준 data 사용 진행중
                    try:
                        entry, short_ep, long_ep, \
                        short_out, long_out, short_tp, long_tp = enlist_epouttp(res_df, config)
                        print('res_df.index[-1] :', res_df.index[-1])
                        print('~ ep enlist time : %.5f' % (time.time() - temp_time))
                        print()

                    except Exception as e:
                        print('error in enlist_epouttp :', e)
                        entry_const_check = True  # res_df 에 data 가 덮여씌워질 것임
                        continue

                    # ------------ open order constraints phase ------------ #
                    try:

                        #        save res_df before, entry const check      #
                        if config.init_set.df_log:
                            excel_name = str(datetime.now()).replace(":", "").split(".")[0]
                            res_df.to_excel(dir_path + "/df_log/%s.xlsx" % excel_name)

                        entry_score = 0

                        # ---------------------- short ---------------------- #

                        #        1. cbline const. compare with close, so 'close' data should be confirmed
                        #        2. or, entry market_price would be replaced with 'close'
                        if entry[config.init_set.last_index] == config.init_set.short_entry:

                            #           tr scheduling        #
                            if (res_df['close'].iloc[config.init_set.last_index] - short_tp.iloc[config.init_set.last_index] - self.trading_fee * res_df['close'].iloc[config.init_set.last_index]) / (
                                    short_out.iloc[config.init_set.last_index] - res_df['close'].iloc[config.init_set.last_index] + self.trading_fee * res_df['close'].iloc[config.init_set.last_index]) >= config.ep_set.tr_thresh:
                                entry_score -= 1

                            #            mr             #
                            prev_entry_cnt = 0
                            for back_i in range(i - 1, 0, -1):
                                if entry[back_i] == 1:
                                    break

                                elif entry[back_i] == -1:
                                    prev_entry_cnt += 1

                            # print("prev_entry_cnt :", prev_entry_cnt)
                            if prev_entry_cnt <= entry_incycle:
                                # if prev_entry_cnt >= entry_incycle:
                                pass
                            else:
                                i += 1
                                if i >= len(res_df):
                                    break
                                continue

                            if res_df['close'].iloc[config.init_set.last_index] <= res_df['cloud_bline_3m'].iloc[
                                config.init_set.last_index]:
                                entry_score -= 1

                            # --------- score summation --------- #
                            if entry_score == -2:
                                open_side = OrderSide.SELL   # inversion

                        # ---------------------- long ---------------------- #
                        if entry[config.init_set.last_index] == -config.init_set.short_entry:

                        if res_df['close'].iloc[config.init_set.last_index] >= res_df['cloud_bline_3m'].iloc[config.init_set.last_index]:

                                        open_side = OrderSide.SELL

                    except Exception as e:
                        print("error in open const phase :", e)
                        continue

                    entry_const_check = False   # once used, turn off const_check

                check_entry_sec = datetime.now().second
                if open_side is not None:

                    #       Todo        #
                    #        1. const. open market second
                    if check_entry_sec > config.ep_set.check_entry_sec:
                        open_side = None
                        print("check_entry_sec :", check_entry_sec)
                        print()
                        continue    # entry_const_check = False 라서 open_side None phase 로 감

                    break

                    #   1. 추후 position change platform 으로 변경 가능
                    #    a. first_iter 사용

                # ------- open_side is None [ waiting zone ] ------- #
                else:
                    time.sleep(config.init_set.realtime_term)  # <-- term for realtime market function

                    #       check tick change      #
                    #       latest df confirmation       #
                    if datetime.timestamp(res_df.index[-1]) < datetime.now().timestamp():
                        entry_const_check = True
                        print('res_df[-1] timestamp :', datetime.timestamp(res_df.index[-1]))
                        print('current timestamp :', datetime.now().timestamp())
                        print()

                    # time.sleep(config.init_set.close_complete_term)   # <-- term for close completion

            # print('realtime_price :', realtime_price)

            first_iter = True  # 포지션 변경하는 경우, 필요함
            while 1:  # <-- loop for 'check order type change condition'

                print('open_side :', open_side)

                # if config.ep_set.entry_type == OrderType.MARKET:
                #     #             ep with market            #
                #     try:
                #         ep = get_market_price_v2(self.sub_client)
                #     except Exception as e:
                #         print('error in get_market_price :', e)
                #         continue
                # else:

                #             ep limit & market open             #
                if open_side == OrderSide.BUY:
                    ep = short_ep.iloc[config.init_set.last_index]  # inversion mode adjusted
                    out = short_out.iloc[config.init_set.last_index]
                    tp = short_tp.iloc[config.init_set.last_index]
                else:
                    ep = long_ep.iloc[config.init_set.last_index]
                    out = long_out.iloc[config.init_set.last_index]
                    tp = long_tp.iloc[config.init_set.last_index]

                leverage = config.lvrg_set.leverage

                #         get price, volume precision --> reatlime 가격 변동으로 인한 precision 변동 가능성       #
                try:
                    price_precision, quantity_precision = get_precision(config.init_set.symbol)
                except Exception as e:
                    print('error in get price & volume precision :', e)
                    continue
                else:
                    print('price_precision :', price_precision)
                    print('quantity_precision :', quantity_precision)

                ep = calc_with_precision(ep, price_precision)
                out = calc_with_precision(out, price_precision)  # this means static_out
                tp = calc_with_precision(tp, price_precision)   # this means static_tp
                leverage = min(limit_leverage, leverage)
                print('ep :', ep)
                print('out :', out)
                print('tp :', tp)
                print('leverage :', leverage)
                print('~ ep out lvrg set time : %.5f' % (time.time() - temp_time))

                try:
                    request_client.change_initial_leverage(symbol=config.init_set.symbol, leverage=leverage)
                except Exception as e:
                    print('error in change_initial_leverage :', e)
                    continue  # -->  ep market 인 경우에 조심해야함
                else:
                    print('leverage changed -->', leverage)

                if first_iter:
                    #          define start asset          #
                    if self.accumulated_income == 0.0:

                        available_balance = self.initial_asset  # USDT

                    else:
                        available_balance += income

                #          get availableBalance          #
                try:
                    max_available_balance = get_availableBalance()

                    #       over_balance 가 저장되어 있다면, 지속적으로 max_balance 와의 비교 진행      #
                    if self.over_balance is not None:
                        if self.over_balance <= max_available_balance * 0.9:
                            available_balance = self.over_balance
                            self.over_balance = None
                        else:
                            available_balance = max_available_balance * 0.9  # over_balance 를 넘지 않는 선에서 max_balance 채택

                    else:  # <-- 예기치 못한 오류로 인해 over balance 상태가 되었을때의 조치
                        if available_balance > max_available_balance:
                            self.over_balance = available_balance
                            available_balance = max_available_balance * 0.9

                    if available_balance < self.min_balance:
                        print('available_balance %.3f < min_balance' % available_balance)
                        print()
                        break

                except Exception as e:
                    print('error in get_availableBalance :', e)
                    print()
                    continue

                print('~ get balance time : %.5f' % (time.time() - temp_time))

                #          get available open_quantity         #
                open_quantity = available_balance / ep * leverage
                open_quantity = calc_with_precision(open_quantity, quantity_precision)
                if self.over_balance is not None:
                    print('available_balance (temp) :', available_balance)
                else:
                    print('available_balance :', available_balance)

                print("open_quantity :", open_quantity)

                real_balance = ep * open_quantity
                print("real_balance :", real_balance)

                #           open order            #
                orderside_changed = False
                open_retry_cnt = 0

                code_1111 = 0
                while 1:  # <-- loop for complete open order

                    try:
                        if config.ep_set.entry_type == OrderType.MARKET:

                            #           market order            #
                            request_client.post_order(symbol=config.init_set.symbol, side=open_side, ordertype=OrderType.MARKET,
                                                      quantity=str(open_quantity))

                        else:
                            #       limit order      #
                            #       limit order needs execution waiting time & check remaining order         #

                            request_client.post_order(timeInForce=TimeInForce.GTC, symbol=config.init_set.symbol,
                                                               side=open_side,
                                                               ordertype=config.ep_set.entry_type,
                                                               quantity=str(open_quantity), price=str(ep),
                                                               reduceOnly=False)
                    except Exception as e:
                        print('error in limit open order :', e)

                        #        1. price precision validation      #
                        if '-4014' in str(e):
                            try:
                                realtime_price = get_market_price_v2(self.sub_client)
                                price_precision = calc_precision(realtime_price)
                                ep = calc_with_precision(ep, price_precision)
                                print('modified price & precision :', ep, price_precision)

                            except Exception as e:
                                print('error in get_market_price :', e)

                            continue

                        #        -1111 : Precision is over the maximum defined for this asset   #
                        #         = quantity precision error        #
                        if '-1111' in str(e):
                            code_1111 += 1
                            try:
                                _, quantity_precision = get_precision(config.init_set.symbol)
                                if code_1111 % 3 == 0:
                                    quantity_precision -= 1
                                elif code_1111 % 3 == 2:
                                    quantity_precision += 1

                                quantity = available_balance / ep * leverage
                                quantity = calc_with_precision(quantity, quantity_precision, def_type='floor')
                                print('modified qty & precision :', quantity, quantity_precision)

                            except Exception as e:
                                print('error in get modified qty_precision :', e)

                            continue

                        open_retry_cnt += 1

                        #        -2019 : Margin is insufficient     #
                        if '-2019' in str(e):
                            try:
                                max_available_balance = get_availableBalance()
                                #    예기치 못한 오류로 인해 over balance 상태가 되었을때의 조치    #
                                self.over_balance = available_balance
                                available_balance = max_available_balance * 0.9
                                #          Get available quantity         #
                                quantity = available_balance / ep * leverage
                                quantity = calc_with_precision(quantity, quantity_precision, def_type='floor')
                                print('available_balance (temp) :', available_balance)
                                print('quantity :', quantity)

                            except Exception as e:
                                print('error in get_availableBalance() :', e)

                        elif open_retry_cnt > 100:
                            print('open_retry_cnt over 100')
                            quit()
                        continue

                    else:
                        print('open order enlisted :', datetime.now())
                        break

                # -------- execution wait time -------- #
                if config.ep_set.entry_type == OrderType.MARKET:
                    #       enough time for open_quantity consumed      #
                    time.sleep(60 - config.init_set.bar_close_second)
                else:
                    #       limit only, order execution wait time & check breakout_qty_ratio         #
                    while 1:

                        #       temporary : 해당 종가까지 체결 대기       #
                        #       datetime.timestamp(res_df.index[-1]) -> 2분이면, 01:59:999 이런식으로 될것
                        if datetime.now().timestamp() > datetime.timestamp(res_df.index[-1]):
                            # if datetime.now().timestamp() + config.ep_set.entry_execution_wait > datetime.timestamp(res_df.index[-1]):
                            break

                        time.sleep(config.init_set.realtime_term)  # <-- for realtime price function

                #           1. when, open order time expired or executed              #
                #           2. regardless to position exist, cancel all open orders       #
                while 1:
                    try:
                        request_client.cancel_all_orders(symbol=config.init_set.symbol)
                        print()
                    except Exception as e:
                        print('error in cancel remaining open order :', e)
                    else:
                        break

                if orderside_changed:
                    first_iter = False
                    print('orderside_changed :', orderside_changed)
                    print()
                    continue
                else:
                    break

            #       reject unacceptable amount of asset     3
            if available_balance < self.min_balance:
                continue

            #       Todo        #
            #        " limit order 가 reduceOnly 인지 잘 확인해야함 "       #
            #        일단은, time_gap 을 두어, executed quantity 검사의 정확도를 높임      #
            #        예외 발생시, 2차 검사 phase 생성     #
            #        ==> 혹시, 여기도 time index 에 종속적인 걸까       #
            time.sleep(1)

            while 1:  # <-- for complete exec_open_quantity function
                try:
                    exec_open_quantity = get_remaining_quantity(config.init_set.symbol)
                except Exception as e:
                    print('error in exec_open_quantity check :', e)
                    continue
                break

            if exec_open_quantity == 0.0:  # <-- open order 가 체결되어야 position 이 생김

                income = 0
                # continue

            #       position / 체결량 이 존재하면 close order 진행      #
            else:

                print('open order executed')

                #        1. save trade_log      #
                #        2. check str(res_df.index[-2]), back_sync entry on close     #
                entry_timeindex = str(res_df.index[-2])
                trade_log[entry_timeindex] = [ep, open_side, "open"]

                with open(dir_path + "/trade_log/" + logger_name, "wb") as dict_f:
                    pickle.dump(trade_log, dict_f)
                    print("entry trade_log dumped !")

                print()

                #           set close side          #
                if open_side == OrderSide.BUY:
                    close_side = OrderSide.SELL
                else:
                    close_side = OrderSide.BUY

                #           market : prevent close at open bar           #
                #           Todo            #
                #            1. 더 정확히하려면, time_index confirmation 진행
                if config.ep_set.entry_type == OrderType.MARKET:
                    time.sleep(60 - datetime.now().second)

                # if config.tp_set.static_tp and config.out_set.static_out:
                #     load_new_df2 = 0    # dynamic out / tp 인 경우만 진행
                # else:
                load_new_df2 = 1
                limit_tp = 0
                remained_tp_id = None
                prev_remained_tp_id = None
                tp_exec_dict = {}

                while 1:

                    #        0. websocket 으로 load_new_df 가능할지 확인             #
                    #        1-1. load new_df in every new_df2 index change           #
                    #             -> 이전 index 와 비교하려면, 어차피 load_new_df 해야함      #
                    #        2. reorder tp 경우, enlist 된 기존 tp cancel 해야함
                    #        2-1. new_df check 동일하게 수행함

                    if not config.tp_set.static_tp or not config.out_set.static_out:
                        
                        if load_new_df2:    # dynamic_out & tp phase
    
                            try:
                                new_df_, _ = concat_candlestick(config.init_set.symbol, config.init_set.interval, days=1, limit=config.init_set.use_rows,
                                                                timesleep=0.2)
                                if datetime.timestamp(new_df_.index[-1]) < datetime.now().timestamp():
                                    continue
    
                                new_df2_, _ = concat_candlestick(config.init_set.symbol, config.init_set.interval2, days=1, limit=config.init_set.use_rows2,
                                                                 timesleep=0.2)
                                if datetime.timestamp(new_df2_.index[-1]) < datetime.now().timestamp():
                                    continue
    
                                new_df3_, _ = concat_candlestick(config.init_set.symbol, config.init_set.interval3, days=1, limit=config.init_set.use_rows3,
                                                                 timesleep=0.2)
                                if datetime.timestamp(new_df3_.index[-1]) < datetime.now().timestamp():
                                    continue
    
                                new_df = new_df_.iloc[-config.init_set.use_rows:].copy()
                                new_df2 = new_df2_.iloc[-config.init_set.use_rows2:].copy()
                                new_df3 = new_df3_.iloc[-config.init_set.use_rows3:].copy()
    
                                res_df = sync_check(new_df, new_df2, new_df3)

                                try:
                                    _, _, _, \
                                    short_out, long_out, short_tp, long_tp = enlist_epouttp(res_df, config.tp_set.tp_gap)
                                    print('res_df.index[-1] (load_new_df2 phase) :', res_df.index[-1])
                                    # print()

                                except Exception as e:
                                    print('error in enlist_epouttp :', e)
                                    continue

                                if open_side == OrderSide.BUY:
                                    out_series = short_out   # inversion mode adjusted
                                    tp_series = short_tp
                                else:
                                    out_series = long_out
                                    tp_series = long_tp

                                out = out_series.iloc[config.init_set.last_index]
                                tp = tp_series.iloc[config.init_set.last_index]

                            except Exception as e:
                                print('error in get new_dfs (load_new_df2 phase) :', e)
                                continue
    
                            else:
                                load_new_df2 = 0

                    # --------- limit tp order 여부 조사 phase --------- #
                    #         Todo          #
                    #          2. static_limit 은 reorder 할필요 없음
                    if config.tp_set.tp_type == OrderType.LIMIT:

                        prev_remained_tp_id = remained_tp_id

                        try:
                            remained_tp_id = remaining_order_check(config.init_set.symbol)

                        except Exception as e:
                            print('error in remaining_order_check :', e)
                            continue

                        if config.tp_set.static_tp:
                            if remained_tp_id is None:  # first tp_order
                                limit_tp = 1
                        else:
                            if remained_tp_id is None:  # first tp_order
                                limit_tp = 1
                            else:
                                #       tp 가 달라진 경우만 진행 = reorder     #
                                if tp_series.iloc[config.init_set.last_index] != tp_series.iloc[config.init_set.last_index - 1]:
                                    limit_tp = 1

                                    #           미체결 tp order 모두 cancel               #
                                    #           regardless to position exist, cancel all tp orders       #
                                    while 1:
                                        try:
                                            request_client.cancel_all_orders(symbol=config.init_set.symbol)
                                        except Exception as e:
                                            print('error in cancel remaining tp order :', e)
                                            continue
                                        else:
                                            break

                    # ------- limit_tp order phase ------- #
                    if limit_tp:

                        while 1:
                            #         get realtime price, volume precision       #
                            try:
                                price_precision, quantity_precision = get_precision(config.init_set.symbol)
                            except Exception as e:
                                print('error in get price & volume precision :', e)
                                continue
                            else:
                                print('price_precision :', price_precision)
                                print('quantity_precision :', quantity_precision)
                                break

                        #        1. prev_remained_tp_id 생성
                        #       Todo        #
                        #        2. (partial) tp_list 를 직접 만드는 방법 밖에 없을까 --> 일단은.
                        #        3. tp_list 는 ultimate_tp 부터 정렬        #
                        # tp_list = [tp, tp2]
                        # tp_series_list = [tp_series, tp2_series]
                        tp_list = [tp]

                        if len(tp_list) > 1:    # if partial_tp,

                            #       Todo        #
                            #        0. sub_tp 체결시마다 기록
                            #         a. 체결 confirm 은 len(remained_tp_id) 를 통해 진행
                            #        1. back_pr calculation
                            #        위해 체결된 시간 (res_df.index[-2]) 과 해당 tp 를 한번만 기록함        #

                            # --------- sub tp log phase --------- #
                            #       1. pop executed sub_tps from tp_list
                            #       2. log executed sub_tps
                            if remained_tp_id is not None and prev_remained_tp_id is not None:

                                # --- sub_tp execution --- #
                                if remained_tp_id != prev_remained_tp_id:

                                    # --- log sub_tp --- #
                                    executed_sub_tp = tp_series_list[-1].iloc[-2]
                                    executed_sub_prev_tp = tp_series_list[-1].iloc[-3]
                                    print("executed_sub_prev_tp, executed_sub_tp :",
                                          executed_sub_prev_tp, executed_sub_tp)
                                    tp_exec_dict[res_df.index[-2]] = [executed_sub_prev_tp, executed_sub_tp]

                                    # --- pop sub_tp --- #
                                    tp_list.pop()
                                    # tp_series_list.pop()
                                    print("tp_list.pop() executed")

                        tp_list = list(map(lambda x: calc_with_precision(x, price_precision), tp_list))
                        print('tp_list :', tp_list)

                        while 1:
                            try:
                                result = partial_limit(config.init_set.symbol, tp_list, close_side, quantity_precision,
                                                       config.tp_set.partial_qty_divider)

                                if result is not None:
                                    print(result)
                                    continue

                            except Exception as e:
                                print('error in partial_limit :', e)
                                time.sleep(1)
                                continue

                            else:
                                print("limit tp order enlisted :", datetime.now())
                                limit_tp = 0
                                break

                    # ---------- market close survey phase ---------- #
                    #            1. limit close (tp) execution check, every minute                 #
                    #            2. check market close signal, simultaneously
                    market_close_on = False
                    log_tp = np.nan
                    load_new_df3 = True
                    while 1:

                        # --------- limit tp execution check phase --------- #
                        if config.tp_set.tp_type == OrderType.LIMIT:
                            try:
                                tp_remain_quantity = get_remaining_quantity(config.init_set.symbol)
                            except Exception as e:
                                print('error in get_remaining_quantity :', e)
                                continue

                            #       들고 있는 position quantity 가 0 이면, 이 거래를 끝임        #
                            #        1. if tp_remain_quantity < 1 / (10 ** quantity_precision): 로 변경되어야할 것
                            # if tp_remain_quantity == 0.0:
                            if tp_remain_quantity <= 1 / (10 ** quantity_precision):
                                # if tp_remain_quantity < 1 / (10 ** quantity_precision):
                                print('all limit tp order executed')

                                #           Todo            #
                                #            1. market tp / out 에 대한 log 방식도 서술
                                #            2. dict 사용 이유 : partial_tp platform
                                if not config.tp_set.static_tp:
                                    tp_exec_dict[res_df.index[-2]] = [tp_series.iloc[config.init_set.last_index - 1], tp]
                                else:
                                    tp_exec_dict[res_df.index[-2]] = [tp]
                                print("tp_exec_dict :", tp_exec_dict)
                                break

                        # --------- market close signal check phase (out | tp) --------- #
                        #          1. 이부분에서 갱신된 res_df 1m data 필요함
                        #           a. confirmed_close 와 out / tp line 을 비교함
                        #          2. market_tp 조건은 tp_type 에 대한 조건을 명시해야함
                        #           a. out 은 market base 로 정의해서 조건문 필요없음
                        try:

                            #        1. 1m df 는 분당 1번만 진행
                            #         a. 그렇지 않을 경우 realtime_price latency 에 영향 줌
                            #        2. load_new_df2 사용하는 경우, load_new_df3 는 불필요함
                            #        3. tp 가 static_limit 이고, static_hl_out 이면 load_new_df3 마저 불필요함
                            #       Todo        #
                            #        back_pr wait_time 과 every minute end loop 의 new_df_ 를 위해 필요함
                            if config.tp_set.static_tp and config.out_set.static_out:

                                # if config.tp_set.tp_type == 'LIMIT' and config.out_set.hl_out:
                                #     pass
                                # else:

                                if load_new_df3:
                                    new_df_, _ = concat_candlestick(config.init_set.symbol, config.init_set.interval, days=1, limit=config.init_set.use_rows,
                                                                    timesleep=0.2)
                                    if datetime.timestamp(new_df_.index[-1]) < datetime.now().timestamp():
                                        continue

                                    load_new_df3 = False

                            #       exists for hl_out       #
                            if config.out_set.hl_out:
                                realtime_price = get_market_price_v2(self.sub_client)

                            # ---------- out & tp check ---------- #
                            if open_side == OrderSide.BUY:
                                #       long out (inversion of short)     #
                                if realtime_price <= out:  # hl_out
                                    market_close_on = True
                                    log_tp = out
                                    print("out :", out)

                                #       long tp (inversion of short)     #
                                #       if config.tp_set.tp_type == OrderType.MARKET,
                                if new_df_['close'].iloc[config.init_set.last_index] >= tp:
                                    market_close_on = True
                                    log_tp = tp
                                    print("tp :", tp)

                            else:
                                #       short out (inversion of long)     #
                                if realtime_price >= out:
                                    market_close_on = True
                                    log_tp = out
                                    print("out :", out)

                                #       short tp (inversion of long)     #
                                if new_df_['close'].iloc[config.init_set.last_index] <= tp:
                                    market_close_on = True
                                    log_tp = tp
                                    print("tp :", tp)

                            if market_close_on:
                                print("realtime_price :", realtime_price)
                                print("market_close_on is True")
                                # --- log tp --- #
                                tp_exec_dict[new_df_.index[-2]] = [log_tp]
                                print("tp_exec_dict :", tp_exec_dict)
                                break

                        except Exception as e:
                            print('error in checking market close signal :', e)
                            continue

                        # ------ minute end phase ------ #
                        if datetime.now().timestamp() > datetime.timestamp(new_df_.index[-1]):

                            if not config.tp_set.static_tp or not config.out_set.static_tp:
                                load_new_df2 = True
                                break
                            else:
                                # if config.tp_set.tp_type == 'LIMIT' and config.out_set.hl_out:
                                #     pass
                                # else:
                                load_new_df3 = True

                    if load_new_df2:    # new_df 와 dynamic limit_tp reorder 가 필요한 경우
                        continue

                    #   all limit close executed     #
                    if not market_close_on:
                        break

                    #               market close phase - get market signal               #
                    else:

                        remain_tp_canceled = False  # 미체결 limit tp 존재가능함
                        while 1:  # <--- This loop for out close & complete close

                            #               cancel all tp order                 #
                            if not remain_tp_canceled:
                                #               remaining tp order check            #
                                try:
                                    remained_orderId = remaining_order_check(config.init_set.symbol)

                                except Exception as e:
                                    print('error in remaining_order_check :', e)
                                    continue

                                if remained_orderId is not None:
                                    #           if remained tp order exist, cancel it       #
                                    try:
                                        request_client.cancel_all_orders(symbol=config.init_set.symbol)

                                    except Exception as e:
                                        print('error in cancel remaining tp order (out phase) :', e)
                                        continue

                                remain_tp_canceled = True

                            #          get remaining close_remain_quantity         #
                            try:
                                close_remain_quantity = get_remaining_quantity(config.init_set.symbol)

                            except Exception as e:
                                print('error in get_remaining_quantity :', e)
                                continue

                            #           get price, volume precision             #
                            #           -> reatlime 가격 변동으로 인한 precision 변동 가능성       #
                            try:
                                _, quantity_precision = get_precision(config.init_set.symbol)

                            except Exception as e:
                                print('error in get price & volume precision :', e)

                            else:
                                print('quantity_precision :', quantity_precision)

                            close_remain_quantity = calc_with_precision(close_remain_quantity, quantity_precision)
                            print('close_remain_quantity :', close_remain_quantity)

                            #           close remaining close_remain_quantity             #
                            #           1. order side should be opposite side to the open          #
                            #           2. reduceOnly = 'true'       #
                            code_1111 = 0
                            code_2022 = 0
                            while 1:  # <-- loop for complete close order
                                try:
                                    #       Todo        #
                                    #        1. limit out 은 한동안 없을 것
                                    if config.out_set.out_type != OrderType.MARKET:

                                        #           stop limit order            #
                                        # if close_side == OrderSide.SELL:
                                        #     stop_price = calc_with_precision(realtime_price + 5 * 10 ** -price_precision,
                                        #                                      price_precision)
                                        # else:
                                        #     stop_price = calc_with_precision(realtime_price - 5 * 10 ** -price_precision,
                                        #                                      price_precision)

                                        # request_client.post_order(timeInForce=TimeInForce.GTC, symbol=config.init_set.symbol,
                                        #                           side=close_side,
                                        #                           ordertype=config.out_set.out_type, stopPrice=str(stop_price),
                                        #                           quantity=str(close_remain_quantity), price=str(realtime_price),
                                        #                           reduceOnly=True)

                                        if pd.isna(out):
                                            exit_price = realtime_price
                                        else:
                                            exit_price = log_tp

                                        #           limit order             #
                                        request_client.post_order(timeInForce=TimeInForce.GTC, symbol=config.init_set.symbol,
                                                                  side=close_side,
                                                                  ordertype=config.out_set.out_type,
                                                                  quantity=str(close_remain_quantity), price=str(exit_price),
                                                                  reduceOnly=True)

                                    else:
                                        #           market order            #
                                        request_client.post_order(symbol=config.init_set.symbol, side=close_side,
                                                                           ordertype=OrderType.MARKET,
                                                                           quantity=str(close_remain_quantity),
                                                                           reduceOnly=True)

                                except Exception as e:
                                    print('error in close order :', e)

                                    #       check error codes     #
                                    #       Todo        #
                                    #       -2022 ReduceOnly Order is rejected      #
                                    if '-2022' in str(e):
                                        code_2022 = 1
                                        break

                                    #        -4003 quantity less than zero ?
                                    if '-4003' in str(e):
                                        print('error code check (-4003) :', str(e))
                                        break

                                    #        -1111 : Precision is over the maximum defined for this asset   #
                                    #         = quantity precision error        #
                                    if '-1111' in str(e):
                                        code_1111 += 1
                                        try:
                                            _, quantity_precision = get_precision(config.init_set.symbol)
                                            if code_1111 % 3 == 0:
                                                quantity_precision -= 1
                                            elif code_1111 % 3 == 2:
                                                quantity_precision += 1

                                            close_remain_quantity = get_remaining_quantity(config.init_set.symbol)
                                            close_remain_quantity = calc_with_precision(close_remain_quantity, quantity_precision,
                                                                           def_type='floor')
                                            print('modified qty & precision :', close_remain_quantity, quantity_precision)

                                        except Exception as e:
                                            print('error in get modified qty_precision :', e)

                                        continue

                                    config.out_set.out_type = OrderType.MARKET   # just 재시도
                                    continue

                                else:
                                    print('market close order enlisted')
                                    break

                            #       enough time for close_remain_quantity to be consumed      #
                            if config.out_set.out_type != OrderType.MARKET:

                                #       wait for bar ends      #
                                time.sleep(config.out_set.exit_execution_wait - datetime.now().second)
                                print("config.out_set.exit_execution_wait - datetime.now().second :",
                                      config.out_set.exit_execution_wait - datetime.now().second)
                                print("datetime.now().second :", datetime.now().second)

                            else:
                                time.sleep(1)   # time for qty consumed

                            # #               check remaining close_remain_quantity             #
                            # try:
                            #     close_remain_quantity = get_remaining_quantity(config.init_set.symbol)
                            # except Exception as e:
                            #     print('error in get_remaining_quantity :', e)
                            #     continue

                            if close_remain_quantity == 0.0 or code_2022:
                                print('market close order executed')
                                break

                            else:
                                #           complete close by market            #
                                print('out_type changed to market')
                                config.out_set.out_type = OrderType.MARKET
                                continue

                        break  # <--- break for all close order loop, break partial tp loop

                # ----------------------- check back_pr ----------------------- #
                #       total_income() function confirming -> wait close confirm       #
                while 1:

                    # if config.tp_set.tp_type == OrderType.LIMIT:
                    #     lastest_close_timeindex = res_df.index[-1]
                    # else:
                    lastest_close_timeindex = new_df_.index[-1]

                    if datetime.now().timestamp() > datetime.timestamp(lastest_close_timeindex):
                        # try:
                        #     back_df, _ = concat_candlestick(config.init_set.symbol, config.init_set.interval, days=1, timesleep=0.2)
                        #
                        #     #       realtime candlestick confirmation       #
                        #     if datetime.timestamp(back_df.index[-1]) > datetime.now().timestamp():
                        #         break
                        #
                        # except Exception as e:
                        #     print('error in back-test profit :', e)

                        break

                # ---------- back_pr calculation ---------- #
                #            1. adjust partial tp             #
                #            2. -1 이 아니라, timestamp index 로 접근해야할듯      #
                #            2-1. len(tp_list) 가 2 -> 1 로 변한 순간을 catch, time_idx 를 기록
                #            3. 마지막 체결 정보를 기록 : (res_df.index[-1]) 과 해당 tp 를 기록        #

                #             real_tp, division           #
                #             prev_tp == tp 인 경우는, tp > open 인 경우에도 real_tp 는 tp 와 동일함      #
                calc_tmp_profit = 1
                r_qty = 1   # base asset ratio
                # ---------- calc in for loop ---------- #
                for q_i, (k_ts, v_tp) in enumerate(sorted(tp_exec_dict.items(), key=lambda x: x[0], reverse=True)):

                    # ---------- get real_tp phase ---------- #
                    if len(v_tp) > 1:
                        prev_tp, tp = v_tp

                        if prev_tp == tp:
                            real_tp = tp

                        else:   # dynamic_tp 로 인한 open tp case
                            if close_side == OrderSide.BUY:
                                if tp > res_df['open'].loc[k_ts]:
                                    real_tp = res_df['open'].loc[k_ts]
                                    print("market tp executed !")
                                else:
                                    real_tp = tp
                            else:
                                if tp < res_df['open'].loc[k_ts]:
                                    real_tp = res_df['open'].loc[k_ts]
                                    print("market tp executed !")
                                else:
                                    real_tp = tp

                    else:   # market close's real_tp | out (partial_tp also have market out)
                        [real_tp] = v_tp

                    #       check partial_tp case        #
                    if len(tp_exec_dict) == 1:
                        temp_qty = r_qty
                    else:
                        # if q_i != 0:
                        if q_i != len(tp_exec_dict) - 1:
                            temp_qty = r_qty / config.tp_set.partial_qty_divider
                        else:
                            temp_qty = r_qty

                    r_qty -= temp_qty

                    if open_side == OrderSide.SELL:
                        calc_tmp_profit += (ep / real_tp - self.trading_fee - 1) * temp_qty
                        # calc_tmp_profit += ep / real_tp - self.trading_fee
                    else:
                        calc_tmp_profit += (real_tp / ep - self.trading_fee - 1) * temp_qty

                    print("real_tp :", real_tp)
                    print("ep :", ep)
                    print("self.trading_fee :", self.trading_fee)
                    print("temp_qty :", temp_qty)

                    #            save exit data         #
                    #            exit_timeindex use "-1" index           #
                    # exit_timeindex = str(res_df.index[-1])
                    # trade_log[exit_timeindex] = [real_tp, "close"]
                    trade_log[k_ts] = [real_tp, "close"]

                with open(dir_path + "/trade_log/" + logger_name, "wb") as dict_f:
                    pickle.dump(trade_log, dict_f)
                    print("exit trade_log dumped !")

                end_timestamp = int(time.time() * 1000)

                #           get total income from this trade       #
                while 1:
                    try:
                        income = total_income(config.init_set.symbol, start_timestamp, end_timestamp)
                        self.accumulated_income += income

                    except Exception as e:
                        print('error in total_income :', e)

                    else:
                        break

                tmp_profit = income / real_balance
                self.accumulated_profit *= (1 + tmp_profit)
                self.calc_accumulated_profit *= 1 + (calc_tmp_profit - 1) * leverage
                print('temporary profit : %.3f (%.3f) %%' % (tmp_profit * 100, (calc_tmp_profit - 1) * leverage * 100))
                print('accumulated profit : %.3f (%.3f) %%' % (
                    (self.accumulated_profit - 1) * 100, (self.calc_accumulated_profit - 1) * 100))
                print('accumulated income :', self.accumulated_income, 'USDT')
                print()

from binance_f.model.constant import OrderSide
from funcs.funcs_indicator_candlescore import *
from funcs.funcs_trader import *

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 2500)
pd.set_option('display.max_columns', 2500)


def sync_check(res_df_list):

    df, fourth_df, _ = res_df_list

    #        1. 필요한 indi. 는 enlist_epouttp & mr_check 보면서 삽입
    #       Todo        #
    #        2. htf use_rows 는 1m use_rows 의 길이를 만족시킬 수 있는 정도
    #         a. 1m use_rows / htf_interval 하면 대략 나옴
    #         b. 또한, htf indi. 를 생성하기 위해 필요한 최소 row 이상

    df = st_price_line(df, fourth_df, '15m')
    df = st_level(df, '15m', 0.5)

    #        sma60         #
    # df['sma_1m'] = df['close'].rolling(60).mean()

    #        cloud bline         #
    # fourth_df['cloud_bline_15m'] = cloud_bline(fourth_df, 26)
    # df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, fourth_df, [-1]), columns=['cloud_bline_15m']))

    return df


def enlist_epouttp(res_df, config):

    res_df['entry'] = np.zeros(len(res_df))

    np_timeidx = np.array(list(map(lambda x: intmin(x), res_df.index)))

    # res_df['entry'] = np.where((np_timeidx % config.ep_set.htf_entry == 0)
    res_df['entry'] = np.where((np_timeidx % config.ep_set.htf_entry == config.ep_set.htf_entry - 1) # 실제 open time 에는 이게 맞음
                               , res_df['entry'] + 1, res_df['entry'])

    # ----- market entry ----- #
    res_df['short_ep'] = np.nan
    res_df['long_ep'] = res_df['open'] - res_df['st_gap_15m'] * config.ep_set.ep_gap

    # --------------- st level consolidation out --------------- #
    res_df['short_out'] = np.nan
    res_df['long_out'] = np.nan

    # --------------- st level tp --------------- #
    res_df['short_tp'] = np.nan
    res_df['long_tp'] = np.nan

    return res_df


def mr_check(res_df, config):

    #       Todo        #
    #        1. check words i_ --> to last_index
    #        2. if short const. copied to long, check reversion
    open_side = None
    mr_const_cnt = 0
    mr_score = 0

    if config.ep_set.entry_type == 'MARKET':
        if config.tp_set.tp_type == 'LIMIT':
            tp_fee = config.init_set.market_fee + config.init_set.limit_fee
        else:
            tp_fee = config.init_set.market_fee + config.init_set.market_fee
        out_fee = config.init_set.market_fee + config.init_set.market_fee
    else:
        if config.tp_set.tp_type == 'LIMIT':
            tp_fee = config.init_set.limit_fee + config.init_set.limit_fee
        else:
            tp_fee = config.init_set.limit_fee + config.init_set.market_fee
        out_fee = config.init_set.limit_fee + config.init_set.market_fee

    # ---------------------- short ---------------------- #

    #        1. cbline const. compare with close, so 'close' data should be confirmed
    #        2. or, entry market_price would be replaced with 'close'

    # ---------------------- long ---------------------- #
    if res_df['entry'][config.init_set.last_index] == -config.ep_set.short_entry_score:

        if mr_const_cnt == mr_score:
            open_side = OrderSide.BUY

    return open_side


if __name__ == '__main__':

    # os.chdir("./..")
    # print()

    from funcs_binance.binance_futures_concat_candlestick import concat_candlestick

    days = 1
    symbol = 'ETHUSDT'
    interval = '1m'
    interval2 = '3m'

    use_rows = 200
    use_rows2 = 100

    gap = 0.00005

    # # print(calc_train_days('1h', 3000))
    # df = pd.read_excel('../candlestick_concated/%s/%s %s.xlsx' % (interval, date, symbol), index_col=0)
    # # print(df.head())
    # print(len(df))

    # new_df_, _ = concat_candlestick(symbol, interval, days=1, timesleep=0.2)
    # new_df2_, _ = concat_candlestick(symbol, interval2, days=1, timesleep=0.2)
    #
    # new_df = new_df_.iloc[-use_rows:].copy()
    # new_df2 = new_df2_.iloc[-use_rows2:].copy()
    # res_df = sync_check(new_df, new_df2)
    #
    # tp = res_df['middle_line'].iloc[-1] * (1 + gap)
    #
    # print("tp :", tp)

    # prev_tp = tp

    # if open_side == OrderSide.SELL:
    #     tp = res_df['middle_line'].iloc[self.last_index] * (1 + self.gap)
    # else:
    #     tp = res_df['middle_line'].iloc[self.last_index] * (1 - self.gap)

    # import time
    #
    # s_time = time.time()
    # #       tp_update       #
    # print(tp_update(df, plotting=False, save_path="test.png"))
    # print("elapsed time :", time.time() - s_time)

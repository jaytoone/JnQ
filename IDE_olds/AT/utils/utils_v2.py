from binance_f.model.constant import OrderSide
from funcs.olds.funcs_indicator_candlescore import *
from funcs.funcs_trader import *

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 2500)
pd.set_option('display.max_columns', 2500)


def sync_check(res_df_list):

    df, third_df, _ = res_df_list

    #       Todo : manual        #
    #        1. 필요한 indi. 는 enlist_epouttp & mr_check 보면서 삽입
    #        2. htf use_rows 는 1m use_rows 의 길이를 만족시킬 수 있는 정도
    #         a. 1m use_rows / htf_interval 하면 대략 나옴
    #         b. 또한, htf indi. 를 생성하기 위해 필요한 최소 row 이상

    df = bb_line(df, third_df, '5m')
    df = bb_level(df, '5m', 1)

    df = dc_line(df, third_df, '5m')
    df = dc_level(df, '5m', 1)

    return df


def enlist_rtc(res_df, config):

    res_df['short_rtc_1'] = res_df['bb_lower_%s' % config.tp_set.tpg_itv]
    res_df['short_rtc_0'] = res_df['dc_upper_%s' % config.out_set.outg_itv]
    res_df['short_rtc_gap'] = res_df['short_rtc_0'] - res_df['short_rtc_1']

    res_df['h_short_rtc_1'] = res_df['bb_lower_%s' % config.tp_set.tpg_itv]
    res_df['h_short_rtc_0'] = res_df['dc_upper_%s' % config.tp_set.tpg_itv]
    res_df['h_short_rtc_gap'] = res_df['h_short_rtc_0'] - res_df['h_short_rtc_1']

    res_df['long_rtc_1'] = res_df['bb_upper_%s' % config.tp_set.tpg_itv]
    res_df['long_rtc_0'] = res_df['dc_lower_%s' % config.out_set.outg_itv]
    res_df['long_rtc_gap'] = res_df['long_rtc_1'] - res_df['long_rtc_0']

    res_df['h_long_rtc_1'] = res_df['bb_upper_%s' % config.tp_set.tpg_itv]
    res_df['h_long_rtc_0'] = res_df['dc_lower_%s' % config.tp_set.tpg_itv]
    res_df['h_long_rtc_gap'] = res_df['h_long_rtc_1'] - res_df['h_long_rtc_0']

    res_df['short_dtk_1'] = res_df['bb_lower_%s' % config.ep_set.dtk_itv]
    res_df['short_dtk_0'] = res_df['dc_upper_%s' % config.ep_set.dtk_itv]
    res_df['short_dtk_gap'] = res_df['short_dtk_0'] - res_df['short_dtk_1']

    res_df['long_dtk_1'] = res_df['bb_upper_%s' % config.ep_set.dtk_itv]
    res_df['long_dtk_0'] = res_df['dc_lower_%s' % config.ep_set.dtk_itv]
    res_df['long_dtk_gap'] = res_df['long_dtk_1'] - res_df['long_dtk_0']

    return res_df


def enlist_tr(res_df, config):

    res_df = enlist_rtc(res_df, config)

    np_timeidx = np.array(list(map(lambda x: intmin(x), res_df.index)))

    res_df['entry'] = np.zeros(len(res_df))
    res_df['h_entry'] = np.zeros(len(res_df))

    res_df['short_ep'] = res_df['h_short_rtc_1'] + res_df['h_short_rtc_gap'] * config.ep_set.ep_gap
    res_df['long_ep'] = res_df['h_long_rtc_1'] - res_df['h_long_rtc_gap'] * config.ep_set.ep_gap

    #       market ver.     #
    if config.ep_set.entry_type == "MARKET":
      res_df['short_ep'] = res_df['close']
      res_df['long_ep'] = res_df['close']

    # ---------------------------------------- short = -1 ---------------------------------------- #
    # res_df['entry'] = np.where((res_df['close'].shift(config.ep_set.tf_entry * 1) >= res_df['bb_lower_5m']) &
    # res_df['entry'] = np.where((res_df['open'] >= res_df['bb_lower_5m']) &
    #                 # (res_df['close'].shift(config.ep_set.tf_entry * 1) <= res_df['bb_lower_5m']) &
    #                 (res_df['close'] < res_df['bb_lower_5m']) &
    #                 (np_timeidx % config.ep_set.tf_entry == (config.ep_set.tf_entry - 1))
    #                 , res_df['entry'] - 1, res_df['entry'])
    #
    # res_df['h_entry'] = np.where(#(res_df['open'] >= res_df['bb_lower_%s' % config.ep_set.dtk_itv]) &
    #                 (res_df['close'].shift(config.ep_set.htf_entry * 1) >= res_df['bb_lower_%s' % config.ep_set.dtk_itv]) &
    #                 (res_df['close'] < res_df['bb_lower_%s' % config.ep_set.dtk_itv]) &
    #                 (np_timeidx % config.ep_set.htf_entry == (config.ep_set.htf_entry - 1))
    #                 , res_df['h_entry'] - 1, res_df['h_entry'])

    # ---------------------------------------- long = 1 ---------------------------------------- #
    # res_df['entry'] = np.where((res_df['close'].shift(config.ep_set.tf_entry * 1) <= res_df['bb_upper_5m']) &
    res_df['entry'] = np.where((res_df['open'] <= res_df['bb_upper_5m']) &
                               # (res_df['close'].shift(config.ep_set.tf_entry * 1) >= res_df['bb_upper_5m']) &
                               (res_df['close'] > res_df['bb_upper_5m']) &
                               (np_timeidx % config.ep_set.tf_entry == (config.ep_set.tf_entry - 1))
                               , res_df['entry'] + 1, res_df['entry'])

    res_df['h_entry'] = np.where(  # (res_df['open'] <= res_df['bb_upper_%s' % config.ep_set.dtk_itv]) &
        (res_df['close'].shift(config.ep_set.htf_entry * 1) <= res_df['bb_upper_%s' % config.ep_set.dtk_itv]) &
        (res_df['close'] > res_df['bb_upper_%s' % config.ep_set.dtk_itv]) &
        (np_timeidx % config.ep_set.htf_entry == (config.ep_set.htf_entry - 1))
        , res_df['h_entry'] + 1, res_df['h_entry'])

    # ------------------------------ rtc tp & out ------------------------------ #
    # --------------- bb rtc out --------------- #
    res_df['short_out'] = res_df['short_rtc_0'] + res_df['short_rtc_gap'] * config.out_set.out_gap
    res_df['long_out'] = res_df['long_rtc_0'] - res_df['long_rtc_gap'] * config.out_set.out_gap

    # ------------------------------ tp ------------------------------ #
    # --------------- bb rtc tp --------------- #
    res_df['short_tp'] = res_df['h_short_rtc_1'] - res_df['h_short_rtc_gap'] * config.tp_set.tp_gap
    res_df['long_tp'] = res_df['h_long_rtc_1'] + res_df['h_long_rtc_gap'] * config.tp_set.tp_gap

    if config.ep_set.use_dtk_line:
        res_df['short_dtk_1'] = np.where(res_df['h_entry'] == -1, res_df['short_dtk_1'], np.nan)
        res_df['short_dtk_1'] = ffill(res_df['short_dtk_1'].values.reshape(1, -1)).reshape(-1, 1)
        res_df['short_dtk_gap'] = np.where(res_df['h_entry'] == -1, res_df['short_dtk_gap'], np.nan)
        res_df['short_dtk_gap'] = ffill(res_df['short_dtk_gap'].values.reshape(1, -1)).reshape(-1, 1)

        res_df['long_dtk_1'] = np.where(res_df['h_entry'] == 1, res_df['long_dtk_1'], np.nan)
        res_df['long_dtk_1'] = ffill(res_df['long_dtk_1'].values.reshape(1, -1)).reshape(-1, 1)
        res_df['long_dtk_gap'] = np.where(res_df['h_entry'] == 1, res_df['long_dtk_gap'], np.nan)
        res_df['long_dtk_gap'] = ffill(res_df['long_dtk_gap'].values.reshape(1, -1)).reshape(-1, 1)

    res_df['dc_upper_v2'] = res_df['high'].rolling(config.ep_set.dc_period * config.ep_set.dtk_dc_itv_num).max()
    res_df['dc_lower_v2'] = res_df['low'].rolling(config.ep_set.dc_period * config.ep_set.dtk_dc_itv_num).min()

    return res_df


def mr_check(res_df, config):

    #       Todo        #
    #        1. check words i_ --> to last_index
    #        2. if short const. copied to long, check reversion error
    open_side = None
    mr_const_cnt = 0
    mr_score = 0
    i = config.init_set.last_index

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
    if res_df['entry'][config.init_set.last_index] == config.ep_set.short_entry_score:

        # -------------- tr scheduling -------------- #
        if config.ep_set.tr_thresh != "None":
            mr_const_cnt += 1
            if (res_df['short_ep'].iloc[i] - res_df['short_tp'].iloc[i] - tp_fee * res_df['short_ep'].iloc[i]) / (
                    res_df['short_out'].iloc[i] - res_df['short_ep'].iloc[i] + out_fee * res_df['short_ep'].iloc[
                i]) >= config.ep_set.tr_thresh:
                mr_score += 1

        # -------------- dtk -------------- #
        if config.ep_set.dt_k != "None":
            mr_const_cnt += 1
            #     dc_v2   #
            if res_df['dc_lower_v2'].iloc[i] >= res_df['short_dtk_1'].iloc[i] - res_df['short_dtk_gap'].iloc[
                i] * config.ep_set.dt_k:

                mr_score += 1

                # -------------- bb_zone  -------------- #
        if config.ep_set.bbz_itv != "None":
            mr_const_cnt += 1
            if res_df['close'].iloc[i] < res_df['bb_lower_%s' % config.ep_set.bbz_itv].iloc[i]:
                mr_score += 1

        if mr_const_cnt == mr_score:
                open_side = OrderSide.SELL

    # ---------------------- long ---------------------- #
    if res_df['entry'][config.init_set.last_index] == -config.ep_set.short_entry_score:

        # -------------- tr scheduling -------------- #
        if config.ep_set.tr_thresh != "None":
            # print('config.ep_set.tr_thresh != "None" :', config.ep_set.tr_thresh != "None")
            # print('config.ep_set.tr_thresh :', config.ep_set.tr_thresh)
            mr_const_cnt += 1
            if (res_df['long_tp'].iloc[i] - res_df['long_ep'].iloc[i] - tp_fee * res_df['long_ep'].iloc[i]) / (
                    res_df['long_ep'].iloc[i] - res_df['long_out'].iloc[i] + out_fee * res_df['long_ep'].iloc[
                i]) >= config.ep_set.tr_thresh:
                mr_score += 1

        # -------------- dtk -------------- #
        if config.ep_set.dt_k != "None":

            mr_const_cnt += 1
            #     dc_v2     #
            if res_df['dc_upper_v2'].iloc[i] <= res_df['long_dtk_1'].iloc[i] + res_df['long_dtk_gap'].iloc[
                i] * config.ep_set.dt_k:

                mr_score += 1

                # -------------- bb_zone  -------------- #
        if config.ep_set.bbz_itv != "None":

            mr_const_cnt += 1
            if res_df['close'].iloc[i] > res_df['bb_upper_%s' % config.ep_set.bbz_itv].iloc[i]:
                mr_score += 1

        if mr_const_cnt == mr_score:
            open_side = OrderSide.BUY

    return open_side


def lvrg_set(res_df, config, limit_leverage, open_side, fee):

    #       Todo        #
    #        dynamic 으로 변경될 경우 추가 logic 필요함
    ep_j = config.init_set.last_index   # <-- init. 이기 때문에 last_index 사용해도 무방함

    if not config.lvrg_set.static_lvrg:

        if open_side == OrderSide.SELL:
            config.lvrg_set.leverage = config.lvrg_set.target_pct / (
                        res_df['short_out'].iloc[ep_j] / res_df['short_ep'].iloc[ep_j] - 1 - (
                            fee + config.init_set.market_fee))
        else:
            config.lvrg_set.leverage = config.lvrg_set.target_pct / (
                        res_df['long_ep'].iloc[ep_j] / res_df['long_out'].iloc[ep_j] - 1 - (
                            fee + config.init_set.market_fee))

        if not config.lvrg_set.allow_float:
            config.lvrg_set.leverage = int(config.lvrg_set.leverage)

        # -------------- leverage rejection -------------- #
        if config.lvrg_set.leverage < 1 and config.lvrg_set.lvrg_rejection:
            return None

        config.lvrg_set.leverage = max(config.lvrg_set.leverage, 1)

    config.lvrg_set.leverage = min(limit_leverage, config.lvrg_set.leverage)

    return config.lvrg_set.leverage


if __name__ == '__main__':

    os.chdir("./..")
    # print()

    days = 1
    symbol = 'ETHUSDT'
    interval = '1m'
    interval2 = '3m'

    use_rows = 200
    use_rows2 = 100

    gap = 0.00005

    # # print(calc_train_days('1h', 3000))
    # df = pd.read_excel('../database/%s/%s %s.xlsx' % (interval, date, symbol), index_col=0)
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

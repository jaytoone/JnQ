from binance_f.model.constant import OrderSide
from funcs.funcs_indicator import *
from funcs.funcs_trader import *

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 2500)
pd.set_option('display.max_columns', 2500)


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


def sync_check(res_df_list):

    df, third_df, fourth_df = res_df_list

    #       add indi. only      #

    #       Todo : manual        #
    #        1. 필요한 indi. 는 enlist_epouttp & mr_check 보면서 삽입
    #        2. htf use_rows 는 1m use_rows 의 길이를 만족시킬 수 있는 정도
    #         a. 1m use_rows / htf_interval 하면 대략 나옴
    #         b. 또한, htf indi. 를 생성하기 위해 필요한 최소 row 이상

    df = bb_line(df, third_df, '5m')
    df = dc_line(df, third_df, '5m')
    df = bb_line(df, fourth_df, '15m')
    df = dc_line(df, fourth_df, '15m')

    return df


def enlist_rtc(res_df, config):
    res_df = bb_level(res_df, '5m', 1)
    res_df = dc_level(res_df, '5m', 1)
    res_df = bb_level(res_df, '15m', 1)
    res_df = dc_level(res_df, '15m', 1)

    res_df['short_rtc_1'] = res_df['bb_lower_%s' % config.loc_set.tpg_itv]
    res_df['short_rtc_0'] = res_df['dc_upper_%s' % config.loc_set.outg_itv]
    res_df['short_rtc_gap'] = res_df['short_rtc_0'] - res_df['short_rtc_1']

    res_df['h_short_rtc_1'] = res_df['bb_lower_%s' % config.loc_set.tpg_itv]
    res_df['h_short_rtc_0'] = res_df['dc_upper_%s' % config.loc_set.tpg_itv]
    res_df['h_short_rtc_gap'] = res_df['h_short_rtc_0'] - res_df['h_short_rtc_1']

    res_df['long_rtc_1'] = res_df['bb_upper_%s' % config.loc_set.tpg_itv]
    res_df['long_rtc_0'] = res_df['dc_lower_%s' % config.loc_set.outg_itv]
    res_df['long_rtc_gap'] = res_df['long_rtc_1'] - res_df['long_rtc_0']

    res_df['h_long_rtc_1'] = res_df['bb_upper_%s' % config.loc_set.tpg_itv]
    res_df['h_long_rtc_0'] = res_df['dc_lower_%s' % config.loc_set.tpg_itv]
    res_df['h_long_rtc_gap'] = res_df['h_long_rtc_1'] - res_df['h_long_rtc_0']

    res_df['short_dtk_1'] = res_df['bb_lower_%s' % config.loc_set.dtk_itv]
    res_df['short_dtk_0'] = res_df['dc_upper_%s' % config.loc_set.dtk_itv]
    res_df['short_dtk_gap'] = res_df['short_dtk_0'] - res_df['short_dtk_1']

    res_df['long_dtk_1'] = res_df['bb_upper_%s' % config.loc_set.dtk_itv]
    res_df['long_dtk_0'] = res_df['dc_lower_%s' % config.loc_set.dtk_itv]
    res_df['long_dtk_gap'] = res_df['long_dtk_1'] - res_df['long_dtk_0']

    res_df = dtk_plot(res_df, dtk_itv2='15m', hhtf_entry=15, use_dtk_line=config.loc_set.use_dtk_line)

    return res_df


def enlist_tr(res_df, config, np_timeidx):

    res_df['entry'] = np.zeros(len(res_df))
    res_df['h_entry'] = np.zeros(len(res_df))

    # -------- set ep level -------- #

    #       limit ver.     #
    #        bb ep         #
    res_df['short_ep'] = res_df['h_short_rtc_1'] + res_df['h_short_rtc_gap'] * config.tr_set.ep_gap
    res_df['long_ep'] = res_df['h_long_rtc_1'] - res_df['h_long_rtc_gap'] * config.tr_set.ep_gap

    if config.tr_set.c_ep_gap != "None":
        res_df['short_ep_org'] = res_df['short_ep'].copy()
        res_df['long_ep_org'] = res_df['long_ep'].copy()

        res_df['short_ep2'] = res_df['h_short_rtc_1'] + res_df['h_short_rtc_gap'] * config.tr_set.c_ep_gap
        res_df['long_ep2'] = res_df['h_long_rtc_1'] - res_df['h_long_rtc_gap'] * config.tr_set.c_ep_gap

    res_df['short_tr_ep'] = res_df['h_short_rtc_1'] + res_df['h_short_rtc_gap'] * config.loc_set.spread_ep_gap
    res_df['long_tr_ep'] = res_df['h_long_rtc_1'] - res_df['h_long_rtc_gap'] * config.loc_set.spread_ep_gap

    #       market ver.     #
    if config.ep_set.entry_type == "MARKET":
        res_df['short_ep'] = res_df['close']
        res_df['long_ep'] = res_df['close']

    # ---------------------------------------- short = -1 ---------------------------------------- #
    # ---------------- ep_time  ---------------- #
    #        bb level entry      #
    # res_df['entry'] = np.where((res_df['close'].shift(config.ep_set.tf_entry * 1) >= res_df['bb_lower_%s' % config.loc_set.exp_itv]) &
    res_df['entry'] = np.where((res_df['open'] >= res_df['bb_lower_%s' % config.loc_set.exp_itv]) &
                               # (res_df['close'].shift(config.ep_set.tf_entry * 1) >= res_df['bb_lower_%s' % config.loc_set.exp_itv]) &
                               (res_df['close'] < res_df['bb_lower_%s' % config.loc_set.exp_itv])
                               , res_df['entry'] - 1, res_df['entry'])

    res_df['entry'] = np.where((res_df['entry'] < 0) &
                               (np_timeidx % config.ep_set.tf_entry == (config.ep_set.tf_entry - 1))
                               , res_df['entry'] - 1, res_df['entry'])

    # res_df['entry'] = np.where((res_df['open'] >= res_df['cloud_bline_%s' % cb_itv]) &
    #                 # (res_df['close'].shift(config.ep_set.tf_entry * 1) <= res_df['cloud_bline_%s' % cb_itv]) &
    #                 (res_df['close'] < res_df['cloud_bline_%s' % cb_itv])
    #                 , res_df['entry'] - 1, res_df['entry'])

    # res_df['entry'] = np.where((res_df['open'] >= res_df['bb_lower_1m']) &
    #                 # (res_df['close'].shift(config.ep_set.tf_entry * 1) <= res_df['bb_lower_1m']) &
    #                 (res_df['close'] < res_df['bb_lower_1m'])
    #                 , res_df['entry'] - 1, res_df['entry'])

    res_df['h_entry'] = np.where(  # (res_df['open'] >= res_df['bb_lower_%s' % config.loc_set.dtk_itv]) &
        (res_df['close'].shift(config.ep_set.htf_entry * 1) >= res_df['bb_lower_%s' % config.loc_set.dtk_itv]) &
        (res_df['close'] < res_df['bb_lower_%s' % config.loc_set.dtk_itv]) &
        (np_timeidx % config.ep_set.htf_entry == (config.ep_set.htf_entry - 1))
        , res_df['h_entry'] - 1, res_df['h_entry'])

    # ---------------------------------------- long = 1 ---------------------------------------- #
    # ---------------------- ep_time ---------------------- #

    #      bb level entry      #
    # res_df['entry'] = np.where((res_df['close'].shift(config.ep_set.tf_entry * 1) <= res_df['bb_upper_%s' % config.loc_set.exp_itv]) &
    res_df['entry'] = np.where((res_df['open'] <= res_df['bb_upper_%s' % config.loc_set.exp_itv]) &
                               # (res_df['close'].shift(config.ep_set.tf_entry * 1) <= res_df['bb_upper_%s' % config.loc_set.exp_itv]) &
                               (res_df['close'] > res_df['bb_upper_%s' % config.loc_set.exp_itv])
                               , res_df['entry'] + 1, res_df['entry'])

    res_df['entry'] = np.where((res_df['entry'] > 0) &
                               (np_timeidx % config.ep_set.tf_entry == (config.ep_set.tf_entry - 1))
                               , res_df['entry'] + 1, res_df['entry'])

    # res_df['entry'] = np.where((res_df['open'] <= res_df['cloud_bline_%s' % cb_itv]) &
    #                   # (res_df['close'].shift(config.ep_set.tf_entry * 1) >= res_df['cloud_bline_%s' % cb_itv]) &
    #                   (res_df['close'] > res_df['cloud_bline_%s' % cb_itv])
    #                 , res_df['entry'] + 1, res_df['entry'])

    # res_df['entry'] = np.where((res_df['open'] <= res_df['bb_upper_1m']) &
    #                   # (res_df['close'].shift(config.ep_set.tf_entry * 1) >= res_df['bb_upper_1m']) &
    #                   (res_df['close'] > res_df['bb_upper_1m'])
    #                 , res_df['entry'] + 1, res_df['entry'])

    res_df['h_entry'] = np.where(  # (res_df['open'] <= res_df['bb_upper_%s' % config.loc_set.dtk_itv]) &
        (res_df['close'].shift(config.ep_set.htf_entry * 1) <= res_df['bb_upper_%s' % config.loc_set.dtk_itv]) &
        (res_df['close'] > res_df['bb_upper_%s' % config.loc_set.dtk_itv]) &
        (np_timeidx % config.ep_set.htf_entry == (config.ep_set.htf_entry - 1))
        , res_df['h_entry'] + 1, res_df['h_entry'])

    # ------------------------------ rtc tp & out ------------------------------ #
    # --------------- bb rtc out --------------- #

    if config.loc_set.outg_dc_period != "None":
        res_df['short_rtc_0'] = res_df['high'].rolling(config.loc_set.outg_dc_period).max()
        res_df['long_rtc_0'] = res_df['low'].rolling(config.loc_set.outg_dc_period).min()

    res_df['short_out'] = res_df['short_rtc_0'] + res_df['short_rtc_gap'] * config.tr_set.out_gap
    res_df['long_out'] = res_df['long_rtc_0'] - res_df['long_rtc_gap'] * config.tr_set.out_gap

    if config.tr_set.t_out_gap != "None":
        res_df['short_out_org'] = res_df['short_out'].copy()
        res_df['long_out_org'] = res_df['long_out'].copy()

        res_df['short_out2'] = res_df['short_rtc_0'] + res_df['short_rtc_gap'] * config.tr_set.t_out_gap
        res_df['long_out2'] = res_df['long_rtc_0'] - res_df['long_rtc_gap'] * config.tr_set.t_out_gap

    # ------------------------------ tp ------------------------------ #
    # --------------- bb rtc tp --------------- #
    res_df['short_tp'] = res_df['h_short_rtc_1'] - res_df['h_short_rtc_gap'] * config.tr_set.tp_gap
    res_df['long_tp'] = res_df['h_long_rtc_1'] + res_df['h_long_rtc_gap'] * config.tr_set.tp_gap

    # --------------- set tp_line / dtk_line --------------- #
    # res_df['short_tp_1'] = np.where(res_df['h_entry'] == -1, res_df['short_rtc_1'], np.nan)
    # res_df['short_tp_1'] = ffill(res_df['short_tp_1'].values.reshape(1, -1)).reshape(-1, 1)
    # res_df['short_tp_gap'] = np.where(res_df['h_entry'] == -1, res_df['h_short_rtc_gap'], np.nan)  # ltf_gap 은 out 을 위한 gap 임
    # res_df['short_tp_gap'] = ffill(res_df['short_tp_gap'].values.reshape(1, -1)).reshape(-1, 1)

    # res_df['long_tp_1'] = np.where(res_df['h_entry'] == 1, res_df['long_rtc_1'], np.nan)
    # res_df['long_tp_1'] = ffill(res_df['long_tp_1'].values.reshape(1, -1)).reshape(-1, 1)
    # res_df['long_tp_gap'] = np.where(res_df['h_entry'] == 1, res_df['h_long_rtc_gap'], np.nan)
    # res_df['long_tp_gap'] = ffill(res_df['long_tp_gap'].values.reshape(1, -1)).reshape(-1, 1)

    # res_df['h_short_tp_1'] = np.where(res_df['h_entry'] == -1, res_df['h_short_rtc_1'], np.nan)
    # res_df['h_short_tp_1'] = ffill(res_df['h_short_tp_1'].values.reshape(1, -1)).reshape(-1, 1)
    # res_df['h_short_tp_gap'] = np.where(res_df['h_entry'] == -1, res_df['h_short_rtc_gap'], np.nan)
    # res_df['h_short_tp_gap'] = ffill(res_df['h_short_tp_gap'].values.reshape(1, -1)).reshape(-1, 1)

    # res_df['h_long_tp_1'] = np.where(res_df['h_entry'] == 1, res_df['h_long_rtc_1'], np.nan)
    # res_df['h_long_tp_1'] = ffill(res_df['h_long_tp_1'].values.reshape(1, -1)).reshape(-1, 1)
    # res_df['h_long_tp_gap'] = np.where(res_df['h_entry'] == 1, res_df['h_long_rtc_gap'], np.nan)
    # res_df['h_long_tp_gap'] = ffill(res_df['h_long_tp_gap'].values.reshape(1, -1)).reshape(-1, 1)

    if config.loc_set.use_dtk_line:
        res_df['short_dtk_1'] = np.where(res_df['h_entry'] == -1, res_df['short_dtk_1'], np.nan)
        res_df['short_dtk_1'] = ffill(res_df['short_dtk_1'].values.reshape(1, -1)).reshape(-1, 1)
        res_df['short_dtk_gap'] = np.where(res_df['h_entry'] == -1, res_df['short_dtk_gap'], np.nan)
        res_df['short_dtk_gap'] = ffill(res_df['short_dtk_gap'].values.reshape(1, -1)).reshape(-1, 1)

        res_df['long_dtk_1'] = np.where(res_df['h_entry'] == 1, res_df['long_dtk_1'], np.nan)
        res_df['long_dtk_1'] = ffill(res_df['long_dtk_1'].values.reshape(1, -1)).reshape(-1, 1)
        res_df['long_dtk_gap'] = np.where(res_df['h_entry'] == 1, res_df['long_dtk_gap'], np.nan)
        res_df['long_dtk_gap'] = ffill(res_df['long_dtk_gap'].values.reshape(1, -1)).reshape(-1, 1)

    res_df['dc_upper_v2'] = res_df['high'].rolling(config.loc_set.dc_period).max()
    res_df['dc_lower_v2'] = res_df['low'].rolling(config.loc_set.dc_period).min()

    res_df['zone_dc_upper_v2'] = res_df['high'].rolling(config.loc_set.zone_dc_period).max()
    res_df['zone_dc_lower_v2'] = res_df['low'].rolling(config.loc_set.zone_dc_period).min()

    return res_df


def short_ep_loc(res_df, config, i, show_detail=True):
    # ------- param init ------- #
    open_side = None

    mr_const_cnt = 0
    mr_score = 0
    zone = 'n'

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

    # -------------- tr scheduling -------------- #
    if config.loc_set.spread != "None":

        mr_const_cnt += 1

        if config.loc_set.spread_ep_gap != "None":
            tr = (res_df['short_tr_ep'].iloc[i] - res_df['short_tp'].iloc[i] - tp_fee * res_df['short_tr_ep'].iloc[
                i]) / (res_df['short_out'].iloc[i] - res_df['short_tr_ep'].iloc[i] + out_fee *
                       res_df['short_tr_ep'].iloc[i])
            if tr >= config.loc_set.spread:
                mr_score += 1

        else:
            tr = (res_df['short_ep'].iloc[i] - res_df['short_tp'].iloc[i] - tp_fee * res_df['short_ep'].iloc[i]) / (
                    res_df['short_out'].iloc[i] - res_df['short_ep'].iloc[i] + out_fee * res_df['short_ep'].iloc[i])
            if tr >= config.loc_set.spread:
                # if (res_df['short_ep'].iloc[i] - res_df['short_tp'].iloc[i] - tp_fee * res_df['short_ep'].iloc[i]) / (res_df['short_out'].iloc[i] - res_df['short_ep'].iloc[i] + tp_fee * res_df['short_ep'].iloc[i]) >= config.loc_set.spread:
                # if (res_df['short_ep'].iloc[i] - res_df['short_tp'].iloc[i] - tp_fee * res_df['short_ep'].iloc[i]) / (res_df['short_out'].iloc[i] - res_df['short_ep'].iloc[i] + out_fee * res_df['short_ep'].iloc[i]) == config.loc_set.spread:
                # if (res_df['short_ep'].iloc[i] - res_df['short_tp'].iloc[i] - out_fee * res_df['short_ep'].iloc[i]) / (res_df['short_out'].iloc[i] - res_df['short_ep'].iloc[i] + out_fee * res_df['short_ep'].iloc[i]) >= config.loc_set.spread:
                mr_score += 1

        if show_detail:
            print("tr :", tr)

    # -------------- dtk -------------- #
    if config.loc_set.dt_k != "None":

        mr_const_cnt += 1
        # if res_df['dc_lower_%s' % config.loc_set.dtk_dc_itv].iloc[i] >= res_df['short_rtc_1'].iloc[i] - res_df['h_short_rtc_gap'].iloc[i] * config.loc_set.dt_k:
        #     dtk_v1 & v2 platform     #
        if config.loc_set.dtk_dc_itv != "None":
            dc = res_df['dc_lower_%s' % config.loc_set.dtk_dc_itv].iloc[i]
            dt_k = res_df['short_dtk_1'].iloc[i] - res_df['short_dtk_gap'].iloc[i] * config.loc_set.dt_k
            if dc >= dt_k:
                mr_score += 1

                #     dc_v2   #
        else:
            dc = res_df['dc_lower_v2'].iloc[i]
            dt_k = res_df['short_dtk_1'].iloc[i] - res_df['short_dtk_gap'].iloc[i] * config.loc_set.dt_k
            if dc >= dt_k:
                # if res_df['dc_lower_v2'].iloc[i] >= res_df['short_dtk_1'].iloc[i] - res_df['short_dtk_gap'].iloc[i] * config.loc_set.dt_k and \
                # res_df['dc_upper_v2'].iloc[i] <= res_df['long_dtk_1'].iloc[i] + res_df['long_dtk_gap'].iloc[i] * config.loc_set.dt_k:

                mr_score += 1

        if show_detail:
            print("dc :", dc)
            print("dt_k :", dt_k)

    # -------------- zone rejection  -------------- #
    # if config.loc_set.bbz_itv != "None":
    #   mr_const_cnt += 1

    #     #       by bb       #
    #   # if res_df['close'].iloc[i] < res_df['bb_lower_%s' % config.loc_set.bbz_itv].iloc[i]:   # org
    #   # if res_df['close'].iloc[i] > res_df['bb_lower_%s' % config.loc_set.bbz_itv].iloc[i]:  # inv
    #   # if res_df['close'].iloc[i] > res_df['bb_upper_%s' % config.loc_set.bbz_itv].iloc[i]:
    #   # if res_df['close'].iloc[i] > res_df['bb_lower2_%s' % config.loc_set.bbz_itv].iloc[i]:

    #     #       by zone_dtk       #
    #   if res_df['zone_dc_upper_v2'].iloc[i] < res_df['long_dtk_plot_1'].iloc[i] + res_df['long_dtk_plot_gap'].iloc[
    #       i] * config.loc_set.zone_dt_k:

    #     mr_score += 1

    # -------------- zoning -------------- #
    if config.tr_set.c_ep_gap != "None":

        #       by bb       #
        # if res_df['close'].iloc[i] > res_df['bb_lower_%s' % config.loc_set.bbz_itv].iloc[i]:

        #       by zone_dtk       #

        #         c_zone        #
        if res_df['zone_dc_upper_v2'].iloc[i] > res_df['long_dtk_plot_1'].iloc[i] + res_df['long_dtk_plot_gap'].iloc[
            i] * config.loc_set.zone_dt_k:

            if config.ep_set.static_ep:
                res_df['short_ep'].iloc[i] = res_df['short_ep2'].iloc[i]
            else:
                res_df['short_ep'] = res_df['short_ep2']

            if config.out_set.static_out:
                res_df['short_out'].iloc[i] = res_df['short_out_org'].iloc[i]
            else:
                res_df['short_out'] = res_df['short_out_org']

            zone = 'c'

        #         t_zone        #
        else:
            if config.ep_set.static_ep:
                res_df['short_ep'].iloc[i] = res_df['short_ep_org'].iloc[i]
            else:
                res_df['short_ep'] = res_df['short_ep_org']

            if config.out_set.static_out:
                res_df['short_out'].iloc[i] = res_df['short_out2'].iloc[i]
            else:
                res_df['short_out'] = res_df['short_out2']

            zone = 't'

    if mr_const_cnt == mr_score:
        open_side = OrderSide.SELL

    return res_df, open_side, zone


def long_ep_loc(res_df, config, i, show_detail=True):
    # ------- param init ------- #
    open_side = None

    mr_const_cnt = 0
    mr_score = 0
    zone = 'n'

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

    # -------------- tr scheduling -------------- #
    if config.loc_set.spread != "None":

        mr_const_cnt += 1

        if config.loc_set.spread_ep_gap != "None":
            tr = (res_df['long_tp'].iloc[i] - res_df['long_tr_ep'].iloc[i] - tp_fee * res_df['long_tr_ep'].iloc[i]) / (
                    res_df['long_tr_ep'].iloc[i] - res_df['long_out'].iloc[i] + out_fee * res_df['long_tr_ep'].iloc[i])
            if tr >= config.loc_set.spread:
                mr_score += 1

        else:
            tr = (res_df['long_tp'].iloc[i] - res_df['long_ep'].iloc[i] - tp_fee * res_df['long_ep'].iloc[i]) / (
                    res_df['long_ep'].iloc[i] - res_df['long_out'].iloc[i] + out_fee * res_df['long_ep'].iloc[i])
            if tr >= config.loc_set.spread:
                # if (res_df['long_tp'].iloc[i] - res_df['long_ep'].iloc[i] - tp_fee * res_df['long_ep'].iloc[i]) / (res_df['long_ep'].iloc[i] - res_df['long_out'].iloc[i] + tp_fee * res_df['long_ep'].iloc[i]) >= config.loc_set.spread:
                # if (res_df['long_tp'].iloc[i] - res_df['long_ep'].iloc[i] - tp_fee * res_df['long_ep'].iloc[i]) / (res_df['long_ep'].iloc[i] - res_df['long_out'].iloc[i] + out_fee * res_df['long_ep'].iloc[i]) == config.loc_set.spread:
                # if (res_df['long_tp'].iloc[i] - res_df['long_ep'].iloc[i] - out_fee * res_df['long_ep'].iloc[i]) / (res_df['long_ep'].iloc[i] - res_df['long_out'].iloc[i] + out_fee * res_df['long_ep'].iloc[i]) >= config.loc_set.spread:
                mr_score += 1

        if show_detail:
            print("tr :", tr)

    # -------------- dtk -------------- #
    if config.loc_set.dt_k != "None":

        mr_const_cnt += 1
        # if res_df['dc_upper_%s' % config.loc_set.dtk_dc_itv].iloc[i] <= res_df['long_rtc_1'].iloc[i] + res_df['long_rtc_gap'].iloc[i] * config.loc_set.dt_k:
        #     dtk_v1 & v2 platform    #
        if config.loc_set.dtk_dc_itv != "None":
            dc = res_df['dc_upper_%s' % config.loc_set.dtk_dc_itv].iloc[i]
            dt_k = res_df['long_dtk_1'].iloc[i] + res_df['long_dtk_gap'].iloc[i] * config.loc_set.dt_k
            if dc <= dt_k:
                mr_score += 1

        else:
            #     dc_v2     #
            dc = res_df['dc_upper_v2'].iloc[i]
            dt_k = res_df['long_dtk_1'].iloc[i] + res_df['long_dtk_gap'].iloc[i] * config.loc_set.dt_k
            if dc <= dt_k:
                # if res_df['dc_upper_v2'].iloc[i] >= res_df['long_dtk_1'].iloc[i] + res_df['long_dtk_gap'].iloc[i] * config.loc_set.dt_k:

                # if res_df['dc_upper_v2'].iloc[i] <= res_df['long_dtk_1'].iloc[i] + res_df['long_dtk_gap'].iloc[i] * config.loc_set.dt_k and \
                #   res_df['dc_lower_v2'].iloc[i] >= res_df['short_dtk_1'].iloc[i] - res_df['short_dtk_gap'].iloc[i] * config.loc_set.dt_k:

                mr_score += 1

        if show_detail:
            print("dc :", dc)
            print("dt_k :", dt_k)

    # -------------- zone rejection  -------------- #
    # if config.loc_set.bbz_itv != "None":
    #   mr_const_cnt += 1

    #     #       by bb       #
    #   # if res_df['close'].iloc[i] > res_df['bb_upper_%s' % config.loc_set.bbz_itv].iloc[i]:    # org
    #   # if res_df['close'].iloc[i] < res_df['bb_upper_%s' % config.loc_set.bbz_itv].iloc[i]:  # inv
    #   # if res_df['close'].iloc[i] < res_df['bb_lower_%s' % config.loc_set.bbz_itv].iloc[i]:
    #   # if res_df['close'].iloc[i] < res_df['bb_upper2_%s' % config.loc_set.bbz_itv].iloc[i]:

    #     #       by zone_dtk       #
    #   if res_df['zone_dc_lower_v2'].iloc[i] > res_df['short_dtk_plot_1'].iloc[i] - res_df['short_dtk_plot_gap'].iloc[i] * config.loc_set.zone_dt_k:

    #     mr_score += 1

    # -------------- zoning -------------- #
    if config.tr_set.c_ep_gap != "None":
        #       by bb       #
        # if res_df['close'].iloc[i] < res_df['bb_upper_%s' % config.loc_set.bbz_itv].iloc[i]:

        #       by zone_dtk       #

        #         c_zone        #
        if res_df['zone_dc_lower_v2'].iloc[i] < res_df['short_dtk_plot_1'].iloc[i] - res_df['short_dtk_plot_gap'].iloc[
            i] * config.loc_set.zone_dt_k:

            if config.ep_set.static_ep:
                res_df['long_ep'].iloc[i] = res_df['long_ep2'].iloc[i]
            else:
                res_df['long_ep'] = res_df['long_ep2']

            if config.out_set.static_out:
                res_df['long_out'].iloc[i] = res_df['long_out_org'].iloc[i]
            else:
                res_df['long_out'] = res_df['long_out_org']

            zone = 'c'

            # mr_const_cnt += 1
            # dc_lb_period = 100
            # if np.sum((res_df['dc_upper_15m'] > res_df['dc_upper_15m'].shift(15)).iloc[i - dc_lb_period:i]) == 0:
            #   mr_score += 1

            #         t_zone        #
        else:

            if config.ep_set.static_ep:
                res_df['long_ep'].iloc[i] = res_df['long_ep_org'].iloc[i]
            else:
                res_df['long_ep'] = res_df['long_ep_org']

            if config.out_set.static_out:
                res_df['long_out'].iloc[i] = res_df['long_out2'].iloc[i]
            else:
                res_df['long_out'] = res_df['long_out2']

            zone = 't'

    if mr_const_cnt == mr_score:
        open_side = OrderSide.BUY

    return res_df, open_side, zone
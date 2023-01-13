import pandas as pd
import numpy as np

def get_line(touch_idx, rtc_):
    touch_idx_copy = touch_idx.copy()

    nan_idx = np.isnan(touch_idx_copy)
    touch_idx_copy[nan_idx] = 0  # for indexing array
    touch_line = rtc_[touch_idx_copy.astype(int)].copy()
    touch_line[nan_idx] = np.nan  # for true comp.

    return touch_line


def enlist_rtc(res_df, config, np_timeidx):
    strat_version = config.strat_version
    # ------------ rtc_gap ------------ #
    short_epout_1_, long_epout_1_ = 'short_epout_1_{}'.format(strat_version), 'long_epout_1_{}'.format(strat_version)
    short_epout_0_, long_epout_0_ = 'short_epout_0_{}'.format(strat_version), 'long_epout_0_{}'.format(strat_version)
    short_tp_1_, long_tp_1_ = 'short_tp_1_{}'.format(strat_version), 'long_tp_1_{}'.format(strat_version)
    short_tp_0_, long_tp_0_ = 'short_tp_0_{}'.format(strat_version), 'long_tp_0_{}'.format(strat_version)

    # b1_itv_num = to_itvnum(config.loc_set.point.p2_itv0)
    # b2_itv_num = to_itvnum(config.loc_set.point.p2_itv0) * 2  # multi 2 for imb_v2

    p1_period1 = config.loc_set.point.p1_period1
    p1_period2 = config.loc_set.point.p1_period2
    res_df[short_tp_1_] = res_df['dc_lower_{}{}'.format(config.loc_set.point.p1_itv1, p1_period1)]
    res_df[short_tp_0_] = res_df['dc_upper_{}{}'.format(config.loc_set.point.p1_itv0, p1_period2)]
    res_df[long_tp_1_] = res_df['dc_upper_{}{}'.format(config.loc_set.point.p1_itv1, p1_period1)]
    res_df[long_tp_0_] = res_df['dc_lower_{}{}'.format(config.loc_set.point.p1_itv0, p1_period2)]

    if config.loc_set.point.p2_itv1 != "None":
        p2_period1 = config.loc_set.point.p2_period1
        p2_period2 = config.loc_set.point.p2_period2
        res_df[short_epout_1_] = res_df['dc_lower_{}{}'.format(config.loc_set.point.p2_itv1, p2_period1)]
        res_df[short_epout_0_] = res_df['dc_upper_{}{}'.format(config.loc_set.point.p2_itv0, p2_period2)]
        res_df[long_epout_1_] = res_df['dc_upper_{}{}'.format(config.loc_set.point.p2_itv1, p2_period1)]
        res_df[long_epout_0_] = res_df['dc_lower_{}{}'.format(config.loc_set.point.p2_itv0, p2_period2)]
    else:
        res_df[short_epout_1_] = res_df['dc_lower_{}{}'.format(config.loc_set.point.p1_itv1, p1_period1)]
        res_df[short_epout_0_] = res_df['dc_upper_{}{}'.format(config.loc_set.point.p1_itv0, p1_period2)]
        res_df[long_epout_1_] = res_df['dc_upper_{}{}'.format(config.loc_set.point.p1_itv1, p1_period1)]
        res_df[long_epout_0_] = res_df['dc_lower_{}{}'.format(config.loc_set.point.p1_itv0, p1_period2)]

    # ------ inversion ------ #
    if config.pos_set.short_inversion or config.pos_set.long_inversion:
        res_df.rename({short_tp_1_: long_tp_1_, long_tp_1_: short_tp_1_}, axis=1, inplace=True)
        res_df.rename({short_tp_0_: long_tp_0_, long_tp_0_: short_tp_0_}, axis=1, inplace=True)
        res_df.rename({short_epout_1_: long_epout_1_, long_epout_1_: short_epout_1_}, axis=1, inplace=True)
        res_df.rename({short_epout_0_: long_epout_0_, long_epout_0_: short_epout_0_}, axis=1, inplace=True)

    res_df['short_tp_gap_{}'.format(strat_version)] = abs(res_df[short_tp_1_] - res_df[short_tp_0_])
    res_df['long_tp_gap_{}'.format(strat_version)] = abs(res_df[long_tp_1_] - res_df[long_tp_0_])
    res_df['short_epout_gap_{}'.format(strat_version)] = abs(res_df[short_epout_1_] - res_df[short_epout_0_])
    res_df['long_epout_gap_{}'.format(strat_version)] = abs(res_df[long_epout_1_] - res_df[long_epout_0_])

    # ------------ dtk_gap ------------ #
    # res_df['short_dtk_1_{}'.format(strat_version)] = res_df['bb_lower_%s' % config.loc_set.zone.dtk_itv]
    # res_df['short_dtk_0_{}'.format(strat_version)] = res_df['dc_upper_%s' % config.loc_set.zone.dtk_itv]
    # res_df['long_dtk_1_{}'.format(strat_version)] = res_df['bb_upper_%s' % config.loc_set.zone.dtk_itv]
    # res_df['long_dtk_0_{}'.format(strat_version)] = res_df['dc_lower_%s' % config.loc_set.zone.dtk_itv]

    # res_df['short_dtk_gap_{}'.format(strat_version)] = abs(
    #     res_df['short_dtk_0_{}'.format(strat_version)] - res_df['short_dtk_1_{}'.format(strat_version)])
    # res_df['long_dtk_gap_{}'.format(strat_version)] = abs(
    #     res_df['long_dtk_1_{}'.format(strat_version)] - res_df['long_dtk_0_{}'.format(strat_version)])

    return res_df


def enlist_tr(res_df, config, np_timeidx, mode='OPEN'):
    strat_version = config.strat_version
    len_df = len(res_df)
    short_open_res = np.ones(len_df)
    long_open_res = np.ones(len_df)

    short_tp_1_col, short_tp_0_col, short_tp_gap_col = 'short_tp_1_{}'.format(strat_version), 'short_tp_0_{}'.format(
        strat_version), 'short_tp_gap_{}'.format(strat_version)
    long_tp_1_col, long_tp_0_col, long_tp_gap_col = 'long_tp_1_{}'.format(strat_version), 'long_tp_0_{}'.format(
        strat_version), 'long_tp_gap_{}'.format(strat_version)
    short_epout_1_col, short_epout_0_col, short_epout_gap_col = 'short_epout_1_{}'.format(strat_version), 'short_epout_0_{}'.format(
        strat_version), 'short_epout_gap_{}'.format(strat_version)
    long_epout_1_col, long_epout_0_col, long_epout_gap_col = 'long_epout_1_{}'.format(strat_version), 'long_epout_0_{}'.format(
        strat_version), 'long_epout_gap_{}'.format(strat_version)

    tp_cols = [short_tp_1_col, short_tp_0_col, short_tp_gap_col, long_tp_1_col, long_tp_0_col, long_tp_gap_col]  # Todo - public_indi 이전에 해야할지도 모름
    epout_cols = [short_epout_1_col, short_epout_0_col, short_epout_gap_col, long_epout_1_col, long_epout_0_col,
                  long_epout_gap_col]  # Todo - public_indi 이전에 해야할지도 모름
    data_cols = ['open', 'high', 'low']  # Todo - public_indi 이전에 해야할지도 모름 # 'close', 'haopen', 'hahigh', 'halow', 'haclose'

    short_tp_1_, short_tp_0_, short_tp_gap_, long_tp_1_, long_tp_0_, long_tp_gap_ = [res_df[col_].to_numpy() for col_ in tp_cols]
    short_epout_1_, short_epout_0_, short_epout_gap_, long_epout_1_, long_epout_0_, long_epout_gap_ = [res_df[col_].to_numpy() for col_ in epout_cols]
    open, high, low = [res_df[col_].to_numpy() for col_ in data_cols]

    # ---------------- point - support_confirmer---------------- #
    point1_to2_period = 60
    p1_itv1 = config.loc_set.point.p1_itv1
    p1_period1 = config.loc_set.point.p1_period1
    p1_period2 = config.loc_set.point.p1_period2
    p2_itv1 = config.loc_set.point.p2_itv1
    p2_period1 = config.loc_set.point.p2_period1
    p2_period2 = config.loc_set.point.p2_period2

    if p2_itv1 != "None":
        short_point1_on2_idx = pd.Series(
            np.where(res_df['short_wave_point_{}{}{}'.format(p1_itv1, p1_period1, p1_period2)], np.arange(len_df), np.nan)).rolling(point1_to2_period,
                                                                                                                                    min_periods=1).max().to_numpy()  # period 내의 max_point1_idx
        long_point1_on2_idx = pd.Series(
            np.where(res_df['long_wave_point_{}{}{}'.format(p1_itv1, p1_period1, p1_period2)], np.arange(len_df), np.nan)).rolling(point1_to2_period,
                                                                                                                                   min_periods=1).max().to_numpy()

        short_point2_idx = pd.Series(
            np.where(res_df['short_wave_point_{}{}{}'.format(p2_itv1, p2_period1, p2_period2)], np.arange(len_df), np.nan)).to_numpy()
        long_point2_idx = pd.Series(
            np.where(res_df['long_wave_point_{}{}{}'.format(p2_itv1, p2_period1, p2_period2)], np.arange(len_df), np.nan)).to_numpy()

        res_df['short_point_idxgap_{}'.format(strat_version)] = short_point2_idx - short_point1_on2_idx
        res_df['long_point_idxgap_{}'.format(strat_version)] = long_point2_idx - long_point1_on2_idx

        # ------ p1 & p2 ------ #
        short_open_res *= ~np.isnan(res_df['short_point_idxgap_{}'.format(strat_version)].to_numpy())
        long_open_res *= ~np.isnan(res_df['long_point_idxgap_{}'.format(strat_version)].to_numpy())

        # print(np.sum(long_open_res == 1))

        # ------ p2 amax > p1_idx (long) ------ #
        short_open_res *= res_df['short_a_touch_idx_{}{}{}'.format(p2_itv1, p2_period1, p2_period2)].to_numpy() > short_point1_on2_idx
        long_open_res *= res_df['long_a_touch_idx_{}{}{}'.format(p2_itv1, p2_period1, p2_period2)].to_numpy() > long_point1_on2_idx

        # print(np.sum(long_open_res == 1))

        # ------ higher low (long) ------ #
        # short_a_line1_on2_ = get_line(short_point1_on2_idx, res_df['short_a_line_{}{}{}'.format(p1_itv1, p1_period1, p1_period2)].to_numpy())
        # long_a_line1_on2_ = get_line(long_point1_on2_idx, res_df['long_a_line_{}{}{}'.format(p1_itv1, p1_period1, p1_period2)].to_numpy())

        # short_a_line2_ = res_df['short_a_line_{}{}{}'.format(p2_itv1, p2_period1, p2_period2)].to_numpy()
        # long_a_line2_ = res_df['long_a_line_{}{}{}'.format(p2_itv1, p2_period1, p2_period2)].to_numpy()

        # short_open_res *= short_a_line1_on2_ >= short_a_line2_
        # long_open_res *= long_a_line1_on2_ <= long_a_line2_

        # print(np.sum(long_open_res == 1))

    else:  # ------ p1 only ------ #
        res_df['short_point_idxgap_{}'.format(strat_version)] = 0  # default
        res_df['long_point_idxgap_{}'.format(strat_version)] = 0

        short_open_res *= res_df['short_wave_point_{}{}{}'.format(p1_itv1, p1_period1, p1_period2)].to_numpy().astype(bool)
        long_open_res *= res_df['long_wave_point_{}{}{}'.format(p1_itv1, p1_period1, p1_period2)].to_numpy().astype(bool)

        # print(np.sum(long_open_res == 1))

    res_df['short_open_{}'.format(strat_version)] = short_open_res
    res_df['long_open_{}'.format(strat_version)] = long_open_res

    # ------------------ tr_set ------------------ #
    # ------------ tpep ------------ #
    tpg = config.tr_set.tp_gap
    res_df['short_tp_{}'.format(strat_version)] = short_tp_1_ - short_tp_gap_ * tpg
    res_df['long_tp_{}'.format(strat_version)] = long_tp_1_ + long_tp_gap_ * tpg

    # ------ limit_ep ------ #
    if config.ep_set.entry_type == "LIMIT":
        epg = -0.5 + config.tr_set.ep_gap
        res_df['short_ep_{}'.format(strat_version)] = short_epout_0_ + short_epout_gap_ * epg
        res_df['long_ep_{}'.format(strat_version)] = long_epout_0_ - long_epout_gap_ * epg

    # ------ market_ep ------ #
    else:
        res_df['short_ep_{}'.format(strat_version)] = res_df['close']
        res_df['long_ep_{}'.format(strat_version)] = res_df['close']

    # ------------ out ------------ #
    outg = config.tr_set.out_gap
    res_df['short_out_{}'.format(strat_version)] = short_epout_0_ + short_epout_gap_ * outg
    res_df['long_out_{}'.format(strat_version)] = long_epout_0_ - long_epout_gap_ * outg

    # ------------ point validation ------------ #
    short_tp_ = res_df['short_tp_{}'.format(strat_version)].to_numpy()
    short_ep_ = res_df['short_ep_{}'.format(strat_version)].to_numpy()
    short_out_ = res_df['short_out_{}'.format(strat_version)].to_numpy()
    short_open_res *= (short_tp_ < short_ep_) & (short_ep_ < short_out_)
    res_df['short_open_{}'.format(strat_version)] = short_open_res

    long_tp_ = res_df['long_tp_{}'.format(strat_version)].to_numpy()
    long_ep_ = res_df['long_ep_{}'.format(strat_version)].to_numpy()
    long_out_ = res_df['long_out_{}'.format(strat_version)].to_numpy()
    long_open_res *= (long_tp_ > long_ep_) & (long_ep_ > long_out_)
    res_df['long_open_{}'.format(strat_version)] = long_open_res

    # print(np.sum(long_open_res == 1))

    # ------ tr ------ #
    res_df['short_tr_{}'.format(strat_version)] = abs(
        (short_ep_ / short_tp_ - config.trader_set.limit_fee - 1) / (short_ep_ / short_out_ - config.trader_set.market_fee - 1))
    res_df['long_tr_{}'.format(strat_version)] = abs(
        (long_tp_ / long_ep_ - config.trader_set.limit_fee - 1) / (long_out_ / long_ep_ - config.trader_set.market_fee - 1))

    # ------ zoned_ep ------ #
    if config.tr_set.c_ep_gap != "None":
        # res_df['short_ep_org_{}'.format(strat_version)] = res_df['short_ep_{}'.format(strat_version)].copy()
        # res_df['long_ep_org_{}'.format(strat_version)] = res_df['long_ep_{}'.format(strat_version)].copy()
        res_df['short_ep2_{}'.format(strat_version)] = short_epout_1_ + short_epout_gap_ * config.tr_set.c_ep_gap
        res_df['long_ep2_{}'.format(strat_version)] = long_epout_1_ - long_epout_gap_ * config.tr_set.c_ep_gap

    # ------ zoned_out ------ #
    if config.tr_set.t_out_gap != "None":
        # res_df['short_out_org_{}'.format(strat_version)] = res_df['short_out_{}'.format(strat_version)].copy()
        # res_df['long_out_org_{}'.format(strat_version)] = res_df['long_out_{}'.format(strat_version)].copy()
        res_df['short_out2_{}'.format(strat_version)] = res_df['short_epout_0_{}'.format(strat_version)] + res_df[
            'short_epout_gap_{}'.format(strat_version)] * config.tr_set.t_out_gap
        res_df['long_out2_{}'.format(strat_version)] = res_df['long_epout_0_{}'.format(strat_version)] - res_df[
            'long_epout_gap_{}'.format(strat_version)] * config.tr_set.t_out_gap

    # ------------ bias ------------ #  Todo - 추후, bias_set 독립시키는게 좋지 않을까
    bias_info_tick = config.tr_set.bias_info_tick
    # ------ bias_info can use future_data ------ #
    # res_df['short_bias_info_{}'.format(strat_version)] = res_df['dc_lower_T'].shift(-bias_info_tick)  # open / ep_tick 으로부터 bias_info_tick 만큼
    # res_df['long_bias_info_{}'.format(strat_version)] = res_df['dc_upper_T'].shift(-bias_info_tick)
    res_df['short_bias_info_{}'.format(strat_version)] = res_df['low'].rolling(bias_info_tick).min().shift(-bias_info_tick)
    res_df['long_bias_info_{}'.format(strat_version)] = res_df['high'].rolling(bias_info_tick).max().shift(-bias_info_tick)

    # bias_thresh 는 결국 tp 가 될 것
    res_df['short_bias_thresh_{}'.format(strat_version)] = res_df['short_tp_{}'.format(strat_version)]
    res_df['long_bias_thresh_{}'.format(strat_version)] = res_df['long_tp_{}'.format(strat_version)]
    # res_df['short_bias_thresh_{}'.format(strat_version)] = res_df['dc_lower_T'] - res_df['short_tp_gap_{}'.format(strat_version)] * config.tr_set.bias_gap
    # res_df['long_bias_thresh_{}'.format(strat_version)] = res_df['dc_upper_T'] + res_df['long_tp_gap_{}'.format(strat_version)] * config.tr_set.bias_gap

    return res_df

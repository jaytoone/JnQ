from funcs.funcs_indicator import *
import numpy as np

#       indi. making 이 이곳에서 이루어져서는 안됨 -> period 계산 안됨       #


def enlist_rtc(res_df, config, np_timeidx):

    strat_version = config.strat_version

    res_df['short_rtc_1_{}'.format(strat_version)] = res_df['dc_lower_%s' % config.loc_set.point.outg_itv1]
    res_df['short_rtc_0_{}'.format(strat_version)] = res_df['bb_upper_%s' % config.loc_set.point.outg_itv0]

    res_df['long_rtc_1_{}'.format(strat_version)] = res_df['bb_upper_%s' % config.loc_set.point.outg_itv1]
    res_df['long_rtc_0_{}'.format(strat_version)] = res_df['dc_lower_%s' % config.loc_set.point.outg_itv0]

    #      entry reversion    #
    if config.ep_set.short_entry_score > 0:
        short_rtc_1_copy = res_df['short_rtc_1_{}'.format(strat_version)].copy()
        res_df['short_rtc_1_{}'.format(strat_version)] = res_df['long_rtc_1_{}'.format(strat_version)]
        res_df['long_rtc_1_{}'.format(strat_version)] = short_rtc_1_copy

        short_rtc_0_copy = res_df['short_rtc_0_{}'.format(strat_version)].copy()
        res_df['short_rtc_0_{}'.format(strat_version)] = res_df['long_rtc_0_{}'.format(strat_version)]
        res_df['long_rtc_0_{}'.format(strat_version)] = short_rtc_0_copy

    res_df['short_rtc_gap_{}'.format(strat_version)] = abs(
        res_df['short_rtc_0_{}'.format(strat_version)] - res_df['short_rtc_1_{}'.format(strat_version)])
    res_df['long_rtc_gap_{}'.format(strat_version)] = abs(
        res_df['long_rtc_1_{}'.format(strat_version)] - res_df['long_rtc_0_{}'.format(strat_version)])

    res_df['h_short_rtc_1_{}'.format(strat_version)] = res_df['dc_lower_%s' % config.loc_set.point.tpg_itv1]
    # res_df['h_short_rtc_1_{}'.format(strat_version)] = res_df['dc_lower_%s' % config.loc_set.point.tpg_itv]
    res_df['h_short_rtc_0_{}'.format(strat_version)] = res_df['bb_upper_%s' % config.loc_set.point.tpg_itv0]

    res_df['h_long_rtc_1_{}'.format(strat_version)] = res_df['bb_upper_%s' % config.loc_set.point.tpg_itv1]
    # res_df['h_long_rtc_1_{}'.format(strat_version)] = res_df['dc_upper_%s' % config.loc_set.point.tpg_itv]
    res_df['h_long_rtc_0_{}'.format(strat_version)] = res_df['dc_lower_%s' % config.loc_set.point.tpg_itv0]

    #      entry reversion    #
    if config.ep_set.short_entry_score > 0:
        h_short_rtc_1_copy = res_df['h_short_rtc_1_{}'.format(strat_version)].copy()
        res_df['h_short_rtc_1_{}'.format(strat_version)] = res_df['h_long_rtc_1_{}'.format(strat_version)]
        res_df['h_long_rtc_1_{}'.format(strat_version)] = h_short_rtc_1_copy

        h_short_rtc_0_copy = res_df['h_short_rtc_0_{}'.format(strat_version)].copy()
        res_df['h_short_rtc_0_{}'.format(strat_version)] = res_df['h_long_rtc_0_{}'.format(strat_version)]
        res_df['h_long_rtc_0_{}'.format(strat_version)] = h_short_rtc_0_copy

    res_df['h_short_rtc_gap_{}'.format(strat_version)] = abs(
        res_df['h_short_rtc_0_{}'.format(strat_version)] - res_df['h_short_rtc_1_{}'.format(strat_version)])
    res_df['h_long_rtc_gap_{}'.format(strat_version)] = abs(
        res_df['h_long_rtc_1_{}'.format(strat_version)] - res_df['h_long_rtc_0_{}'.format(strat_version)])

    res_df['short_dtk_1_{}'.format(strat_version)] = res_df['bb_lower_%s' % config.loc_set.zone.dtk_itv]
    res_df['short_dtk_0_{}'.format(strat_version)] = res_df['dc_upper_%s' % config.loc_set.zone.dtk_itv]

    res_df['long_dtk_1_{}'.format(strat_version)] = res_df['bb_upper_%s' % config.loc_set.zone.dtk_itv]
    res_df['long_dtk_0_{}'.format(strat_version)] = res_df['dc_lower_%s' % config.loc_set.zone.dtk_itv]

    res_df['short_dtk_gap_{}'.format(strat_version)] = abs(
        res_df['short_dtk_0_{}'.format(strat_version)] - res_df['short_dtk_1_{}'.format(strat_version)])
    res_df['long_dtk_gap_{}'.format(strat_version)] = abs(
        res_df['long_dtk_1_{}'.format(strat_version)] - res_df['long_dtk_0_{}'.format(strat_version)])

    return res_df


def enlist_tr(res_df, config, np_timeidx):
    strat_version = config.strat_version

    res_df['entry_{}'.format(strat_version)] = np.zeros(len(res_df))
    res_df['h_entry_{}'.format(strat_version)] = np.zeros(len(res_df))

    # -------- set ep level -------- #

    #       limit ver.     #
    # res_df['short_ep_{}'.format(strat_version)] = res_df['short_rtc_1_{}'.format(strat_version)] + res_df['short_rtc_gap_{}'.format(strat_version)] * config.tr_set.ep_gap
    # res_df['long_ep_{}'.format(strat_version)] = res_df['long_rtc_1_{}'.format(strat_version)] - res_df['long_rtc_gap_{}'.format(strat_version)] * config.tr_set.ep_gap

    res_df['short_ep_{}'.format(strat_version)] = res_df['close'] + res_df[
        'h_short_rtc_gap_{}'.format(strat_version)] * config.tr_set.ep_gap
    res_df['long_ep_{}'.format(strat_version)] = res_df['close'] - res_df[
        'h_long_rtc_gap_{}'.format(strat_version)] * config.tr_set.ep_gap

    if config.tr_set.c_ep_gap != "None":
        res_df['short_ep_org_{}'.format(strat_version)] = res_df['short_ep_{}'.format(strat_version)].copy()
        res_df['long_ep_org_{}'.format(strat_version)] = res_df['long_ep_{}'.format(strat_version)].copy()

        res_df['short_ep2_{}'.format(strat_version)] = res_df['h_short_rtc_1_{}'.format(strat_version)] + res_df[
            'h_short_rtc_gap_{}'.format(strat_version)] * config.tr_set.c_ep_gap
        res_df['long_ep2_{}'.format(strat_version)] = res_df['h_long_rtc_1_{}'.format(strat_version)] - res_df[
            'h_long_rtc_gap_{}'.format(strat_version)] * config.tr_set.c_ep_gap

    res_df['short_spread_ep_{}'.format(strat_version)] = res_df[
        'bb_lower_5m']  # + res_df['h_short_rtc_gap_{}'.format(strat_version)] * config.loc_set.zone.spread_ep_gap
    res_df['long_spread_ep_{}'.format(strat_version)] = res_df[
        'bb_upper_5m']  # - res_df'h_long_rtc_gap_{}'.format(strat_version)] * config.loc_set.zone.spread_ep_gap

    #       market ver.     #
    if config.ep_set.entry_type == "MARKET":
        res_df['short_ep_{}'.format(strat_version)] = res_df['close']
        res_df['long_ep_{}'.format(strat_version)] = res_df['close']

    # ---------------------------------------- short = -1 ---------------------------------------- #
    # ---------------- ep_time  ---------------- #

    rsi_upper = 50 + config.loc_set.point.osc_band
    rsi_lower = 50 - config.loc_set.point.osc_band

    res_df['rsi_%s_shift' % config.loc_set.point.exp_itv] = res_df['rsi_%s' % config.loc_set.point.exp_itv].shift(
        config.loc_set.point.tf_entry)

    res_df['entry_{}'.format(strat_version)] = np.where(
        (res_df['rsi_%s_shift' % config.loc_set.point.exp_itv] >= rsi_upper) &
        (res_df['rsi_%s' % config.loc_set.point.exp_itv] < rsi_upper)
        , res_df['entry_{}'.format(strat_version)] - 1, res_df['entry_{}'.format(strat_version)])

    res_df['entry_{}'.format(strat_version)] = np.where((res_df['entry_{}'.format(strat_version)] < 0) &
                                                        (np_timeidx % config.loc_set.point.tf_entry == (
                                                                    config.loc_set.point.tf_entry - 1))
                                                        , res_df['entry_{}'.format(strat_version)] - 1,
                                                        res_df['entry_{}'.format(strat_version)])

    res_df['h_entry_{}'.format(strat_version)] = np.where(
        # (res_df['open'] >= res_df['bb_lower_%s' % config.loc_set.zone.dtk_itv]) &
        (res_df['close'].shift(config.loc_set.point.htf_entry * 1) >= res_df[
            'bb_lower_%s' % config.loc_set.zone.dtk_itv]) &
        (res_df['close'] < res_df['bb_lower_%s' % config.loc_set.zone.dtk_itv]) &
        (np_timeidx % config.loc_set.point.htf_entry == (config.loc_set.point.htf_entry - 1))
        , res_df['h_entry_{}'.format(strat_version)] - 1, res_df['h_entry_{}'.format(strat_version)])

    # ---------------------------------------- long = 1 ---------------------------------------- #
    # ---------------------- ep_time ---------------------- #

    res_df['entry_{}'.format(strat_version)] = np.where(
        (res_df['rsi_%s_shift' % config.loc_set.point.exp_itv] <= rsi_lower) &
        (res_df['rsi_%s' % config.loc_set.point.exp_itv] > rsi_lower)
        , res_df['entry_{}'.format(strat_version)] + 1, res_df['entry_{}'.format(strat_version)])

    res_df['entry_{}'.format(strat_version)] = np.where((res_df['entry_{}'.format(strat_version)] > 0) &
                                                        (np_timeidx % config.loc_set.point.tf_entry == (
                                                                    config.loc_set.point.tf_entry - 1))
                                                        , res_df['entry_{}'.format(strat_version)] + 1,
                                                        res_df['entry_{}'.format(strat_version)])

    res_df['h_entry_{}'.format(strat_version)] = np.where(
        # (res_df['open'] <= res_df['bb_upper_%s' % config.loc_set.zone.dtk_itv]) &
        (res_df['close'].shift(config.loc_set.point.htf_entry * 1) <= res_df[
            'bb_upper_%s' % config.loc_set.zone.dtk_itv]) &
        (res_df['close'] > res_df['bb_upper_%s' % config.loc_set.zone.dtk_itv]) &
        (np_timeidx % config.loc_set.point.htf_entry == (config.loc_set.point.htf_entry - 1))
        , res_df['h_entry_{}'.format(strat_version)] + 1, res_df['h_entry_{}'.format(strat_version)])

    # ------------------------------ rtc tp & out ------------------------------ #
    # --------------- bb rtc out --------------- #

    if config.loc_set.point.outg_dc_period != "None":
        res_df['short_rtc_0_{}'.format(strat_version)] = res_df['high'].rolling(
            config.loc_set.point.outg_dc_period).max()
        res_df['long_rtc_0_{}'.format(strat_version)] = res_df['low'].rolling(config.loc_set.point.outg_dc_period).min()

    res_df['short_out_{}'.format(strat_version)] = res_df['short_rtc_0_{}'.format(strat_version)] + res_df[
        'short_rtc_gap_{}'.format(strat_version)] * config.tr_set.out_gap
    res_df['long_out_{}'.format(strat_version)] = res_df['long_rtc_0_{}'.format(strat_version)] - res_df[
        'long_rtc_gap_{}'.format(strat_version)] * config.tr_set.out_gap

    if config.tr_set.t_out_gap != "None":
        res_df['short_out_org_{}'.format(strat_version)] = res_df['short_out_{}'.format(strat_version)].copy()
        res_df['long_out_org_{}'.format(strat_version)] = res_df['long_out_{}'.format(strat_version)].copy()

        res_df['short_out2_{}'.format(strat_version)] = res_df['short_rtc_0_{}'.format(strat_version)] + res_df[
            'short_rtc_gap_{}'.format(strat_version)] * config.tr_set.t_out_gap
        res_df['long_out2_{}'.format(strat_version)] = res_df['long_rtc_0_{}'.format(strat_version)] - res_df[
            'long_rtc_gap_{}'.format(strat_version)] * config.tr_set.t_out_gap

    # ------------------------------ tp ------------------------------ #
    # --------------- bb rtc tp --------------- #
    res_df['short_tp_{}'.format(strat_version)] = res_df['h_short_rtc_1_{}'.format(strat_version)] - res_df[
        'h_short_rtc_gap_{}'.format(strat_version)] * config.tr_set.tp_gap
    res_df['long_tp_{}'.format(strat_version)] = res_df['h_long_rtc_1_{}'.format(strat_version)] + res_df[
        'h_long_rtc_gap_{}'.format(strat_version)] * config.tr_set.tp_gap

    # --------------- set tp_line / dtk_line --------------- #
    if config.loc_set.zone.use_dtk_line:
        res_df['short_dtk_1_{}'.format(strat_version)] = np.where(res_df['h_entry_{}'.format(strat_version)] == -1,
                                                                  res_df['short_dtk_1_{}'.format(strat_version)],
                                                                  np.nan)
        res_df['short_dtk_1_{}'.format(strat_version)] = ffill(
            res_df['short_dtk_1_{}'.format(strat_version)].values.reshape(1, -1)).reshape(-1, 1)
        res_df['short_dtk_gap_{}'.format(strat_version)] = np.where(res_df['h_entry_{}'.format(strat_version)] == -1,
                                                                    res_df['short_dtk_gap_{}'.format(strat_version)],
                                                                    np.nan)
        res_df['short_dtk_gap_{}'.format(strat_version)] = ffill(
            res_df['short_dtk_gap_{}'.format(strat_version)].values.reshape(1, -1)).reshape(-1, 1)

        res_df['long_dtk_1_{}'.format(strat_version)] = np.where(res_df['h_entry_{}'.format(strat_version)] == 1,
                                                                 res_df['long_dtk_1_{}'.format(strat_version)], np.nan)
        res_df['long_dtk_1_{}'.format(strat_version)] = ffill(
            res_df['long_dtk_1_{}'.format(strat_version)].values.reshape(1, -1)).reshape(-1, 1)
        res_df['long_dtk_gap_{}'.format(strat_version)] = np.where(res_df['h_entry_{}'.format(strat_version)] == 1,
                                                                   res_df['long_dtk_gap_{}'.format(strat_version)],
                                                                   np.nan)
        res_df['long_dtk_gap_{}'.format(strat_version)] = ffill(
            res_df['long_dtk_gap_{}'.format(strat_version)].values.reshape(1, -1)).reshape(-1, 1)

    return res_df

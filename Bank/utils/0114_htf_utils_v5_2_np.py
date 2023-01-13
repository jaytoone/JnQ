from funcs.funcs_indicator_candlescore import *
import numpy as np


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


def enlist_tr(res_df, config, np_timeidx, mode='OPEN'):
    strat_version = config.strat_version
    # ---------------- get open_res ---------------- #
    rsi_upper = 50 + config.loc_set.point.osc_band
    rsi_lower = 50 - config.loc_set.point.osc_band
    rsi = res_df['rsi_%s' % config.loc_set.point.exp_itv].to_numpy()
    rsi_shift = res_df['rsi_%s' % config.loc_set.point.exp_itv].shift(config.loc_set.point.tf_entry).to_numpy()
    
    len_df = len(res_df)
    short_open_res = np.ones(len_df)
    long_open_res = np.ones(len_df)

    short_open_res *= (rsi_shift >= rsi_upper) & (rsi < rsi_upper)
    short_open_res *= np_timeidx % config.loc_set.point.tf_entry == (config.loc_set.point.tf_entry - 1)
    res_df['short_open_{}'.format(strat_version)] = short_open_res

    long_open_res *= (rsi_shift <= rsi_lower) & (rsi > rsi_lower)
    long_open_res *= np_timeidx % config.loc_set.point.tf_entry == (config.loc_set.point.tf_entry - 1)
    res_df['long_open_{}'.format(strat_version)] = long_open_res
    
    # ---------------- set ep level ---------------- #
    # -------- limit ver. -------- #
    if config.ep_set.entry_type == "LIMIT":
      # res_df['short_ep_{}'.format(strat_version)] = res_df['short_rtc_1_{}'.format(strat_version)] + res_df['short_rtc_gap_{}'.format(strat_version)] * config.tr_set.ep_gap
      # res_df['long_ep_{}'.format(strat_version)] = res_df['long_rtc_1_{}'.format(strat_version)] - res_df['long_rtc_gap_{}'.format(strat_version)] * config.tr_set.ep_gap

      res_df['short_ep_{}'.format(strat_version)] = res_df['close'] + res_df['h_short_rtc_gap_{}'.format(strat_version)] * config.tr_set.ep_gap
      res_df['long_ep_{}'.format(strat_version)] = res_df['close'] - res_df['h_long_rtc_gap_{}'.format(strat_version)] * config.tr_set.ep_gap       

    # -------- market ver. -------- #
    else:
        res_df['short_ep_{}'.format(strat_version)] = res_df['close']
        res_df['long_ep_{}'.format(strat_version)] = res_df['close']

    # -------- zoned_ep -------- #
    if config.tr_set.c_ep_gap != "None":
        res_df['short_ep_org_{}'.format(strat_version)] = res_df['short_ep_{}'.format(strat_version)].copy()
        res_df['long_ep_org_{}'.format(strat_version)] = res_df['long_ep_{}'.format(strat_version)].copy()

        res_df['short_ep2_{}'.format(strat_version)] = res_df['h_short_rtc_1_{}'.format(strat_version)] + res_df[
            'h_short_rtc_gap_{}'.format(strat_version)] * config.tr_set.c_ep_gap
        res_df['long_ep2_{}'.format(strat_version)] = res_df['h_long_rtc_1_{}'.format(strat_version)] - res_df[
            'h_long_rtc_gap_{}'.format(strat_version)] * config.tr_set.c_ep_gap

    # --------------- tp --------------- #
    res_df['short_tp_{}'.format(strat_version)] = res_df['h_short_rtc_1_{}'.format(strat_version)] - res_df[
        'h_short_rtc_gap_{}'.format(strat_version)] * config.tr_set.tp_gap
    res_df['long_tp_{}'.format(strat_version)] = res_df['h_long_rtc_1_{}'.format(strat_version)] + res_df[
        'h_long_rtc_gap_{}'.format(strat_version)] * config.tr_set.tp_gap

    # --------------- out --------------- #
    if config.loc_set.point.outg_dc_period != "None":  # out 만 영향 줄려고 tp 까지 해놓고, rtc_0 변경 
        res_df['short_rtc_0_{}'.format(strat_version)] = res_df['high'].rolling(config.loc_set.point.outg_dc_period).max()
        res_df['long_rtc_0_{}'.format(strat_version)] = res_df['low'].rolling(config.loc_set.point.outg_dc_period).min()

    res_df['short_out_{}'.format(strat_version)] = res_df['short_rtc_0_{}'.format(strat_version)] + res_df[
        'short_rtc_gap_{}'.format(strat_version)] * config.tr_set.out_gap
    res_df['long_out_{}'.format(strat_version)] = res_df['long_rtc_0_{}'.format(strat_version)] - res_df[
        'long_rtc_gap_{}'.format(strat_version)] * config.tr_set.out_gap

    # -------- zoned_out -------- #
    if config.tr_set.t_out_gap != "None":
        res_df['short_out_org_{}'.format(strat_version)] = res_df['short_out_{}'.format(strat_version)].copy()
        res_df['long_out_org_{}'.format(strat_version)] = res_df['long_out_{}'.format(strat_version)].copy()

        res_df['short_out2_{}'.format(strat_version)] = res_df['short_rtc_0_{}'.format(strat_version)] + res_df[
            'short_rtc_gap_{}'.format(strat_version)] * config.tr_set.t_out_gap
        res_df['long_out2_{}'.format(strat_version)] = res_df['long_rtc_0_{}'.format(strat_version)] - res_df[
            'long_rtc_gap_{}'.format(strat_version)] * config.tr_set.t_out_gap

        
    return res_df

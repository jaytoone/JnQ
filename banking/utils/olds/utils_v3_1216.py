from funcs.olds.funcs_indicator_candlescore import *
import numpy as np


def enlist_rtc(res_df, config):

    strat_version = config.strat_version

    res_df['short_rtc_1_{}'.format(strat_version)] = res_df['bb_lower_%s' % config.loc_set.point.tpg_itv]
    res_df['short_rtc_0_{}'.format(strat_version)] = res_df['dc_upper_%s' % config.loc_set.point.outg_itv]

    res_df['long_rtc_1_{}'.format(strat_version)] = res_df['bb_upper_%s' % config.loc_set.point.tpg_itv]
    res_df['long_rtc_0_{}'.format(strat_version)] = res_df['dc_lower_%s' % config.loc_set.point.outg_itv]
    
    #      entry reversion    #
    if config.ep_set.short_entry_score > 0:      
      short_rtc_1_copy = res_df['short_rtc_1_{}'.format(strat_version)].copy()
      res_df['short_rtc_1_{}'.format(strat_version)] = res_df['long_rtc_1_{}'.format(strat_version)]
      res_df['long_rtc_1_{}'.format(strat_version)] = short_rtc_1_copy

      short_rtc_0_copy = res_df['short_rtc_0_{}'.format(strat_version)].copy()
      res_df['short_rtc_0_{}'.format(strat_version)] = res_df['long_rtc_0_{}'.format(strat_version)]
      res_df['long_rtc_0_{}'.format(strat_version)] = short_rtc_0_copy

    res_df['short_rtc_gap_{}'.format(strat_version)] = abs(res_df['short_rtc_0_{}'.format(strat_version)] - res_df['short_rtc_1_{}'.format(strat_version)])
    res_df['long_rtc_gap_{}'.format(strat_version)] = abs(res_df['long_rtc_1_{}'.format(strat_version)] - res_df['long_rtc_0_{}'.format(strat_version)])

    res_df['h_short_rtc_1_{}'.format(strat_version)] = res_df['bb_lower_%s' % config.loc_set.point.tpg_itv]
    # res_df['h_short_rtc_1_{}'.format(strat_version)] = res_df['dc_lower_%s' % config.loc_set.point.tpg_itv]
    res_df['h_short_rtc_0_{}'.format(strat_version)] = res_df['dc_upper_%s' % config.loc_set.point.tpg_itv]

    res_df['h_long_rtc_1_{}'.format(strat_version)] = res_df['bb_upper_%s' % config.loc_set.point.tpg_itv]
    # res_df['h_long_rtc_1_{}'.format(strat_version)] = res_df['dc_upper_%s' % config.loc_set.point.tpg_itv]
    res_df['h_long_rtc_0_{}'.format(strat_version)] = res_df['dc_lower_%s' % config.loc_set.point.tpg_itv]

    #      entry reversion    #
    if config.ep_set.short_entry_score > 0:      
      h_short_rtc_1_copy = res_df['h_short_rtc_1_{}'.format(strat_version)].copy()
      res_df['h_short_rtc_1_{}'.format(strat_version)] = res_df['h_long_rtc_1_{}'.format(strat_version)]
      res_df['h_long_rtc_1_{}'.format(strat_version)] = h_short_rtc_1_copy

      h_short_rtc_0_copy = res_df['h_short_rtc_0_{}'.format(strat_version)].copy()
      res_df['h_short_rtc_0_{}'.format(strat_version)] = res_df['h_long_rtc_0_{}'.format(strat_version)]
      res_df['h_long_rtc_0_{}'.format(strat_version)] = h_short_rtc_0_copy

    res_df['h_short_rtc_gap_{}'.format(strat_version)] = abs(res_df['h_short_rtc_0_{}'.format(strat_version)] - res_df['h_short_rtc_1_{}'.format(strat_version)])
    res_df['h_long_rtc_gap_{}'.format(strat_version)] = abs(res_df['h_long_rtc_1_{}'.format(strat_version)] - res_df['h_long_rtc_0_{}'.format(strat_version)])   

    res_df['short_dtk_1_{}'.format(strat_version)] = res_df['bb_lower_%s' % config.loc_set.zone.dtk_itv]
    res_df['short_dtk_0_{}'.format(strat_version)] = res_df['dc_upper_%s' % config.loc_set.zone.dtk_itv]
    res_df['short_dtk_gap_{}'.format(strat_version)] = res_df['short_dtk_0_{}'.format(strat_version)] - res_df['short_dtk_1_{}'.format(strat_version)]

    res_df['long_dtk_1_{}'.format(strat_version)] = res_df['bb_upper_%s' % config.loc_set.zone.dtk_itv]
    res_df['long_dtk_0_{}'.format(strat_version)] = res_df['dc_lower_%s' % config.loc_set.zone.dtk_itv]
    res_df['long_dtk_gap_{}'.format(strat_version)] = res_df['long_dtk_1_{}'.format(strat_version)] - res_df['long_dtk_0_{}'.format(strat_version)]

    res_df = dtk_plot(res_df, dtk_itv2='15m', hhtf_entry=15, use_dtk_line=config.loc_set.zone.use_dtk_line)

    return res_df


def enlist_tr(res_df, config, np_timeidx):

    strat_version = config.strat_version

    res_df['entry_{}'.format(strat_version)] = np.zeros(len(res_df))
    res_df['h_entry_{}'.format(strat_version)] = np.zeros(len(res_df))

    # -------- set ep level -------- #

    #       limit ver.     #
    res_df['short_ep_{}'.format(strat_version)] = res_df['h_short_rtc_1_{}'.format(strat_version)] + res_df['h_short_rtc_gap_{}'.format(strat_version)] * config.tr_set.ep_gap
    res_df['long_ep_{}'.format(strat_version)] = res_df['h_long_rtc_1_{}'.format(strat_version)] - res_df['h_long_rtc_gap_{}'.format(strat_version)] * config.tr_set.ep_gap

    # res_df['short_ep'] = res_df['close'] #+ res_df['h_short_rtc_gap'] * config.tr_set.ep_gap
    # res_df['long_ep'] = res_df['close'] #- res_df['h_long_rtc_gap'] * config.tr_set.ep_gap


    if config.tr_set.c_ep_gap != "None":
        res_df['short_ep_org_{}'.format(strat_version)] = res_df['short_ep_{}'.format(strat_version)].copy()
        res_df['long_ep_org_{}'.format(strat_version)] = res_df['long_ep_{}'.format(strat_version)].copy()

        res_df['short_ep2_{}'.format(strat_version)] = res_df['h_short_rtc_1_{}'.format(strat_version)] + res_df['h_short_rtc_gap_{}'.format(strat_version)] * config.tr_set.c_ep_gap
        res_df['long_ep2_{}'.format(strat_version)] = res_df['h_long_rtc_1_{}'.format(strat_version)] - res_df['h_long_rtc_gap_{}'.format(strat_version)] * config.tr_set.c_ep_gap

    res_df['short_spread_ep_{}'.format(strat_version)] = res_df['bb_lower_5m'] #+ res_df['h_short_rtc_gap'] * config.loc_set.zone.spread_ep_gap
    res_df['long_spread_ep_{}'.format(strat_version)] = res_df['bb_upper_5m']  #- res_df['h_long_rtc_gap'] * config.loc_set.zone.spread_ep_gap

    #       market ver.     #
    if config.ep_set.entry_type == "MARKET":
        res_df['short_ep_{}'.format(strat_version)] = res_df['close']
        res_df['long_ep_{}'.format(strat_version)] = res_df['close']

    # ---------------------------------------- short = -1 ---------------------------------------- #
    # ---------------- ep_time  ---------------- #  
      
    # res_df['entry_{}'.format(strat_version)] = np.where((res_df['close'].shift(config.loc_set.point.tf_entry * 1) >= res_df['bb_lower_%s' % config.loc_set.point.exp_itv]) &
    res_df['entry_{}'.format(strat_version)] = np.where((res_df['open'] >= res_df['bb_lower_%s' % config.loc_set.point.exp_itv]) &
                              # (res_df['close'].shift(config.loc_set.point.tf_entry * 1) >= res_df['bb_lower_%s' % config.loc_set.point.exp_itv]) &
                              (res_df['close'] < res_df['bb_lower_%s' % config.loc_set.point.exp_itv])
                              , res_df['entry_{}'.format(strat_version)] - 1, res_df['entry_{}'.format(strat_version)])

    res_df['entry_{}'.format(strat_version)] = np.where((res_df['entry_{}'.format(strat_version)] < 0) &
                              (np_timeidx % config.loc_set.point.tf_entry == (config.loc_set.point.tf_entry - 1))
                              , res_df['entry_{}'.format(strat_version)] - 1, res_df['entry_{}'.format(strat_version)])

    res_df['h_entry_{}'.format(strat_version)] = np.where(  # (res_df['open'] >= res_df['bb_lower_%s' % config.loc_set.zone.dtk_itv]) &
        (res_df['close'].shift(config.loc_set.point.htf_entry * 1) >= res_df['bb_lower_%s' % config.loc_set.zone.dtk_itv]) &
        (res_df['close'] < res_df['bb_lower_%s' % config.loc_set.zone.dtk_itv]) &
        (np_timeidx % config.loc_set.point.htf_entry == (config.loc_set.point.htf_entry - 1))
        , res_df['h_entry_{}'.format(strat_version)] - 1, res_df['h_entry_{}'.format(strat_version)])

    # ---------------------------------------- long = 1 ---------------------------------------- #
    # ---------------------- ep_time ---------------------- #      
      
    # res_df['entry_{}'.format(strat_version)] = np.where((res_df['close'].shift(config.loc_set.point.tf_entry * 1) <= res_df['bb_upper_%s' % config.loc_set.point.exp_itv]) &
    res_df['entry_{}'.format(strat_version)] = np.where((res_df['open'] <= res_df['bb_upper_%s' % config.loc_set.point.exp_itv]) &
                              # (res_df['close'].shift(config.loc_set.point.tf_entry * 1) <= res_df['bb_upper_%s' % config.loc_set.point.exp_itv]) &
                              (res_df['close'] > res_df['bb_upper_%s' % config.loc_set.point.exp_itv])
                              , res_df['entry_{}'.format(strat_version)] + 1, res_df['entry_{}'.format(strat_version)])

    res_df['entry_{}'.format(strat_version)] = np.where((res_df['entry_{}'.format(strat_version)] > 0) &
                              (np_timeidx % config.loc_set.point.tf_entry == (config.loc_set.point.tf_entry - 1))
                              , res_df['entry_{}'.format(strat_version)] + 1, res_df['entry_{}'.format(strat_version)])

    res_df['h_entry_{}'.format(strat_version)] = np.where(  # (res_df['open'] <= res_df['bb_upper_%s' % config.loc_set.zone.dtk_itv]) &
        (res_df['close'].shift(config.loc_set.point.htf_entry * 1) <= res_df['bb_upper_%s' % config.loc_set.zone.dtk_itv]) &
        (res_df['close'] > res_df['bb_upper_%s' % config.loc_set.zone.dtk_itv]) &
        (np_timeidx % config.loc_set.point.htf_entry == (config.loc_set.point.htf_entry - 1))
        , res_df['h_entry_{}'.format(strat_version)] + 1, res_df['h_entry_{}'.format(strat_version)])

    # ------------------------------ rtc tp & out ------------------------------ #
    # --------------- bb rtc out --------------- #

    if config.loc_set.point.outg_dc_period != "None":
        res_df['short_rtc_0_{}'.format(strat_version)] = res_df['high'].rolling(config.loc_set.point.outg_dc_period).max()
        res_df['long_rtc_0_{}'.format(strat_version)] = res_df['low'].rolling(config.loc_set.point.outg_dc_period).min()

    res_df['short_out_{}'.format(strat_version)] = res_df['short_rtc_0_{}'.format(strat_version)] + res_df['short_rtc_gap_{}'.format(strat_version)] * config.tr_set.out_gap
    res_df['long_out_{}'.format(strat_version)] = res_df['long_rtc_0_{}'.format(strat_version)] - res_df['long_rtc_gap_{}'.format(strat_version)] * config.tr_set.out_gap

    if config.tr_set.t_out_gap != "None":
        res_df['short_out_org_{}'.format(strat_version)] = res_df['short_out_{}'.format(strat_version)].copy()
        res_df['long_out_org_{}'.format(strat_version)] = res_df['long_out_{}'.format(strat_version)].copy()

        res_df['short_out2_{}'.format(strat_version)] = res_df['short_rtc_0_{}'.format(strat_version)] + res_df['short_rtc_gap_{}'.format(strat_version)] * config.tr_set.t_out_gap
        res_df['long_out2_{}'.format(strat_version)] = res_df['long_rtc_0_{}'.format(strat_version)] - res_df['long_rtc_gap_{}'.format(strat_version)] * config.tr_set.t_out_gap

    # ------------------------------ tp ------------------------------ #
    # --------------- bb rtc tp --------------- #
    res_df['short_tp_{}'.format(strat_version)] = res_df['h_short_rtc_1_{}'.format(strat_version)] - res_df['h_short_rtc_gap_{}'.format(strat_version)] * config.tr_set.tp_gap
    res_df['long_tp_{}'.format(strat_version)] = res_df['h_long_rtc_1_{}'.format(strat_version)] + res_df['h_long_rtc_gap_{}'.format(strat_version)] * config.tr_set.tp_gap

    # --------------- set tp_line / dtk_line --------------- #
    # res_df['short_tp_1_{}'.format(strat_version)] = np.where(res_df['h_entry_{}'.format(strat_version)] == -1, res_df['short_rtc_1_{}'.format(strat_version)], np.nan)
    # res_df['short_tp_1_{}'.format(strat_version)] = ffill(res_df['short_tp_1_{}'.format(strat_version)].values.reshape(1, -1)).reshape(-1, 1)
    # res_df['short_tp_gap_{}'.format(strat_version)] = np.where(res_df['h_entry_{}'.format(strat_version)] == -1, res_df['h_short_rtc_gap_{}'.format(strat_version)], np.nan)  # ltf_gap 은 out 을 위한 gap 임
    # res_df['short_tp_gap_{}'.format(strat_version)] = ffill(res_df['short_tp_gap_{}'.format(strat_version)].values.reshape(1, -1)).reshape(-1, 1)

    # res_df['long_tp_1_{}'.format(strat_version)] = np.where(res_df['h_entry_{}'.format(strat_version)] == 1, res_df['long_rtc_1_{}'.format(strat_version)], np.nan)
    # res_df['long_tp_1_{}'.format(strat_version)] = ffill(res_df['long_tp_1_{}'.format(strat_version)].values.reshape(1, -1)).reshape(-1, 1)
    # res_df['long_tp_gap_{}'.format(strat_version)] = np.where(res_df['h_entry_{}'.format(strat_version)] == 1, res_df['h_long_rtc_gap_{}'.format(strat_version)], np.nan)
    # res_df['long_tp_gap_{}'.format(strat_version)] = ffill(res_df['long_tp_gap_{}'.format(strat_version)].values.reshape(1, -1)).reshape(-1, 1)

    # res_df['h_short_tp_1_{}'.format(strat_version)] = np.where(res_df['h_entry_{}'.format(strat_version)] == -1, res_df['h_short_rtc_1_{}'.format(strat_version)], np.nan)
    # res_df['h_short_tp_1_{}'.format(strat_version)] = ffill(res_df['h_short_tp_1_{}'.format(strat_version)].values.reshape(1, -1)).reshape(-1, 1)
    # res_df['h_short_tp_gap_{}'.format(strat_version)] = np.where(res_df['h_entry_{}'.format(strat_version)] == -1, res_df['h_short_rtc_gap_{}'.format(strat_version)], np.nan)
    # res_df['h_short_tp_gap_{}'.format(strat_version)] = ffill(res_df['h_short_tp_gap_{}'.format(strat_version)].values.reshape(1, -1)).reshape(-1, 1)

    # res_df['h_long_tp_1_{}'.format(strat_version)] = np.where(res_df['h_entry_{}'.format(strat_version)] == 1, res_df['h_long_rtc_1_{}'.format(strat_version)], np.nan)
    # res_df['h_long_tp_1_{}'.format(strat_version)] = ffill(res_df['h_long_tp_1_{}'.format(strat_version)].values.reshape(1, -1)).reshape(-1, 1)
    # res_df['h_long_tp_gap_{}'.format(strat_version)] = np.where(res_df['h_entry_{}'.format(strat_version)] == 1, res_df['h_long_rtc_gap_{}'.format(strat_version)], np.nan)
    # res_df['h_long_tp_gap_{}'.format(strat_version)] = ffill(res_df['h_long_tp_gap_{}'.format(strat_version)].values.reshape(1, -1)).reshape(-1, 1)

    if config.loc_set.zone.use_dtk_line:
      res_df['short_dtk_1_{}'.format(strat_version)] = np.where(res_df['h_entry_{}'.format(strat_version)] == -1, res_df['short_dtk_1_{}'.format(strat_version)], np.nan)
      res_df['short_dtk_1_{}'.format(strat_version)] = ffill(res_df['short_dtk_1_{}'.format(strat_version)].values.reshape(1, -1)).reshape(-1, 1)
      res_df['short_dtk_gap_{}'.format(strat_version)] = np.where(res_df['h_entry_{}'.format(strat_version)] == -1, res_df['short_dtk_gap_{}'.format(strat_version)], np.nan)
      res_df['short_dtk_gap_{}'.format(strat_version)] = ffill(res_df['short_dtk_gap_{}'.format(strat_version)].values.reshape(1, -1)).reshape(-1, 1)

      res_df['long_dtk_1_{}'.format(strat_version)] = np.where(res_df['h_entry_{}'.format(strat_version)] == 1, res_df['long_dtk_1_{}'.format(strat_version)], np.nan)
      res_df['long_dtk_1_{}'.format(strat_version)] = ffill(res_df['long_dtk_1_{}'.format(strat_version)].values.reshape(1, -1)).reshape(-1, 1)
      res_df['long_dtk_gap_{}'.format(strat_version)] = np.where(res_df['h_entry_{}'.format(strat_version)] == 1, res_df['long_dtk_gap_{}'.format(strat_version)], np.nan)
      res_df['long_dtk_gap_{}'.format(strat_version)] = ffill(res_df['long_dtk_gap_{}'.format(strat_version)].values.reshape(1, -1)).reshape(-1, 1)

    res_df['dc_upper_v2_{}'.format(strat_version)] = res_df['high'].rolling(config.loc_set.zone.dc_period).max()
    res_df['dc_lower_v2_{}'.format(strat_version)] = res_df['low'].rolling(config.loc_set.zone.dc_period).min()
    
    res_df['zone_dc_upper_v2_{}'.format(strat_version)] = res_df['high'].rolling(config.loc_set.zone.zone_dc_period).max()
    res_df['zone_dc_lower_v2_{}'.format(strat_version)] = res_df['low'].rolling(config.loc_set.zone.zone_dc_period).min()

    return res_df





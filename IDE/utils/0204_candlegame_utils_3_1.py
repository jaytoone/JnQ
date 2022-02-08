from funcs.funcs_indicator_candlescore_addnb import *
import numpy as np
import logging

sys_log = logging.getLogger()


def enlist_rtc(res_df, config, np_timeidx):
    return res_df


def enlist_tr(res_df, config, np_timeidx, mode='OPEN'):
    try:
        strat_version = config.strat_version
        if mode == 'OPEN':
            short_open_np = np.zeros(len(res_df))
            long_open_np = np.zeros(len(res_df))
            res_df['short_open_{}'.format(strat_version)] = np.where((np_timeidx % config.loc_set.point.tf_entry == 0), short_open_np - 1, short_open_np)
            res_df['long_open_{}'.format(strat_version)] = np.where((np_timeidx % config.loc_set.point.tf_entry == 0), long_open_np + 1, long_open_np)
        else:
            short_close_np = np.zeros(len(res_df))
            long_close_np = np.zeros(len(res_df))
            res_df['short_close_{}'.format(strat_version)] = np.where((np_timeidx % config.loc_set.point.tf_entry == config.loc_set.point.tf_entry - 1)
                                                                     , short_close_np - 1, short_close_np)
            res_df['long_close_{}'.format(strat_version)] = np.where((np_timeidx % config.loc_set.point.tf_entry == config.loc_set.point.tf_entry - 1)
                                                                     , long_close_np + 1, long_close_np)

        # ------------------ ep ------------------ #
        close_shift1 = res_df['close'].shift(1)
        # close = res_df['close']

        # ------------ limit ver. ------------ #
        if config.ep_set.entry_type == "LIMIT":
            res_df['short_ep_{}'.format(strat_version)] = close_shift1
            res_df['long_ep_{}'.format(strat_version)] = close_shift1

            if config.tr_set.c_ep_gap != "None":
                res_df['short_ep_org_{}'.format(strat_version)] = res_df['short_ep_{}'.format(strat_version)].copy()
                res_df['long_ep_org_{}'.format(strat_version)] = res_df['long_ep_{}'.format(strat_version)].copy()

                res_df['short_ep2_{}'.format(strat_version)] = res_df['h_short_rtc_1_{}'.format(strat_version)] + res_df[
                    'h_short_rtc_gap_{}'.format(strat_version)] * config.tr_set.c_ep_gap
                res_df['long_ep2_{}'.format(strat_version)] = res_df['h_long_rtc_1_{}'.format(strat_version)] - res_df[
                    'h_long_rtc_gap_{}'.format(strat_version)] * config.tr_set.c_ep_gap
        # ------------ market ver. ------------ #
        else:
            res_df['short_ep_{}'.format(strat_version)] = close_shift1
            res_df['long_ep_{}'.format(strat_version)] = close_shift1

        # ------------------ tp ------------------ #
        res_df['short_tp_{}'.format(strat_version)] = np.nan
        res_df['long_tp_{}'.format(strat_version)] = np.nan

        # ------------------ out ------------------ #
        res_df['short_out_{}'.format(strat_version)] = np.nan
        res_df['long_out_{}'.format(strat_version)] = np.nan

        if config.tr_set.t_out_gap != "None":
            res_df['short_out_org_{}'.format(strat_version)] = res_df['short_out_{}'.format(strat_version)].copy()
            res_df['long_out_org_{}'.format(strat_version)] = res_df['long_out_{}'.format(strat_version)].copy()

            res_df['short_out2_{}'.format(strat_version)] = res_df['short_rtc_0_{}'.format(strat_version)] + res_df[
                'short_rtc_gap_{}'.format(strat_version)] * config.tr_set.t_out_gap
            res_df['long_out2_{}'.format(strat_version)] = res_df['long_rtc_0_{}'.format(strat_version)] - res_df[
                'long_rtc_gap_{}'.format(strat_version)] * config.tr_set.t_out_gap

    except Exception as e:
        sys_log.error("error in enlist_tr :", e)

    return res_df

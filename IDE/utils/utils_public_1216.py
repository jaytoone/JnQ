from funcs.funcs_indicator import *
from funcs.funcs_trader import *


class OrderSide:
    BUY = "BUY"
    SELL = "SELL"
    INVALID = None


def lvrg_set(res_df, config, open_side, ep_, out_, fee, limit_leverage=50):

    strat_version = config.strat_version

    if open_side == OrderSide.SELL:

        if strat_version == "v3":
            # config.lvrg_set.leverage = config.lvrg_set.target_pct / (out_ / ep_ - 1 - (fee + config.trader_set.market_fee))
            config.lvrg_set.leverage = config.lvrg_set.target_pct / (
                        out_ / ep_ - 1 - (fee + config.trader_set.market_fee))

            #     zone 에 따른 c_ep_gap 를 고려 (loss 완화 방향) / 윗 줄은 수익 극대화 방향
            # config.lvrg_set.leverage = config.lvrg_set.target_pct / (out_ / res_df['short_ep_org'].iloc[ep_j] - 1 - (fee + config.trader_set.market_fee))

        elif strat_version == "v5_2":
            # config.lvrg_set.leverage = config.lvrg_set.target_pct / abs(ep_ / out_ - 1 - (fee + config.trader_set.market_fee))
            config.lvrg_set.leverage = config.lvrg_set.target_pct / abs(
                ep_ / out_ - 1 - (fee + config.trader_set.market_fee))
            # config.lvrg_set.leverage = config.lvrg_set.target_pct / abs(res_df['short_ep_org'].iloc[ep_j] / out_ - 1 - (fee + config.trader_set.market_fee))

    else:
        #   윗 phase 는 min_pr 의 오차가 커짐
        if strat_version == "v3":
            # config.lvrg_set.leverage = config.lvrg_set.target_pct / (ep_ / out_ - 1 - (fee + config.trader_set.market_fee))
            config.lvrg_set.leverage = config.lvrg_set.target_pct / (
                        ep_ / out_ - 1 - (fee + config.trader_set.market_fee))
            # config.lvrg_set.leverage = config.lvrg_set.target_pct / (res_df['long_ep_org'].iloc[ep_j] / out_ - 1 - (fee + config.trader_set.market_fee))

        elif strat_version == "v5_2":
            # config.lvrg_set.leverage = config.lvrg_set.target_pct / abs(out_ / ep_ - 1 - (fee + config.trader_set.market_fee))
            config.lvrg_set.leverage = config.lvrg_set.target_pct / abs(
                out_ / ep_ - 1 - (fee + config.trader_set.market_fee))
            # config.lvrg_set.leverage = config.lvrg_set.target_pct / abs(out_ / res_df['long_ep_org'].iloc[ep_j] - 1 - (fee + config.trader_set.market_fee))

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
    df = dc_line(df, None, '1m', dc_period=20)
    df = bb_line(df, None, '1m')
    df = bb_line(df, third_df, '5m')
    df = dc_line(df, third_df, '5m')
    df = bb_line(df, fourth_df, '15m')
    df = dc_line(df, fourth_df, '15m')

    df['rsi_1m'] = rsi(df, 14)

    third_df['ema_5m'] = ema(third_df['close'], 200)
    df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, third_df, [-1]), columns=['ema_5m']))

    return df


def public_indi(res_df):
    res_df = bb_level(res_df, '5m', 1)
    res_df = dc_level(res_df, '5m', 1)
    res_df = bb_level(res_df, '15m', 1)
    res_df = dc_level(res_df, '15m', 1)
    # res_df = bb_level(res_df, '30m', 1)
    # res_df = dc_level(res_df, '30m', 1)

    return res_df


def short_ep_loc(res_df, config, i, show_detail=True):
    strat_version = config.strat_version

    # ------- param init ------- #
    open_side = None

    mr_const_cnt = 0
    mr_score = 0
    zone = 'n'

    if config.ep_set.entry_type == 'MARKET':
        if config.tp_set.tp_type != 'MARKET':
            tp_fee = config.trader_set.market_fee + config.trader_set.limit_fee
        else:
            tp_fee = config.trader_set.market_fee + config.trader_set.market_fee
        out_fee = config.trader_set.market_fee + config.trader_set.market_fee
    else:
        if config.tp_set.tp_type != 'MARKET':
            tp_fee = config.trader_set.limit_fee + config.trader_set.limit_fee
        else:
            tp_fee = config.trader_set.limit_fee + config.trader_set.market_fee
        out_fee = config.trader_set.limit_fee + config.trader_set.market_fee

    # # -------------- tr scheduling -------------- #
    # if config.loc_set.zone.tr_thresh != "None":

    #   mr_const_cnt += 1
    #   tr = ((done_tp - ep_list[0] - tp_fee * ep_list[0]) / (ep_list[0] - done_out + out_fee * ep_list[0]))

    # -------------- spread scheduling -------------- #
    if config.loc_set.zone.short_spread != "None":

        mr_const_cnt += 1

        spread = (res_df['bb_base_5m'].iloc[i] - res_df['bb_lower_5m'].iloc[i] - tp_fee * res_df['bb_base_5m'].iloc[
            # spread = (res_df['bb_base_5m'].iloc[i] - res_df['bb_lower_5m'].iloc[i] - out_fee * res_df['bb_base_5m'].iloc[
            # i]) / (res_df['bb_base_5m'].iloc[i] - res_df['bb_lower_5m'].iloc[i] + tp_fee *
            i]) / (res_df['bb_base_5m'].iloc[i] - res_df['bb_lower_5m'].iloc[i] + out_fee *
                   res_df['bb_base_5m'].iloc[i])
        # spread = (res_df['bb_base_15m'].iloc[i] - res_df['bb_lower_5m'].iloc[i] - tp_fee * res_df['bb_base_15m'].iloc[
        #     i]) / (res_df['bb_base_15m'].iloc[i] - res_df['bb_lower_5m'].iloc[i] + out_fee *
        #             res_df['bb_base_15m'].iloc[i])

        # spread = (res_df['dc_upper_5m'].iloc[i] - res_df['bb_lower_5m'].iloc[i] - tp_fee * res_df['bb_lower_5m'].iloc[
        #     i]) / (res_df['dc_upper_5m'].iloc[i] - res_df['bb_lower_5m'].iloc[i] + out_fee *
        #             res_df['bb_lower_5m'].iloc[i])
        # spread = (res_df['short_rtc_gap'].iloc[i] * (0.443) - tp_fee * res_df['short_ep'].iloc[
        #     i]) / (res_df['short_rtc_gap'].iloc[i] * (0.417) + out_fee * res_df['short_ep'].iloc[i])

        # spread = (res_df['dc_upper_15m'].iloc[i] - res_df['dc_lower_5m'].iloc[i] - tp_fee * res_df['dc_lower_5m'].iloc[
        #     i]) / (res_df['dc_upper_15m'].iloc[i] - res_df['dc_lower_5m'].iloc[i] + out_fee *
        #             res_df['dc_lower_5m'].iloc[i])
        # spread = ((res_df['dc_upper_15m'].iloc[i] - res_df['dc_lower_5m'].iloc[i])/2 - tp_fee * res_df['dc_lower_5m'].iloc[
        #     i]) / ((res_df['dc_upper_15m'].iloc[i] - res_df['dc_lower_5m'].iloc[i])/2 + out_fee *
        #             res_df['dc_lower_5m'].iloc[i])

        if spread >= config.loc_set.zone.short_spread:
            mr_score += 1

        if show_detail:
            print("spread :", spread)

    # -------------- dtk -------------- #
    if config.loc_set.zone.dt_k != "None":

        mr_const_cnt += 1
        # if res_df['dc_lower_%s' % config.loc_set.zone.dtk_dc_itv].iloc[i] >= res_df['short_rtc_1'].iloc[i] - res_df['h_short_rtc_gap'].iloc[i] * config.loc_set.zone.dt_k:
        #     dtk_v1 & v2 platform     #
        if config.loc_set.zone.dtk_dc_itv != "None":
            dc = res_df['dc_lower_%s' % config.loc_set.zone.dtk_dc_itv].iloc[i]
            dt_k = res_df['short_dtk_1_{}'.format(strat_version)].iloc[i] - \
                   res_df['short_dtk_gap_{}'.format(strat_version)].iloc[i] * config.loc_set.zone.dt_k
            if dc >= dt_k:
                mr_score += 1

                #     dc_v2   #
        else:
            dc = res_df['dc_lower_v2_{}'.format(strat_version)].iloc[i]
            dt_k = res_df['short_dtk_1_{}'.format(strat_version)].iloc[i] - \
                   res_df['short_dtk_gap_{}'.format(strat_version)].iloc[i] * config.loc_set.zone.dt_k
            if dc >= dt_k:
                # if res_df['dc_lower_v2_{}'.format(strat_version)].iloc[i] >= res_df['short_dtk_1_{}'.format(strat_version)].iloc[i] - res_df['short_dtk_gap_{}'.format(strat_version)].iloc[i] * config.loc_set.zone.dt_k and \
                # res_df['dc_upper_v2_{}'.format(strat_version)].iloc[i] <= res_df['long_dtk_1_{}'.format(strat_version)].iloc[i] + res_df['long_dtk_gap_{}'.format(strat_version)].iloc[i] * config.loc_set.zone.dt_k:

                mr_score += 1

        if show_detail:
            print("dc :", dc)
            print("dt_k :", dt_k)

    # -------------- zone rejection  -------------- #
    if config.loc_set.zone.zone_rejection:

        #       by bb and dc - rsi      #
        # mr_const_cnt += 1
        # if res_df['bb_base_%s' % config.loc_set.zone.bbz_itv].iloc[i] <= res_df['dc_upper_1m'].iloc[i] <= res_df['bb_upper_%s' % config.loc_set.zone.bbz_itv].iloc[i]:
        #   mr_score += 1

        #       by bb       # 
        mr_const_cnt += 1
        # if res_df['bb_upper_5m'].iloc[i] < res_df['bb_base_%s' % config.loc_set.zone.bbz_itv].iloc[i]:

        # if res_df['close'].iloc[i] < res_df['bb_lower_%s' % config.loc_set.zone.bbz_itv].iloc[i]:   # org
        # if res_df['close'].iloc[i] > res_df['bb_lower_%s' % config.loc_set.zone.bbz_itv].iloc[i]:  # inv
        # if res_df['close'].iloc[i] > res_df['bb_upper_%s' % config.loc_set.zone.bbz_itv].iloc[i]:
        if res_df['close'].iloc[i] > res_df['bb_upper2_%s' % config.loc_set.zone.bbz_itv].iloc[i]:
            # if res_df['close'].iloc[i] > res_df['bb_upper3_%s' % config.loc_set.zone.bbz_itv].iloc[i]:

            mr_score += 1

            #       by ema       #
        mr_const_cnt += 1
        # if res_df['short_ep'].iloc[i] < res_df['ema_5m'].iloc[i]:
        if res_df['close'].iloc[i] < res_df['ema_5m'].iloc[i]:
            mr_score += 1

            #       by zone_dtk       #
        # mr_const_cnt += 1
        # if res_df['zone_dc_upper_v2_{}'.format(strat_version)].iloc[i] < res_df['long_dtk_plot_1'].iloc[i] + res_df['long_dtk_plot_gap'].iloc[
        #     i] * config.loc_set.zone.zone_dt_k:
        #   mr_score += 1

    # -------------- zoning -------------- #
    if config.tr_set.c_ep_gap != "None":

        #       by bb       # 
        # if res_df['close'].iloc[i] > res_df['bb_lower_%s' % config.loc_set.zone.bbz_itv].iloc[i]:

        #       by zone_dtk       #

        #         c_zone        #
        if res_df['zone_dc_upper_v2_{}'.format(strat_version)].iloc[i] > res_df['long_dtk_plot_1'].iloc[i] + \
                res_df['long_dtk_plot_gap'].iloc[
                    i] * config.loc_set.zone.zone_dt_k:

            if config.ep_set.static_ep:
                res_df['short_ep_{}'.format(strat_version)].iloc[i] = res_df['short_ep2_{}'.format(strat_version)].iloc[
                    i]
            else:
                res_df['short_ep_{}'.format(strat_version)] = res_df['short_ep2_{}'.format(strat_version)]

            if config.out_set.static_out:
                res_df['short_out_{}'.format(strat_version)].iloc[i] = \
                res_df['short_out_org_{}'.format(strat_version)].iloc[i]
            else:
                res_df['short_out_{}'.format(strat_version)] = res_df['short_out_org_{}'.format(strat_version)]

            zone = 'c'

        #         t_zone        #
        else:

            # mr_const_cnt += 1   # zone_rejection - temporary

            if config.ep_set.static_ep:
                res_df['short_ep_{}'.format(strat_version)].iloc[i] = \
                res_df['short_ep_org_{}'.format(strat_version)].iloc[i]
            else:
                res_df['short_ep_{}'.format(strat_version)] = res_df['short_ep_org_{}'.format(strat_version)]

            if config.out_set.static_out:
                res_df['short_out_{}'.format(strat_version)].iloc[i] = \
                res_df['short_out2_{}'.format(strat_version)].iloc[i]
            else:
                res_df['short_out_{}'.format(strat_version)] = res_df['short_out2_{}'.format(strat_version)]

            zone = 't'

    if mr_const_cnt == mr_score:
        open_side = OrderSide.SELL

    return res_df, open_side, zone


def long_ep_loc(res_df, config, i, show_detail=True):
    strat_version = config.strat_version

    # ------- param init ------- #
    open_side = None

    mr_const_cnt = 0
    mr_score = 0
    zone = 'n'

    if config.ep_set.entry_type == 'MARKET':
        if config.tp_set.tp_type != 'MARKET':
            tp_fee = config.trader_set.market_fee + config.trader_set.limit_fee
        else:
            tp_fee = config.trader_set.market_fee + config.trader_set.market_fee
        out_fee = config.trader_set.market_fee + config.trader_set.market_fee
    else:
        if config.tp_set.tp_type != 'MARKET':
            tp_fee = config.trader_set.limit_fee + config.trader_set.limit_fee
        else:
            tp_fee = config.trader_set.limit_fee + config.trader_set.market_fee
        out_fee = config.trader_set.limit_fee + config.trader_set.market_fee

    # -------------- spread scheduling -------------- #
    if config.loc_set.zone.long_spread != "None":

        mr_const_cnt += 1

        # spread = (res_df['bb_upper_5m'].iloc[i] - res_df['bb_base_5m'].iloc[i] - tp_fee * res_df['bb_base_5m'].iloc[
        #     i]) / (res_df['bb_base_5m'].iloc[i] - res_df['bb_lower_5m'].iloc[i] + out_fee *
        #             res_df['bb_base_5m'].iloc[i])
        # spread = (res_df['bb_upper_5m'].iloc[i] - res_df['bb_base_15m'].iloc[i] - tp_fee * res_df['bb_base_15m'].iloc[
        #     i]) / (res_df['bb_upper_5m'].iloc[i] - res_df['bb_base_15m'].iloc[i] + out_fee *
        #             res_df['bb_base_15m'].iloc[i])

        spread = (res_df['bb_upper_5m'].iloc[i] - res_df['dc_lower_5m'].iloc[i] - tp_fee * res_df['bb_upper_5m'].iloc[
            # spread = (res_df['bb_upper_5m'].iloc[i] - res_df['dc_lower_5m'].iloc[i] - out_fee * res_df['bb_upper_5m'].iloc[
            # i]) / (res_df['bb_upper_5m'].iloc[i] - res_df['dc_lower_5m'].iloc[i] + tp_fee *
            i]) / (res_df['bb_upper_5m'].iloc[i] - res_df['dc_lower_5m'].iloc[i] + out_fee *
                   res_df['bb_upper_5m'].iloc[i])
        # spread = (res_df['long_rtc_gap'].iloc[i] * (0.443) - tp_fee * res_df['long_ep'].iloc[
        #     i]) / (res_df['long_rtc_gap'].iloc[i] * (0.417) + out_fee * res_df['long_ep'].iloc[i])

        # spread = (res_df['dc_upper_5m'].iloc[i] - res_df['dc_lower_15m'].iloc[i] - tp_fee * res_df['dc_upper_5m'].iloc[
        #     i]) / (res_df['dc_upper_5m'].iloc[i] - res_df['dc_lower_15m'].iloc[i] + out_fee *
        #             res_df['dc_upper_5m'].iloc[i])
        # spread = ((res_df['dc_upper_5m'].iloc[i] - res_df['dc_lower_15m'].iloc[i])/2 - tp_fee * res_df['dc_upper_5m'].iloc[
        #     i]) / ((res_df['dc_upper_5m'].iloc[i] - res_df['dc_lower_15m'].iloc[i])/2 + out_fee *
        #             res_df['dc_upper_5m'].iloc[i])

        if spread >= config.loc_set.zone.long_spread:
            mr_score += 1

        if show_detail:
            print("spread :", spread)

    # -------------- dtk -------------- #
    if config.loc_set.zone.dt_k != "None":

        mr_const_cnt += 1
        # if res_df['dc_upper_%s' % config.loc_set.zone.dtk_dc_itv].iloc[i] <= res_df['long_rtc_1'].iloc[i] + res_df['long_rtc_gap'].iloc[i] * config.loc_set.zone.dt_k:
        #     dtk_v1 & v2 platform    #
        if config.loc_set.zone.dtk_dc_itv != "None":
            dc = res_df['dc_upper_%s' % config.loc_set.zone.dtk_dc_itv].iloc[i]
            dt_k = res_df['long_dtk_1_{}'.format(strat_version)].iloc[i] + \
                   res_df['long_dtk_gap_{}'.format(strat_version)].iloc[i] * config.loc_set.zone.dt_k
            if dc <= dt_k:
                mr_score += 1

        else:
            #     dc_v2     #
            dc = res_df['dc_upper_v2_{}'.format(strat_version)].iloc[i]
            dt_k = res_df['long_dtk_1_{}'.format(strat_version)].iloc[i] + \
                   res_df['long_dtk_gap_{}'.format(strat_version)].iloc[i] * config.loc_set.zone.dt_k
            if dc <= dt_k:
                # if res_df['dc_upper_v2_{}'.format(strat_version)].iloc[i] >= res_df['long_dtk_1_{}'.format(strat_version)].iloc[i] + res_df['long_dtk_gap_{}'.format(strat_version)].iloc[i] * config.loc_set.zone.dt_k:

                # if res_df['dc_upper_v2_{}'.format(strat_version)].iloc[i] <= res_df['long_dtk_1_{}'.format(strat_version)].iloc[i] + res_df['long_dtk_gap_{}'.format(strat_version)].iloc[i] * config.loc_set.zone.dt_k and \
                #   res_df['dc_lower_v2_{}'.format(strat_version)].iloc[i] >= res_df['short_dtk_1_{}'.format(strat_version)].iloc[i] - res_df['short_dtk_gap_{}'.format(strat_version)].iloc[i] * config.loc_set.zone.dt_k:

                mr_score += 1

        if show_detail:
            print("dc :", dc)
            print("dt_k :", dt_k)

    # -------------- zone rejection  -------------- #
    if config.loc_set.zone.zone_rejection:

        # #       by bb and dc - rsi      #
        # mr_const_cnt += 1
        # if res_df['bb_base_%s' % config.loc_set.zone.bbz_itv].iloc[i] >= res_df['dc_lower_1m'].iloc[i] >= res_df['bb_lower_%s' % config.loc_set.zone.bbz_itv].iloc[i]:
        #   mr_score += 1

        #       by bb       #       
        mr_const_cnt += 1
        # if  res_df['bb_lower_5m'].iloc[i] > res_df['bb_base_%s' % config.loc_set.zone.bbz_itv].iloc[i]:

        # if res_df['close'].iloc[i] > res_df['bb_upper_%s' % config.loc_set.zone.bbz_itv].iloc[i]:    # org
        # if res_df['close'].iloc[i] < res_df['bb_upper_%s' % config.loc_set.zone.bbz_itv].iloc[i]:  # inv
        # if res_df['close'].iloc[i] < res_df['bb_lower_%s' % config.loc_set.zone.bbz_itv].iloc[i]:
        if res_df['close'].iloc[i] < res_df['bb_lower2_%s' % config.loc_set.zone.bbz_itv].iloc[i]:
            # if res_df['close'].iloc[i] < res_df['bb_lower3_%s' % config.loc_set.zone.bbz_itv].iloc[i]:
            mr_score += 1

            #       by ema       #
        mr_const_cnt += 1
        # if  res_df['long_ep'].iloc[i] > res_df['ema_5m'].iloc[i]:
        if res_df['close'].iloc[i] > res_df['ema_5m'].iloc[i]:
            mr_score += 1

            #       by zone_dtk       #
        # mr_const_cnt += 1
        # if res_df['zone_dc_lower_v2_{}'.format(strat_version)].iloc[i] > res_df['short_dtk_plot_1'].iloc[i] - res_df['short_dtk_plot_gap'].iloc[i] * config.loc_set.zone.zone_dt_k:
        #   mr_score += 1

    # -------------- zoning -------------- #
    if config.tr_set.c_ep_gap != "None":
        #       by bb       # 
        # if res_df['close'].iloc[i] < res_df['bb_upper_%s' % config.loc_set.zone.bbz_itv].iloc[i]:

        #       by zone_dtk       #

        #         c_zone        #
        if res_df['zone_dc_lower_v2_{}'.format(strat_version)].iloc[i] < res_df['short_dtk_plot_1'].iloc[i] - \
                res_df['short_dtk_plot_gap'].iloc[i] * config.loc_set.zone.zone_dt_k:

            if config.ep_set.static_ep:
                res_df['long_ep_{}'.format(strat_version)].iloc[i] = res_df['long_ep2_{}'.format(strat_version)].iloc[i]
            else:
                res_df['long_ep_{}'.format(strat_version)] = res_df['long_ep2_{}'.format(strat_version)]

            if config.out_set.static_out:
                res_df['long_out_{}'.format(strat_version)].iloc[i] = \
                res_df['long_out_org_{}'.format(strat_version)].iloc[i]
            else:
                res_df['long_out_{}'.format(strat_version)] = res_df['long_out_org_{}'.format(strat_version)]

            zone = 'c'

            # mr_const_cnt += 1
            # dc_lb_period = 100
            # if np.sum((res_df['dc_upper_15m'] > res_df['dc_upper_15m'].shift(15)).iloc[i - dc_lb_period:i]) == 0:
            #   mr_score += 1

            #         t_zone        #
        else:

            # mr_const_cnt += 1   # zone_rejection - temporary

            if config.ep_set.static_ep:
                res_df['long_ep_{}'.format(strat_version)].iloc[i] = \
                res_df['long_ep_org_{}'.format(strat_version)].iloc[i]
            else:
                res_df['long_ep_{}'.format(strat_version)] = res_df['long_ep_org_{}'.format(strat_version)]

            if config.out_set.static_out:
                res_df['long_out_{}'.format(strat_version)].iloc[i] = res_df['long_out2_{}'.format(strat_version)].iloc[
                    i]
            else:
                res_df['long_out_{}'.format(strat_version)] = res_df['long_out2_{}'.format(strat_version)]

            zone = 't'

    if mr_const_cnt == mr_score:
        open_side = OrderSide.BUY

    return res_df, open_side, zone

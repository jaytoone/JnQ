from funcs.funcs_indicator import *
from funcs.funcs_trader import *
import logging

sys_log3 = logging.getLogger()


class OrderSide:
    BUY = "BUY"
    SELL = "SELL"
    INVALID = None


def lvrg_set(res_df, config, open_side, ep_, out_, fee, limit_leverage=50):
    strat_version = config.strat_version

    if open_side == OrderSide.SELL:

        if strat_version in ['v3']:
            # config.lvrg_set.leverage = config.lvrg_set.target_pct / (out_ / ep_ - 1 - (fee + config.trader_set.market_fee))
            config.lvrg_set.leverage = config.lvrg_set.target_pct / (
                    out_ / ep_ - 1 - (fee + config.trader_set.market_fee))

            #     zone 에 따른 c_ep_gap 를 고려 (loss 완화 방향) / 윗 줄은 수익 극대화 방향
            # config.lvrg_set.leverage = config.lvrg_set.target_pct / (out_ / res_df['short_ep_org'].iloc[ep_j] - 1 - (fee + config.trader_set.market_fee))

        elif strat_version in ['v5_2', 'v7_3']:
            # config.lvrg_set.leverage = config.lvrg_set.target_pct / abs(ep_ / out_ - 1 - (fee + config.trader_set.market_fee))
            config.lvrg_set.leverage = config.lvrg_set.target_pct / abs(
                ep_ / out_ - 1 - (fee + config.trader_set.market_fee))
            # config.lvrg_set.leverage = config.lvrg_set.target_pct / abs(res_df['short_ep_org'].iloc[ep_j] / out_ - 1 - (fee + config.trader_set.market_fee))

    else:
        #   윗 phase 는 min_pr 의 오차가 커짐
        if strat_version in ['v3']:
            # config.lvrg_set.leverage = config.lvrg_set.target_pct / (ep_ / out_ - 1 - (fee + config.trader_set.market_fee))
            config.lvrg_set.leverage = config.lvrg_set.target_pct / (
                    ep_ / out_ - 1 - (fee + config.trader_set.market_fee))
            # config.lvrg_set.leverage = config.lvrg_set.target_pct / (res_df['long_ep_org'].iloc[ep_j] / out_ - 1 - (fee + config.trader_set.market_fee))

        elif strat_version in ['v5_2', 'v7_3']:
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


def sync_check(df_, config, order_side="OPEN", row_slice=True):

    make_itv_list = [m_itv.replace('m', 'T') for m_itv in config.trader_set.itv_list]
    row_list = config.trader_set.row_list
    rec_row_list = config.trader_set.rec_row_list
    offset_list = config.trader_set.offset_list

    assert len(make_itv_list) == len(offset_list), "length of itv & offset_list should be equal"
    htf_df_list = [to_htf(df_, itv_=itv_, offset=offset_) for itv_idx, (itv_, offset_)
                   in enumerate(zip(make_itv_list, offset_list)) if itv_idx != 0]   #
    htf_df_list.insert(0, df_)

    # for htf_df_ in htf_df_list:
    #     print(htf_df_.tail())

    #       Todo        #
    #        1. row_list calc.
    #           a. indi. 를 만들기 위한 최소 period 가 존재하고, 그 indi. 를 사용한 lb_period 가 존재함
    #           b. => default_period + lb_period
    #               i. from sync_check, public_indi, ep_point2, ep_dur 의 tf 별 max lb_period check
    #                   1. default_period + max lb_period check
    #                       a. 현재까지 lb_period_list
    #                           h_prev_idx (open / close) 60
    #                           dc_period 135
    #                           zone_dc_period 135

    # --------- slicing (in trader phase only) --------- #
    #               --> latency 영향도가 높은 곳은 이곳
    if row_slice:   # recursive 가 아닌 indi. 의 latency 를 고려한 slicing
        df, df_5T, df_15T, df_30T, df_4H = [df_s.iloc[-row_list[row_idx]:].copy() for row_idx, df_s in enumerate(htf_df_list)]
        rec_df, rec_df_5T, rec_df_15T, rec_df_30T, rec_df_4H = [df_s.iloc[-rec_row_list[row_idx]:].copy() for row_idx, df_s in enumerate(htf_df_list)]
    else:
        df, df_5T, df_15T, df_30T, df_4H = htf_df_list
        rec_df, rec_df_5T, rec_df_15T, rec_df_30T, rec_df_4H = htf_df_list

    # --------- add indi. --------- #

    #        1. 필요한 indi. 는 enlist_epouttp & mr_check 보면서 삽입
    #        2. min use_rows 계산을 위해서, tf 별로 gathering 함        #
    start_0 = time.time()

    df = dc_line(df, None, '1m', dc_period=20)
    df = bb_line(df, None, '1m')
    # print(df.tail())

    df = dc_line(df, df_5T, '5m')
    df = bb_line(df, df_5T, '5m')

    df = dc_line(df, df_15T, '15m')
    df = bb_line(df, df_15T, '15m')

    df = bb_line(df, df_30T, '30m')

    # print(time.time() - start_0)

    # start_0 = time.time()
    df = bb_line(df, df_4H, '4h')
    # print(time.time() - start_0)

    rec_df['rsi_1m'] = rsi(rec_df, 14)  # Todo - recursive, 250 period
    df = df.join(to_lower_tf_v2(df, rec_df.iloc[:, [-1]], [-1], backing_i=0), how='inner')  # <-- join same_tf manual
    # print(df.rsi_1m.tail())

    if order_side in ["OPEN"]:
        rec_df_5T['ema_5m'] = ema(rec_df_5T['close'], 195)  # Todo - recursive, 1100 period (5T)
        df = df.join(to_lower_tf_v2(df, rec_df_5T, [-1]), how='inner')

    return df


def public_indi(res_df, config, np_timeidx, order_side="OPEN"):

    strat_version = config.strat_version
    res_df = dc_level(res_df, '5m', 1)
    res_df = bb_level(res_df, '5m', 1)
    # res_df = st_level(res_df, '5m', 1)

    res_df = dc_level(res_df, '15m', 1)
    res_df = bb_level(res_df, '15m', 1)
    res_df = dtk_plot(res_df, dtk_itv2='15m', hhtf_entry=15, use_dtk_line=config.loc_set.zone.use_dtk_line, np_timeidx=np_timeidx)

    # res_df = st_level(res_df, '15m', 1)

    # res_df = dc_level(res_df, '30m', 1)
    # res_df = bb_level(res_df, '30m', 1)
    # res_df = st_level(res_df, '30m', 1)

    # res_df = bb_level(res_df, '1h', 1)

    res_df = bb_level(res_df, '4h', 1)

    res_df['dc_upper_v2'.format(strat_version)] = res_df['high'].rolling(config.loc_set.zone.dc_period).max()
    res_df['dc_lower_v2'.format(strat_version)] = res_df['low'].rolling(config.loc_set.zone.dc_period).min()

    res_df['zone_dc_upper_v2'.format(strat_version)] = res_df['high'].rolling(config.loc_set.zone.zone_dc_period).max()
    res_df['zone_dc_lower_v2'.format(strat_version)] = res_df['low'].rolling(config.loc_set.zone.zone_dc_period).min()

    if order_side in ["OPEN"]:
        start_0 = time.time()

        res_df["wick_score"], res_df['body_score'] = candle_score(res_df, unsigned=False)

        # print("~ wick_score() elapsed time : {}".format(time.time() - start_0))

        start_0 = time.time()

        h_c_intv1 = '15T'
        h_c_intv2 = 'H'
        res_df = h_candle_v2(res_df, h_c_intv1)
        res_df = h_candle_v2(res_df, h_c_intv2)

        # sys_log3.warning("~ h_wick_score elapsed time : {}".format(time.time() - start_0))
        # print("wick_score() ~ h_candle() elapsed time : {}".format(time.time() - start_0))

        h_candle_col = ['hopen_{}'.format(h_c_intv2), 'hhigh_{}'.format(h_c_intv2), 'hlow_{}'.format(h_c_intv2),
                        'hclose_{}'.format(h_c_intv2)]

        res_df['h_wick_score'], res_df['h_body_score'] = candle_score(res_df, ohlc_col=h_candle_col, unsigned=False)

    #     temp indi.    #
    # res_df["ma30_1m"] = res_df['close'].rolling(30).mean()
    # res_df["ma60_1m"] = res_df['close'].rolling(60).mean()
    # res_df = dtk_plot(res_df, dtk_itv2='15m', hhtf_entry=15, use_dtk_line=config.loc_set.zone.use_dtk_line, np_timeidx=np_timeidx)

    return res_df


def short_point2(res_df, config, i, out_j, allow_ep_in):
    if config.strat_version == 'v5_2' and allow_ep_in == 0:

        if (res_df['dc_upper_1m'].iloc[i - 1] <= res_df['dc_upper_15m'].iloc[i]) & \
                (res_df['dc_upper_15m'].iloc[i - 1] != res_df['dc_upper_15m'].iloc[i]):
            # if (res_df['dc_upper_1m'].iloc[e_j - 1] <= res_df['dc_upper_5m'].iloc[e_j]) & \
            #     (res_df['dc_upper_5m'].iloc[e_j - 1] != res_df['dc_upper_5m'].iloc[e_j]):
            # (res_df['dc_upper_15m'].iloc[e_j] <= res_df['ema_5m'].iloc[e_j]):
            pass  # 일단, pass (non_logging)
        else:
            return 0, out_j  # input out_j could be initial_i

        # if res_df['ema_5m'].iloc[i] + res_df['dc_gap_5m'].iloc[i] * config.loc_set.point2.ce_gap <= res_df['close'].iloc[i]:
        #   pass
        # else:
        #   return 0, out_j

        # if res_df['bb_upper_5m'].iloc[i] <= res_df['ema_5m'].iloc[i]:
        #   pass
        # else:
        #   return 0, out_j

        return 1, i  # i = e_j

    return 1, out_j  # allow_ep_in, out_j (for idep)


def long_point2(res_df, config, i, out_j, allow_ep_in):
    if config.strat_version == 'v5_2' and allow_ep_in == 0:

        if (res_df['dc_lower_1m'].iloc[i - 1] >= res_df['dc_lower_15m'].iloc[i]) & \
                (res_df['dc_lower_15m'].iloc[i - 1] != res_df['dc_lower_15m'].iloc[i]):
            pass
        else:
            return 0, out_j  # input out_j could be initial_i

        # if res_df['ema_5m'].iloc[i] - res_df['dc_gap_5m'].iloc[i] * config.loc_set.point2.ce_gap >= res_df['close'].iloc[i]:
        #   pass
        # else:
        #   return 0, out_j

        # if res_df['bb_lower_5m'].iloc[i] >= res_df['ema_5m'].iloc[i]:
        #   pass
        # else:
        #   return 0, out_j

        return 1, i  # i = e_j

    return 1, out_j


def mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail):
    mr_const_cnt += 1
    if const_:
        mr_score += 1
    # else:
    #     if show_detail:
    #         pass
    #     else:
    #         return res_df, open_side, zone

    return mr_const_cnt, mr_score


def short_ep_loc(res_df, config, i, np_timeidx, show_detail=True):

    #       Todo : const warning msg manual        #
    #        1. just copy & paste const as string
    #           a. ex) const_ = a < b -> "a < b"
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

    # -------------- candle_score -------------- #
    if config.loc_set.point.wick_score != "None":

        # -------------- candle_score_v0 (1m initial tick 기준임)  -------------- #
        if strat_version in ['v5_2', '1_1']:

            wick_score = res_df['wick_score'].iloc[i]
            # body_score_ = res_df['body_score'].iloc[i]

            const_ = wick_score <= -config.loc_set.point.wick_score
            if show_detail:
                sys_log3.warning(
                    "wick_score <= -config.loc_set.point.wick_score : {:.5f} <= {:.5f} ({})".format(wick_score,
                                                                                                        -config.loc_set.point.wick_score,
                                                                                                        const_))

            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

        # -------------- candle_score_v1 (previous)  -------------- #
        if strat_version in ['v7_3', '2_2', 'v3']:

            prev_hclose_idx = i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1)

            if i < 0 or (i >= 0 and prev_hclose_idx >= 0):   # trader or colab env.

                h_wick_score = res_df['h_wick_score'].iloc[prev_hclose_idx]
                h_body_score = res_df['h_body_score'].iloc[prev_hclose_idx]

                if strat_version in ['v7_3']:

                    total_ratio = h_wick_score + h_body_score / 100

                    const_ = total_ratio <= -config.loc_set.point.wick_score
                    if show_detail:
                        sys_log3.warning(
                            "total_ratio <= -config.loc_set.point.wick_score : {:.5f} <= {:.5f} ({})".format(
                                total_ratio,
                                -config.loc_set.point.wick_score,
                                const_))

                    mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone,
                                                     show_detail)

                elif strat_version in ['2_2', 'v3', '1_1']:

                    const_ = h_wick_score <= -config.loc_set.point.wick_score
                    if show_detail:
                        sys_log3.warning(
                            "h_wick_score <= -config.loc_set.point.wick_score : {:.5f} <= {:.5f} ({})".format(
                                h_wick_score,
                                -config.loc_set.point.wick_score,
                                const_))

                    mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone,
                                                     show_detail)

                    if config.loc_set.point.body_score != "None":

                        const_ = h_body_score >= config.loc_set.point.body_score
                        if show_detail:
                            sys_log3.warning(
                                "h_body_score >= config.loc_set.point.body_score : {:.5f} >= {:.5f} ({})".format(
                                    h_body_score,
                                    config.loc_set.point.body_score,
                                    const_))

                        mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone,
                                                         show_detail)

    # ------------------ candle_score_v2 (current) ------------------ #
    if config.loc_set.point.wick_score2 != "None":  # wick_score1 과 동시적용 기능을 위해 병렬 구성

        h_wick_score = res_df['h_wick_score'].iloc[i]
        h_body_score = res_df['h_body_score'].iloc[i]
        ho = res_df['hopen_H'].iloc[i]
        hc = res_df['hclose_H'].iloc[i]

        const_ = h_wick_score <= -config.loc_set.point.wick_score2
        if show_detail:
            sys_log3.warning(
                "h_wick_score <= -config.loc_set.point.wick_score2 : {:.5f} <= {:.5f} ({})".format(h_wick_score,
                                                                                                       -config.loc_set.point.wick_score2,
                                                                                                       const_))

        mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

        if config.loc_set.point.body_score2 != "None":

            const_ = ho > hc and h_body_score >= config.loc_set.point.body_score2
            if show_detail:
                sys_log3.warning(
                    "ho > hc and h_body_score >= config.loc_set.point.body_score2 : {:.5f} > {:.5f} and {:.5f} >= {:.5f} ({})"
                        .format(ho, hc, h_body_score, config.loc_set.point.body_score2, const_))

            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

            # # -------------- tr scheduling -------------- #
    # if config.loc_set.zone.tr_thresh != "None":

    #
    #   tr = ((done_tp - ep_list[0] - tp_fee * ep_list[0]) / (ep_list[0] - done_out + out_fee * ep_list[0]))

    # -------------- spread scheduling -------------- #
    if config.loc_set.zone.short_spread != "None":

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

        const_ = spread >= config.loc_set.zone.short_spread
        if show_detail:
            sys_log3.warning("spread >= config.loc_set.zone.short_spread : {:.5f} >= {:.5f} ({})".format(spread,
                                                                                                         config.loc_set.zone.short_spread,
                                                                                                         const_))

        mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

        # -------------- dtk -------------- #
    if config.loc_set.zone.dt_k != "None":

        # if res_df['dc_lower_%s' % config.loc_set.zone.dtk_dc_itv].iloc[i] >= res_df['short_rtc_1'].iloc[i] - res_df['h_short_rtc_gap'].iloc[i] * config.loc_set.zone.dt_k:
        #     dtk_v1 & v2 platform     #
        if config.loc_set.zone.dtk_dc_itv != "None":
            dc = res_df['dc_lower_%s' % config.loc_set.zone.dtk_dc_itv].iloc[i]
            dt_k = res_df['short_dtk_1_{}'.format(strat_version)].iloc[i] - \
                   res_df['short_dtk_gap_{}'.format(strat_version)].iloc[i] * config.loc_set.zone.dt_k

            const_ = dc >= dt_k  # short 이라, tp_line 이 dc 보다 작아야하는게 맞음 (dc 가 이미 tp_line 에 다녀오면 안되니까)
            if show_detail:
                sys_log3.warning("dc >= dt_k : {:.5f} >= {:.5f} ({})".format(dc, dt_k, const_))

            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

            #     dc_v2   #
        else:
            dc = res_df['dc_lower_v2'.format(strat_version)].iloc[i]
            dt_k = res_df['short_dtk_1_{}'.format(strat_version)].iloc[i] - \
                   res_df['short_dtk_gap_{}'.format(strat_version)].iloc[i] * config.loc_set.zone.dt_k

            const_ = dc >= dt_k
            if show_detail:
                sys_log3.warning("dc >= dt_k : {:.5f} >= {:.5f} ({})".format(dc, dt_k, const_))

            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

            # -------------- candle_dt_k -------------- #
    # mr_const_cnt += 1
    # # if res_df['dc_lower_1m'].iloc[i] >= res_df['hclose60'].iloc[i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1)]:
    # if res_df['dc_lower_1m'].iloc[i] >= res_df['hlow_60'].iloc[i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1)]:
    #   mr_score += 1

    # mr_const_cnt += 1
    # if res_df['dc_upper_1m'].iloc[i] <= res_df['hhigh_60'].iloc[i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1)]:
    # # if res_df['dc_upper_1m'].iloc[i] <= res_df['hopen_60'].iloc[i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1)]:
    #   mr_score += 1

    # -------------- zone rejection  -------------- #
    if config.loc_set.zone.zone_rejection:

        #       config 로 통제할 수 없는 rejection 은 strat_version 으로 조건문을 나눔 (lvrg_set 과 동일)

        # --------- by bb --------- #

        #     bb & close   #
        if strat_version in ['v5_2']:

            close = res_df['close'].iloc[i]
            bb_upper2_ = res_df['bb_upper2_%s' % config.loc_set.zone.bbz_itv].iloc[i]

            const_ = bb_upper2_ < close
            if show_detail:
                sys_log3.warning("bb_upper2_ < close : {:.5f} < {:.5f} ({})".format(bb_upper2_, close, const_))

            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

            # bb_base_4h = res_df['bb_base_4h'].iloc[i]
            #
            # # const_ = close > bb_base_4h
            # const_ = close < bb_base_4h
            # if show_detail:
            #     sys_log3.warning("close < bb_base_4h : {:.5f} < {:.5f} ({})".format(close, bb_base_4h, const_))
            #
            # mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

            #     bb & bb   #
        if strat_version in ['v7_3', '1_1']:

            bb_upper_5m = res_df['bb_upper_5m'].iloc[i]
            bb_base_ = res_df['bb_base_%s' % config.loc_set.zone.bbz_itv].iloc[i]

            const_ = bb_upper_5m < bb_base_
            if show_detail:
                sys_log3.warning("bb_upper_5m < bb_base_ : {:.5f} < {:.5f} ({})".format(bb_upper_5m, bb_base_, const_))

            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

            #     bb & ep   #
            short_ep_ = res_df['short_ep_{}'.format(strat_version)].iloc[i]
            bb_upper_5m = res_df['bb_upper_5m'].iloc[i]

            const_ = bb_upper_5m > short_ep_
            if show_detail:
                sys_log3.warning(
                    "bb_upper_5m > short_ep_ : {:.5f} > {:.5f} ({})".format(bb_upper_5m, short_ep_, const_))

            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

            #     bb & dc   #

            # if res_df['bb_base_%s' % config.loc_set.zone.bbz_itv].iloc[i] <= res_df['dc_upper_1m'].iloc[i] <= res_df['bb_upper_%s' % config.loc_set.zone.bbz_itv].iloc[i]:

            prev_hopen_idx = i - (np_timeidx[
                                      i] % config.loc_set.zone.c_itv_ticks + 1 + config.loc_set.zone.c_itv_ticks) + config.loc_set.zone.ad_idx

            if i < 0 or (i >= 0 and prev_hopen_idx >= 0):  # considering ide & trader platform
                # if res_df['dc_upper_5m'].iloc[prev_hopen_idx] < res_df['bb_upper_15m'].iloc[i]:

                dc_upper_5m = res_df['dc_upper_5m'].iloc[prev_hopen_idx]
                bb_upper_15m = res_df['bb_upper_15m'].iloc[prev_hopen_idx]

                const_ = bb_upper_15m > dc_upper_5m
                if show_detail:
                    sys_log3.warning(
                        "bb_upper_15m > dc_upper_5m : {:.5f} > {:.5f} ({})".format(bb_upper_15m, dc_upper_5m, const_))

                mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

                # --------- by ema --------- #

            #    dc & ema   #
        if strat_version in ['v7_3', '1_1']:

            dc_upper_5m = res_df['dc_upper_5m'].iloc[i]
            ema_5m = res_df['ema_5m'].iloc[i]

            const_ = dc_upper_5m < ema_5m
            if show_detail:
                sys_log3.warning("dc_upper_5m < ema_5m : {:.5f} < {:.5f} ({})".format(dc_upper_5m, ema_5m, const_))

            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

            #    close, bb & ema   #
        if strat_version in ['v5_2', 'v3']:

            close = res_df['close'].iloc[i]
            ema_5m = res_df['ema_5m'].iloc[i]

            const_ = close < ema_5m
            if show_detail:
                sys_log3.warning("close < ema_5m : {:.5f} > {:.5f} ({})".format(close, ema_5m, const_))

            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

            # bb_upper_30m = res_df['bb_upper_30m'].iloc[i]

            # const_ = bb_upper_30m > ema_5m
            # if show_detail:
            #   sys_log3.warning("bb_upper_30m > ema_5m : {:.5f} > {:.5f} ({})".format(bb_upper_30m, ema_5m, const_))

            # mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

        # --------- by st --------- #
        # if strat_version in ['v5_2']:

        #   mr_const_cnt += 1
        #   if res_df['close'].iloc[i] < res_df['st_base_5m'].iloc[i]:
        #   # if res_df['close'].iloc[i] > res_df['st_base_5m'].iloc[i]:
        #       mr_score += 1

        #       if show_detail:
        #         sys_log3.warning("close & st : {:.5f} {:.5f} ({})".format(bb_, bb_1, const_))

        # --------- by dc --------- #

        #     descending dc    #
        # mr_const_cnt += 1
        # if res_df['dc_lower_5m'].iloc[i] <= res_df['dc_lower_5m'].iloc[i - 50 : i].min():
        #   mr_score += 1

        # --------- by candle --------- #
        # if strat_version in ['2_2']:

        #   prev_hclose_idx = i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1)
        #   if prev_hclose_idx >= 0:

        #     mr_const_cnt += 1
        #     # if res_df['short_ep_{}'.format(strat_version)].iloc[i] <= res_df['hclose60'].iloc[prev_hclose_idx)]:
        #     if res_df['close'].iloc[i] <= res_df['hclose60'].iloc[prev_hclose_idx]:
        #         mr_score += 1

        #     else:
        #       if show_detail:
        #       pass
        # else:
        #         return res_df, open_side, zone

        # --------- by macd --------- #
        # mr_const_cnt += 1
        # if res_df['ma30_1m'].iloc[i] < res_df['ma60_1m'].iloc[i]:
        #     mr_score += 1

        # --------- by zone_dtk --------- #
        # mr_const_cnt += 1
        # if res_df['zone_dc_upper_v2'.format(strat_version)].iloc[i] < res_df['long_dtk_plot_1'].iloc[i] + res_df['long_dtk_plot_gap'].iloc[
        #     i] * config.loc_set.zone.zone_dt_k:
        #   mr_score += 1

    # -------------- zoned tr_set - post_work -------------- #
    if config.tr_set.c_ep_gap != "None":

        #       by bb       #
        # if res_df['close'].iloc[i] > res_df['bb_lower_%s' % config.loc_set.zone.bbz_itv].iloc[i]:

        #       by zone_dtk       #

        #         c_zone        #
        zone_dc_upper_v2_ = res_df['zone_dc_upper_v2'.format(strat_version)].iloc[i]
        long_dtk_plot_ = res_df['long_dtk_plot_1'].iloc[i] + res_df['long_dtk_plot_gap'].iloc[
            i] * config.loc_set.zone.zone_dt_k

        const_ = zone_dc_upper_v2_ > long_dtk_plot_
        if show_detail:
            sys_log3.warning(
                "zone_dc_upper_v2_ > long_dtk_plot_ : {:.5f} > {:.5f} ({})".format(zone_dc_upper_v2_, long_dtk_plot_,
                                                                                   const_))

        if const_:
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

            #    # zone_rejection - temporary

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


def long_ep_loc(res_df, config, i, np_timeidx, show_detail=True):
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

    # -------------- candle_score -------------- #
    if config.loc_set.point.wick_score != "None":

        # -------------- candle_score_v0 (1m initial tick 기준임)  -------------- #
        if strat_version in ['v5_2', '1_1']:

            wick_score = res_df['wick_score'].iloc[i]
            # body_score_ = res_df['body_score'].iloc[i]

            const_ = wick_score >= config.loc_set.point.wick_score
            if show_detail:
                sys_log3.warning(
                    "wick_score >= config.loc_set.point.wick_score : {:.5f} >= {:.5f} ({})".format(wick_score,
                                                                                                       config.loc_set.point.wick_score,
                                                                                                       const_))

            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

            # -------------- candle_score_v1 (previous)  -------------- #
        if strat_version in ['v7_3', '2_2', 'v3']:

            prev_hclose_idx = i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1)

            if i < 0 or (i >= 0 and prev_hclose_idx >= 0):

                h_wick_score = res_df['h_wick_score'].iloc[prev_hclose_idx]
                h_body_score = res_df['h_body_score'].iloc[prev_hclose_idx]

                if strat_version in ['v7_3']:

                    total_ratio = h_wick_score + h_body_score / 100

                    const_ = total_ratio >= config.loc_set.point.wick_score
                    if show_detail:
                        sys_log3.warning(
                            "total_ratio >= config.loc_set.point.wick_score : {:.5f} >= {:.5f} ({})".format(
                                total_ratio,
                                config.loc_set.point.wick_score,
                                const_))

                    mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone,
                                                     show_detail)

                elif strat_version in ['2_2', 'v3', '1_1']:

                    const_ = h_wick_score >= config.loc_set.point.wick_score
                    if show_detail:
                        sys_log3.warning(
                            "h_wick_score >= config.loc_set.point.wick_score : {:.5f} >= {:.5f} ({})".format(
                                h_wick_score,
                                config.loc_set.point.wick_score,
                                const_))

                    mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone,
                                                     show_detail)

                    if config.loc_set.point.body_score != "None":

                        const_ = h_body_score >= config.loc_set.point.body_score
                        if show_detail:
                            sys_log3.warning(
                                "h_body_score >= config.loc_set.point.body_score : {:.5f} >= {:.5f} ({})".format(
                                    h_body_score,
                                    config.loc_set.point.wick_score,
                                    const_))

                        mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone,
                                                         show_detail)

    # ------------------ candle_score_v2 (current) ------------------ #
    if config.loc_set.point.wick_score2 != "None":

        h_wick_score = res_df['h_wick_score'].iloc[i]
        h_body_score = res_df['h_body_score'].iloc[i]
        ho = res_df['hopen_H'].iloc[i]
        hc = res_df['hclose_H'].iloc[i]

        const_ = h_wick_score >= config.loc_set.point.wick_score2
        if show_detail:
            sys_log3.warning(
                "h_wick_score >= config.loc_set.point.wick_score2 : {:.5f} >= {:.5f} ({})".format(h_wick_score,
                                                                                                      config.loc_set.point.wick_score2,
                                                                                                      const_))

        mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

        if config.loc_set.point.body_score2 != "None":

            const_ = ho < hc and h_body_score >= config.loc_set.point.body_score2
            if show_detail:
                sys_log3.warning(
                    "ho < hc and h_body_score >= config.loc_set.point.body_score2 : {:.5f} >= {:.5f} and {:.5f} >= {:.5f} ({})"
                        .format(ho, hc, h_body_score, config.loc_set.point.body_score2, const_))

            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

            # -------------- spread scheduling -------------- #
    if config.loc_set.zone.long_spread != "None":

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

        const_ = spread >= config.loc_set.zone.long_spread
        if show_detail:
            sys_log3.warning("spread >= config.loc_set.zone.long_spread : {:.5f} >= {:.5f} ({})".format(spread,
                                                                                                        config.loc_set.zone.long_spread,
                                                                                                        const_))

        mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

    # -------------- dtk -------------- #
    if config.loc_set.zone.dt_k != "None":

        # if res_df['dc_upper_%s' % config.loc_set.zone.dtk_dc_itv].iloc[i] <= res_df['long_rtc_1'].iloc[i] + res_df['long_rtc_gap'].iloc[i] * config.loc_set.zone.dt_k:
        #     dtk_v1 & v2 platform    #
        if config.loc_set.zone.dtk_dc_itv != "None":
            dc = res_df['dc_upper_%s' % config.loc_set.zone.dtk_dc_itv].iloc[i]
            dt_k = res_df['long_dtk_1_{}'.format(strat_version)].iloc[i] + \
                   res_df['long_dtk_gap_{}'.format(strat_version)].iloc[i] * config.loc_set.zone.dt_k

            const_ = dc <= dt_k
            if show_detail:
                sys_log3.warning("dc <= dt_k : {:.5f} <= {:.5f} ({})".format(dc, dt_k, const_))

            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

        else:
            #     dc_v2     #
            dc = res_df['dc_upper_v2'.format(strat_version)].iloc[i]
            dt_k = res_df['long_dtk_1_{}'.format(strat_version)].iloc[i] + \
                   res_df['long_dtk_gap_{}'.format(strat_version)].iloc[i] * config.loc_set.zone.dt_k

            const_ = dc <= dt_k
            if show_detail:
                sys_log3.warning("dc <= dt_k : {:.5f} <= {:.5f} ({})".format(dc, dt_k, const_))

            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

            # -------------- candle_dt_k -------------- #
    # mr_const_cnt += 1
    # # if res_df['dc_upper_1m'].iloc[i] <= res_df['hclose60'].iloc[i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1)]:
    # if res_df['dc_upper_1m'].iloc[i] <= res_df['hhigh_60'].iloc[i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1)]:
    #   mr_score += 1

    # mr_const_cnt += 1
    # if res_df['dc_lower_1m'].iloc[i] >= res_df['hlow_60'].iloc[i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1)]:
    # # if res_df['dc_lower_1m'].iloc[i] >= res_df['hopen_60'].iloc[i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1)]:
    #   mr_score += 1

    # -------------- zone rejection  -------------- #
    if config.loc_set.zone.zone_rejection:

        # --------- by bb --------- #

        #     bb & close   #
        if strat_version in ['v5_2']:

            close = res_df['close'].iloc[i]
            bb_lower2_ = res_df['bb_lower2_%s' % config.loc_set.zone.bbz_itv].iloc[i]

            const_ = bb_lower2_ > close
            if show_detail:
                sys_log3.warning("bb_lower2_ > close : {:.5f} > {:.5f} ({})".format(bb_lower2_, close, const_))
            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

            # bb_base_4h = res_df['bb_base_4h'].iloc[i]
            #
            # # const_ = close < bb_base_4h
            # const_ = close > bb_base_4h
            # if show_detail:
            #     sys_log3.warning("close > bb_base_4h : {:.5f} > {:.5f} ({})".format(close, bb_base_4h, const_))
            # mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

            #     bb & bb   #
        if strat_version in ['v7_3', '1_1']:

            bb_lower_5m = res_df['bb_lower_5m'].iloc[i]
            bb_base_ = res_df['bb_base_%s' % config.loc_set.zone.bbz_itv].iloc[i]

            const_ = bb_lower_5m > bb_base_
            if show_detail:
                sys_log3.warning("bb_lower_5m > bb_base_ : {:.5f} > {:.5f} ({})".format(bb_lower_5m, bb_base_, const_))

            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

            #     bb & ep   #
            long_ep_ = res_df['long_ep_{}'.format(strat_version)].iloc[i]
            bb_lower_5m = res_df['bb_lower_5m'].iloc[i]

            const_ = bb_lower_5m < long_ep_
            if show_detail:
                sys_log3.warning("bb_lower_5m < long_ep_ : {:.5f} < {:.5f} ({})".format(bb_lower_5m, long_ep_, const_))

            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

            #     bb & dc   #

            # if res_df['bb_base_%s' % config.loc_set.zone.bbz_itv].iloc[i] >= res_df['dc_lower_1m'].iloc[i] >= res_df['bb_lower_%s' % config.loc_set.zone.bbz_itv].iloc[i]:

            prev_hopen_idx = i - (np_timeidx[
                                      i] % config.loc_set.zone.c_itv_ticks + 1 + config.loc_set.zone.c_itv_ticks) + config.loc_set.zone.ad_idx

            if i < 0 or (i >= 0 and prev_hopen_idx >= 0):  # considering ide & trader platform
                # if res_df['dc_lower_5m'].iloc[prev_hopen_idx] > res_df['bb_lower_15m'].iloc[i]:

                dc_lower_5m = res_df['dc_lower_5m'].iloc[prev_hopen_idx]
                bb_lower_15m = res_df['bb_lower_15m'].iloc[prev_hopen_idx]

                const_ = bb_lower_15m < dc_lower_5m
                if show_detail:
                    sys_log3.warning(
                        "bb_lower_15m < dc_lower_5m : {:.5f} < {:.5f} ({})".format(bb_lower_15m, dc_lower_5m, const_))

                mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

        # --------- by ema --------- #

        #     dc & ema   #
        if strat_version in ['v7_3', '1_1']:

            dc_lower_5m = res_df['dc_lower_5m'].iloc[i]
            ema_5m = res_df['ema_5m'].iloc[i]

            const_ = dc_lower_5m > ema_5m
            if show_detail:
                sys_log3.warning("dc_lower_5m > ema_5m : {:.5f} > {:.5f} ({})".format(dc_lower_5m, ema_5m, const_))

            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

            #     close, bb & ema     #
        if strat_version in ['v5_2', 'v3']:

            close = res_df['close'].iloc[i]
            ema_5m = res_df['ema_5m'].iloc[i]

            const_ = close > ema_5m
            if show_detail:
                sys_log3.warning("close > ema_5m : {:.5f} > {:.5f} ({})".format(close, ema_5m, const_))

            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

            # bb_lower_30m = res_df['bb_lower_30m'].iloc[i]

            # const_ = bb_lower_30m < ema_5m
            # if show_detail:
            #     sys_log3.warning("bb_lower_30m < ema_5m : {:.5f} < {:.5f} ({})".format(bb_lower_30m, ema_5m, const_))

            # mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

        # if strat_version in ["2_2"]:

        #   mr_const_cnt += 1
        #   # if  res_df['long_ep'].iloc[i] > res_df['ema_5m'].iloc[i]:
        #   if res_df['close'].iloc[i] < res_df['ema_5m'].iloc[i]:
        #       mr_score += 1

        #       if show_detail:
        #         sys_log3.warning("close & ema : {:.5f} {:.5f} ({})".format(bb_, bb_1, const_))

        # --------- by st --------- #
        # if strat_version in ['v5_2']:
        #   mr_const_cnt += 1
        #   if res_df['close'].iloc[i] > res_df['st_base_5m'].iloc[i]:
        #   # if res_df['close'].iloc[i] < res_df['st_base_15m'].iloc[i]:
        #       mr_score += 1

        #       if show_detail:
        #         sys_log3.warning("close & st : {:.5f} {:.5f} ({})".format(bb_, bb_1, const_))

        # --------- by dc --------- #

        #     ascending dc    #
        # mr_const_cnt += 1
        # if res_df['dc_upper_5m'].iloc[i] >= res_df['dc_upper_5m'].iloc[i - 50 : i].max():
        #   mr_score += 1

        # --------- by candle --------- #
        # if strat_version in ['2_2']:

        #   prev_hclose_idx = i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1)
        #   if prev_hclose_idx >= 0:
        #     mr_const_cnt += 1
        #     if res_df['close'].iloc[i] >= res_df['hclose60'].iloc[prev_hclose_idx]:
        #         mr_score += 1
        #     else:
        #       if show_detail:
        #       pass
        # else:
        #         return res_df, open_side, zone

        # --------- by macd --------- #
        # mr_const_cnt += 1
        # if res_df['ma30_1m'].iloc[i] > res_df['ma60_1m'].iloc[i]:
        #     mr_score += 1

        # --------- by zone_dtk --------- #
        # mr_const_cnt += 1
        # if res_df['zone_dc_lower_v2'.format(strat_version)].iloc[i] > res_df['short_dtk_plot_1'].iloc[i] - res_df['short_dtk_plot_gap'].iloc[i] * config.loc_set.zone.zone_dt_k:
        #   mr_score += 1

    # -------------- zoned tr_set - post_work -------------- #
    if config.tr_set.c_ep_gap != "None":
        #       by bb       #
        # if res_df['close'].iloc[i] < res_df['bb_upper_%s' % config.loc_set.zone.bbz_itv].iloc[i]:

        #       by zone_dtk       #

        #         c_zone        #
        zone_dc_lower_v2_ = res_df['zone_dc_lower_v2'.format(strat_version)].iloc[i]
        short_dtk_plot_ = res_df['short_dtk_plot_1'].iloc[i] - res_df['short_dtk_plot_gap'].iloc[
            i] * config.loc_set.zone.zone_dt_k

        const_ = zone_dc_lower_v2_ < short_dtk_plot_
        if show_detail:
            sys_log3.warning(
                "zone_dc_lower_v2_ < short_dtk_plot_ : {:.5f} < {:.5f} ({})".format(zone_dc_lower_v2_, short_dtk_plot_,
                                                                                    const_))

        if const_:

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

            # 
            # dc_lb_period = 100
            # if np.sum((res_df['dc_upper_15m'] > res_df['dc_upper_15m'].shift(15)).iloc[i - dc_lb_period:i]) == 0:

            #         t_zone        #
        else:

            #    # zone_rejection - temporary

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

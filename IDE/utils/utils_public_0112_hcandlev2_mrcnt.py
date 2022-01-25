from funcs.funcs_indicator_candlescore import *
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


def sync_check(res_df_list, order_side="OPEN"):
    df, third_df, fourth_df = res_df_list

    #       add indi. only      #

    #       Todo : manual        #
    #        1. 필요한 indi. 는 enlist_epouttp & mr_check 보면서 삽입
    #        2. htf use_rows 는 1m use_rows 의 길이를 만족시킬 수 있는 정도
    #         a. 1m use_rows / htf_interval 하면 대략 나옴
    #         b. 또한, htf indi. 를 생성하기 위해 필요한 최소 row 이상
    df = dc_line(df, None, '1m', dc_period=20)
    df = dc_line(df, third_df, '5m')
    df = dc_line(df, fourth_df, '15m')

    df = bb_line(df, None, '1m')
    df = bb_line(df, third_df, '5m')
    df = bb_line(df, fourth_df, '15m')
    df = bb_line(df, fourth_df, '30m')

    df['rsi_1m'] = rsi(df, 14)

    if order_side in ["OPEN"]:
        third_df['ema_5m'] = ema(third_df['close'], 155)
        df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf_v2(df, third_df, [-1]), columns=['ema_5m']))

    return df


def public_indi(res_df, np_timeidx, order_side="OPEN"):
    res_df = dc_level(res_df, '5m', 1)
    res_df = dc_level(res_df, '15m', 1)
    # res_df = dc_level(res_df, '30m', 1)

    res_df = bb_level(res_df, '5m', 1)
    res_df = bb_level(res_df, '15m', 1)
    res_df = bb_level(res_df, '30m', 1)

    # res_df = st_level(res_df, '5m', 1)
    # res_df = st_level(res_df, '15m', 1)
    # res_df = st_level(res_df, '30m', 1)

    if order_side in ["OPEN"]:
        start_0 = time.time()

        res_df["candle_ratio"], res_df['body_ratio'] = candle_ratio(res_df, unsigned=False)

        # print("~ candle_ratio() elapsed time : {}".format(time.time() - start_0))

        start_0 = time.time()

        h_c_intv1 = '15T'
        h_c_intv2 = 'H'
        res_df = h_candle_v2(res_df, h_c_intv1)
        res_df = h_candle_v2(res_df, h_c_intv2)

        # sys_log3.warning("~ h_candle_ratio elapsed time : {}".format(time.time() - start_0))
        # print("candle_ratio() ~ h_candle() elapsed time : {}".format(time.time() - start_0))

        h_candle_col = ['hopen_{}'.format(h_c_intv2), 'hhigh_{}'.format(h_c_intv2), 'hlow_{}'.format(h_c_intv2),
                        'hclose_{}'.format(h_c_intv2)]

        res_df['h_candle_ratio'], res_df['h_body_ratio'] = candle_ratio(res_df, ohlc_col=h_candle_col, unsigned=False)

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

    return 1, out_j


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

    # -------------- candle_ratio -------------- #
    if config.loc_set.point.candle_ratio != "None":

        # -------------- candle_ratio_v0 (1m initial tick 기준임)  -------------- #
        if strat_version in ['v5_2', '1_1']:
            #
            candle_ratio_ = res_df['candle_ratio'].iloc[i]
            # body_ratio_ = res_df['body_ratio'].iloc[i]

            const_ = candle_ratio_ <= -config.loc_set.point.candle_ratio
            if show_detail:
                sys_log3.warning(
                    "candle_ratio_0 : {:.5f} <= {:.5f} ({})".format(candle_ratio_, -config.loc_set.point.candle_ratio,
                                                                    const_))

            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

        # -------------- candle_ratio_v1 (previous)  -------------- #
        if strat_version in ['v7_3', '2_2', 'v3']:

            prev_hclose_idx = i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1)

            if i < 0 or (i >= 0 and prev_hclose_idx >= 0):

                h_candle_ratio_ = res_df['h_candle_ratio'].iloc[prev_hclose_idx]
                h_body_ratio_ = res_df['h_body_ratio'].iloc[prev_hclose_idx]

                if strat_version in ['v7_3']:

                    total_ratio = h_candle_ratio_ + h_body_ratio_ / 100

                    const_ = total_ratio <= -config.loc_set.point.candle_ratio
                    if show_detail:
                        sys_log3.warning("candle_ratio1 : {:.5f} <= {:.5f} ({})".format(total_ratio,
                                                                                        -config.loc_set.point.candle_ratio,
                                                                                        const_))

                    mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone,
                                                     show_detail)

                elif strat_version in ['2_2', 'v3', '1_1']:

                    const_ = h_candle_ratio_ <= -config.loc_set.point.candle_ratio
                    if show_detail:
                        sys_log3.warning("candle_ratio1 : {:.5f} <= {:.5f} ({})".format(h_candle_ratio_,
                                                                                        -config.loc_set.point.candle_ratio,
                                                                                        const_))

                    mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone,
                                                     show_detail)

                    if config.loc_set.point.body_ratio != "None":

                        const_ = h_body_ratio_ >= config.loc_set.point.body_ratio
                        if show_detail:
                            sys_log3.warning("body_ratio1 : {:.5f} >= {:.5f} ({})".format(h_body_ratio_,
                                                                                          config.loc_set.point.body_ratio,
                                                                                          const_))

                        mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone,
                                                         show_detail)

    if config.loc_set.point.candle_ratio2 != "None":  # candle_ratio1 과 동시적용 기능을 위해 병렬 구성

        #     candle_ratio_v2 (current)     #
        h_candle_ratio_ = res_df['h_candle_ratio'].iloc[i]
        h_body_ratio_ = res_df['h_body_ratio'].iloc[i]
        ho = res_df['hopen_H']
        hc = res_df['hclose_H']

        const_ = h_candle_ratio_ <= -config.loc_set.point.candle_ratio2
        if show_detail:
            sys_log3.warning(
                "candle_ratio2 : {:.5f} <= {:.5f} ({})".format(h_candle_ratio_, -config.loc_set.point.candle_ratio2,
                                                               const_))

        mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

        if config.loc_set.point.body_ratio2 != "None":

            const_ = ho > hc and h_body_ratio_ >= config.loc_set.point.body_ratio2
            if show_detail:
                sys_log3.warning("ho > hc : {:.5f} >= {:.5f}, body_ratio2 : {:.5f} >= {:.5f} ({})"
                                 .format(ho, hc, h_body_ratio_, config.loc_set.point.body_ratio2, const_))

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
            sys_log3.warning("spread : {:.5f} >= {:.5f} ({})".format(spread, config.loc_set.zone.short_spread, const_))

        mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

        # -------------- dtk -------------- #
    if config.loc_set.zone.dt_k != "None":

        # if res_df['dc_lower_%s' % config.loc_set.zone.dtk_dc_itv].iloc[i] >= res_df['short_rtc_1'].iloc[i] - res_df['h_short_rtc_gap'].iloc[i] * config.loc_set.zone.dt_k:
        #     dtk_v1 & v2 platform     #
        if config.loc_set.zone.dtk_dc_itv != "None":
            dc = res_df['dc_lower_%s' % config.loc_set.zone.dtk_dc_itv].iloc[i]
            dt_k = res_df['short_dtk_1_{}'.format(strat_version)].iloc[i] - \
                   res_df['short_dtk_gap_{}'.format(strat_version)].iloc[i] * config.loc_set.zone.dt_k

            const_ = dc >= dt_k
            if show_detail:
                sys_log3.warning("dc >= dt_k : {:.5f} >= {:.5f} ({})".format(dc, dt_k, const_))

            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

            #     dc_v2   #
        else:
            dc = res_df['dc_lower_v2_{}'.format(strat_version)].iloc[i]
            dt_k = res_df['short_dtk_1_{}'.format(strat_version)].iloc[i] - \
                   res_df['short_dtk_gap_{}'.format(strat_version)].iloc[i] * config.loc_set.zone.dt_k

            const_ = dc >= dt_k
            if show_detail:
                sys_log3.warning("dc >= dt_k : {:.5f} >= {:.5f} ({})".format(dc, dt_k, const_))

            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

                # -------------- candle_dt_k -------------- #
    # mr_const_cnt += 1
    # # if res_df['dc_lower_1m'].iloc[i] >= res_df['hclose_60'].iloc[i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1)]:
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

            close_ = res_df['close'].iloc[i]
            # bb_ = res_df['bb_lower_%s' % config.loc_set.zone.bbz_itv].iloc[i]
            # bb_ = res_df['bb_upper_%s' % config.loc_set.zone.bbz_itv].iloc[i]
            bb_ = res_df['bb_upper2_%s' % config.loc_set.zone.bbz_itv].iloc[i]
            # bb_ = res_df['bb_upper3_%s' % config.loc_set.zone.bbz_itv].iloc[i]

            const_ = bb_ < close_
            if show_detail:
                sys_log3.warning("bb_upper2_ & close : {:.5f} < {:.5f} ({})".format(bb_, close_, const_))

            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

                #     bb & bb   #
        if strat_version in ['v7_3', '1_1']:

            bb_ = res_df['bb_upper_5m'].iloc[i]
            bb_1 = res_df['bb_base_%s' % config.loc_set.zone.bbz_itv].iloc[i]

            const_ = bb_ < bb_1
            if show_detail:
                sys_log3.warning("bb_upper_5m & bb_base_ : {:.5f} < {:.5f} ({})".format(bb_, bb_1, const_))

            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

                #     bb & ep   #
            ep_ = res_df['short_ep_{}'.format(strat_version)].iloc[i]
            bb_ = res_df['bb_upper_5m'].iloc[i]

            const_ = bb_ > ep_
            if show_detail:
                sys_log3.warning("bb_upper_5m & short_ep_ : {:.5f} > {:.5f} ({})".format(bb_, ep_, const_))

            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

            #     bb & dc   #

            # if res_df['bb_base_%s' % config.loc_set.zone.bbz_itv].iloc[i] <= res_df['dc_upper_1m'].iloc[i] <= res_df['bb_upper_%s' % config.loc_set.zone.bbz_itv].iloc[i]:

            prev_hopen_idx = i - (np_timeidx[
                                      i] % config.loc_set.zone.c_itv_ticks + 1 + config.loc_set.zone.c_itv_ticks) + config.loc_set.zone.ad_idx

            if i < 0 or (i >= 0 and prev_hopen_idx >= 0):  # considering ide & trader platform
                # if res_df['dc_upper_5m'].iloc[prev_hopen_idx] < res_df['bb_upper_15m'].iloc[i]:

                dc_ = res_df['dc_upper_5m'].iloc[prev_hopen_idx]
                bb_ = res_df['bb_upper_15m'].iloc[prev_hopen_idx]

                const_ = bb_ > dc_
                if show_detail:
                    sys_log3.warning("bb_upper_15m & dc_upper_5m : {:.5f} > {:.5f} ({})".format(bb_, dc_, const_))

                mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

                # --------- by ema --------- #

            #    dc & ema   #
        if strat_version in ['v7_3', '1_1']:

            dc_ = res_df['dc_upper_5m'].iloc[i]
            ema_ = res_df['ema_5m'].iloc[i]

            const_ = dc_ < ema_
            if show_detail:
                sys_log3.warning("dc_upper_5m & ema_5m : {:.5f} < {:.5f} ({})".format(dc_, ema_, const_))

            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

                #    close, bb & ema   #
        if strat_version in ['v5_2', 'v3']:

            close_ = res_df['close'].iloc[i]
            ema_ = res_df['ema_5m'].iloc[i]
            bb_ = res_df['bb_upper_30m'].iloc[i]

            const_ = ema_ > close_
            if show_detail:
                sys_log3.warning("close & ema_5m : {:.5f} > {:.5f} ({})".format(close_, ema_, const_))

            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

            # if show_detail:
            #   sys_log3.warning("bb_upper_30m & ema_5m : {:.5f} > {:.5f} ({})".format(bb_, ema_, const_))

            # mr_const_cnt += 1
            # if ema_ < bb_:
            #   mr_score += 1
            # else:
            #   if show_detail:
            #   pass
            # else:
            #         return res_df, open_side, zone

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
        #     # if res_df['short_ep_{}'.format(strat_version)].iloc[i] <= res_df['hclose_60'].iloc[prev_hclose_idx)]:
        #     if res_df['close'].iloc[i] <= res_df['hclose_60'].iloc[prev_hclose_idx]:
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
        # if res_df['zone_dc_upper_v2_{}'.format(strat_version)].iloc[i] < res_df['long_dtk_plot_1'].iloc[i] + res_df['long_dtk_plot_gap'].iloc[
        #     i] * config.loc_set.zone.zone_dt_k:
        #   mr_score += 1

    # -------------- zoned tr_set - post_work -------------- #
    if config.tr_set.c_ep_gap != "None":

        #       by bb       # 
        # if res_df['close'].iloc[i] > res_df['bb_lower_%s' % config.loc_set.zone.bbz_itv].iloc[i]:

        #       by zone_dtk       #

        #         c_zone        #
        zone_dc_ = res_df['zone_dc_upper_v2_{}'.format(strat_version)].iloc[i]
        l_dtk_ = res_df['long_dtk_plot_1'].iloc[i] + res_df['long_dtk_plot_gap'].iloc[i] * config.loc_set.zone.zone_dt_k

        const_ = zone_dc_ > l_dtk_
        if show_detail:
            sys_log3.warning(
                "zone_dc_upper_v2_ & long_dtk_plot_ : {:.5f} > {:.5f} ({})".format(zone_dc_, l_dtk_, const_))

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

    # -------------- candle_ratio -------------- #
    if config.loc_set.point.candle_ratio != "None":

        # -------------- candle_ratio_v0 (1m initial tick 기준임)  -------------- #
        if strat_version in ['v5_2', '1_1']:

            candle_ratio_ = res_df['candle_ratio'].iloc[i]
            # body_ratio_ = res_df['body_ratio'].iloc[i]

            const_ = candle_ratio_ >= config.loc_set.point.candle_ratio
            if show_detail:
                sys_log3.warning(
                    "candle_ratio_0 : {:.5f} >= {:.5f} ({})".format(candle_ratio_, config.loc_set.point.candle_ratio,
                                                                    const_))

            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

            # -------------- candle_ratio_v1 (previous)  -------------- #
        if strat_version in ['v7_3', '2_2', 'v3']:

            prev_hclose_idx = i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1)

            if i < 0 or (i >= 0 and prev_hclose_idx >= 0):

                h_candle_ratio_ = res_df['h_candle_ratio'].iloc[prev_hclose_idx]
                h_body_ratio_ = res_df['h_body_ratio'].iloc[prev_hclose_idx]

                if strat_version in ['v7_3']:

                    total_ratio = h_candle_ratio_ + h_body_ratio_ / 100

                    const_ = total_ratio >= config.loc_set.point.candle_ratio
                    if show_detail:
                        sys_log3.warning("candle_ratio1 : {:.5f} >= {:.5f} ({})".format(total_ratio,
                                                                                        config.loc_set.point.candle_ratio,
                                                                                        const_))

                    mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

                elif strat_version in ['2_2', 'v3', '1_1']:

                    const_ = h_candle_ratio_ >= config.loc_set.point.candle_ratio
                    if show_detail:
                        sys_log3.warning("candle_ratio1 : {:.5f} >= {:.5f} ({})".format(h_candle_ratio_,
                                                                                        config.loc_set.point.candle_ratio,
                                                                                        const_))

                    mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone,
                                                     show_detail)

                    if config.loc_set.point.body_ratio != "None":

                        const_ = h_body_ratio_ >= config.loc_set.point.body_ratio
                        if show_detail:
                            sys_log3.warning("body_ratio1 : {:.5f} >= {:.5f} ({})".format(h_body_ratio_,
                                                                                          config.loc_set.point.candle_ratio,
                                                                                          const_))

                        mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone,
                                                         show_detail)

    if config.loc_set.point.candle_ratio2 != "None":

        #     candle_ratio_v2 (current)     #
        h_candle_ratio_ = res_df['h_candle_ratio'].iloc[i]
        h_body_ratio_ = res_df['h_body_ratio'].iloc[i]
        ho = res_df['hopen_H']
        hc = res_df['hclose_H']

        const_ = h_candle_ratio_ >= config.loc_set.point.candle_ratio2
        if show_detail:
            sys_log3.warning(
                "candle_ratio2 : {:.5f} >= {:.5f} ({})".format(h_candle_ratio_, config.loc_set.point.candle_ratio2,
                                                               const_))

        mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

        if config.loc_set.point.body_ratio2 != "None":

            const_ = ho < hc and h_body_ratio_ >= config.loc_set.point.body_ratio2
            if show_detail:
                sys_log3.warning("ho < hc : {:.5f} >= {:.5f} body_ratio2 : {:.5f} >= {:.5f} ({})"
                                 .format(ho, hc, h_body_ratio_, config.loc_set.point.body_ratio2, const_))

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
            sys_log3.warning("spread : {:.5f} >= {:.5f} ({})".format(spread, config.loc_set.zone.long_spread, const_))

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
                sys_log3.warning("dc : {:.5f} <= {:.5f} ({})".format(dc, dt_k, const_))

            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

        else:
            #     dc_v2     #
            dc = res_df['dc_upper_v2_{}'.format(strat_version)].iloc[i]
            dt_k = res_df['long_dtk_1_{}'.format(strat_version)].iloc[i] + \
                   res_df['long_dtk_gap_{}'.format(strat_version)].iloc[i] * config.loc_set.zone.dt_k

            const_ = dc <= dt_k
            if show_detail:
                sys_log3.warning("dc : {:.5f} <= {:.5f} ({})".format(dc, dt_k, const_))

            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

            # -------------- candle_dt_k -------------- #
    # mr_const_cnt += 1
    # # if res_df['dc_upper_1m'].iloc[i] <= res_df['hclose_60'].iloc[i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1)]:
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

            close_ = res_df['close'].iloc[i]
            # bb_ = res_df['bb_lower_%s' % config.loc_set.zone.bbz_itv].iloc[i]
            # bb_ = res_df['bb_upper_%s' % config.loc_set.zone.bbz_itv].iloc[i]
            bb_ = res_df['bb_lower2_%s' % config.loc_set.zone.bbz_itv].iloc[i]
            # bb_ = res_df['bb_lower3_%s' % config.loc_set.zone.bbz_itv].iloc[i]

            const_ = bb_ > close_
            if show_detail:
                sys_log3.warning("bb_lower2_ & close : {:.5f} > {:.5f} ({})".format(bb_, close_, const_))

            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

            #     bb & bb   #
        if strat_version in ['v7_3', '1_1']:

            bb_ = res_df['bb_lower_5m'].iloc[i]
            bb_1 = res_df['bb_base_%s' % config.loc_set.zone.bbz_itv].iloc[i]

            const_ = bb_ > bb_1
            if show_detail:
                sys_log3.warning("bb_lower_5m & bb_base_ : {:.5f} > {:.5f} ({})".format(bb_, bb_1, const_))

            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

                #     bb & ep   #
            ep_ = res_df['long_ep_{}'.format(strat_version)].iloc[i]
            bb_ = res_df['bb_lower_5m'].iloc[i]

            const_ = bb_ < ep_
            if show_detail:
                sys_log3.warning("bb_lower_5m & long_ep_ : {:.5f} < {:.5f} ({})".format(bb_, ep_, const_))

            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

            #     bb & dc   #

            # if res_df['bb_base_%s' % config.loc_set.zone.bbz_itv].iloc[i] >= res_df['dc_lower_1m'].iloc[i] >= res_df['bb_lower_%s' % config.loc_set.zone.bbz_itv].iloc[i]:

            prev_hopen_idx = i - (np_timeidx[
                                      i] % config.loc_set.zone.c_itv_ticks + 1 + config.loc_set.zone.c_itv_ticks) + config.loc_set.zone.ad_idx

            if i < 0 or (i >= 0 and prev_hopen_idx >= 0):  # considering ide & trader platform
                # if res_df['dc_lower_5m'].iloc[prev_hopen_idx] > res_df['bb_lower_15m'].iloc[i]:

                dc_ = res_df['dc_lower_5m'].iloc[prev_hopen_idx]
                bb_ = res_df['bb_lower_15m'].iloc[prev_hopen_idx]

                const_ = bb_ < dc_
                if show_detail:
                    sys_log3.warning("bb_lower_15m & dc_lower_5m : {:.5f} < {:.5f} ({})".format(bb_, dc_, const_))

                mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

        # --------- by ema --------- # 

        #     dc & ema   #
        if strat_version in ['v7_3', '1_1']:

            dc_ = res_df['dc_lower_5m'].iloc[i]
            ema_ = res_df['ema_5m'].iloc[i]

            const_ = dc_ > ema_
            if show_detail:
                sys_log3.warning("dc_lower_5m & ema_5m : {:.5f} > {:.5f} ({})".format(dc_, ema_, const_))

            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

            #     close, bb & ema     #
        if strat_version in ['v5_2', 'v3']:

            close_ = res_df['close'].iloc[i]
            ema_ = res_df['ema_5m'].iloc[i]
            bb_ = res_df['bb_lower_30m'].iloc[i]

            const_ = close_ > ema_
            if show_detail:
                sys_log3.warning("close & ema_5m : {:.5f} > {:.5f} ({})".format(close_, ema_, const_))

            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, res_df, open_side, zone, show_detail)

            # if show_detail:
            #     sys_log3.warning("bb_lower_30m & ema_5m : {:.5f} < {:.5f} ({})".format(bb_, ema_, const_))

            # mr_const_cnt += 1
            # if bb_ < ema_:
            #   mr_score += 1
            # else:
            #   if show_detail:
            #   pass
            # else:
            #         return res_df, open_side, zone

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
        #     if res_df['close'].iloc[i] >= res_df['hclose_60'].iloc[prev_hclose_idx]:
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
        # if res_df['zone_dc_lower_v2_{}'.format(strat_version)].iloc[i] > res_df['short_dtk_plot_1'].iloc[i] - res_df['short_dtk_plot_gap'].iloc[i] * config.loc_set.zone.zone_dt_k:
        #   mr_score += 1

    # -------------- zoned tr_set - post_work -------------- #
    if config.tr_set.c_ep_gap != "None":
        #       by bb       # 
        # if res_df['close'].iloc[i] < res_df['bb_upper_%s' % config.loc_set.zone.bbz_itv].iloc[i]:

        #       by zone_dtk       #

        #         c_zone        #
        zone_dc_ = res_df['zone_dc_lower_v2_{}'.format(strat_version)].iloc[i]
        l_dtk_ = res_df['short_dtk_plot_1'].iloc[i] - res_df['short_dtk_plot_gap'].iloc[
            i] * config.loc_set.zone.zone_dt_k

        const_ = zone_dc_ < l_dtk_
        if show_detail:
            sys_log3.warning(
                "zone_dc_lower_v2_ & short_dtk_plot : {:.5f} < {:.5f} ({})".format(zone_dc_, l_dtk_, const_))

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

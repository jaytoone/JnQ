from funcs.funcs_duration import *


#       Todo        #
#        1. future_data warning
#        2. lookback indi. warning - index error
#        3. ep_loc 의 i 는 close 기준임 -> 즉, last_index = -2 warning


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
                   in enumerate(zip(make_itv_list, offset_list)) if itv_idx != 0]  #
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
    if row_slice:  # recursive 가 아닌 indi. 의 latency 를 고려한 slicing
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
    #       Todo        #
    #        1. future data warning
    res_df["ema_1m"] = ema(res_df['close'], 200)

    strat_version = config.strat_version
    res_df = dc_level(res_df, '5m', 1)
    res_df = bb_level(res_df, '5m', 1)
    # res_df = st_level(res_df, '5m', 1)

    res_df = dc_level(res_df, '15m', 1)
    res_df = bb_level(res_df, '15m', 1)
    res_df = dtk_plot(res_df, dtk_itv2='15m', hhtf_entry=15, use_dtk_line=config.loc_set.zone.use_dtk_line, np_timeidx=np_timeidx)

    norm_cols = ['open', 'high', 'low', 'close', 'ema_1m']
    target_col = ['ema_1m']
    norm_period = 120
    rolled_data = res_df[norm_cols].rolling(norm_period)
    res_df['norm_range'] = rolled_data.max().max(axis=1) - rolled_data.min().min(axis=1)
    res_df['target_diff'] = res_df[target_col].diff(norm_period)

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

        #       score 로 표현한 이유 : ratio 의 의미를 드러냄        #
        res_df["front_wick_score"], res_df['body_score'], res_df["back_wick_score"] = candle_score(res_df)
        # print("~ wick_score() elapsed time : {}".format(time.time() - start_0))

        start_0 = time.time()

        h_c_intv1 = '15T'
        h_c_intv2 = 'H'
        res_df = h_candle_v2(res_df, h_c_intv1)
        res_df = h_candle_v2(res_df, h_c_intv2)

        # sys_log3.warning("~ h_wick_score elapsed time : {}".format(time.time() - start_0))
        # print("wick_score() ~ h_candle() elapsed time : {}".format(time.time() - start_0))

        h_c_itv = h_c_intv1
        h_candle_col = ['hopen_{}'.format(h_c_itv), 'hhigh_{}'.format(h_c_itv), 'hlow_{}'.format(h_c_itv),
                        'hclose_{}'.format(h_c_itv)]

        res_df['h_front_wick_score'], res_df['h_body_score'], res_df['h_back_wick_score'] = candle_score(res_df, ohlc_col=h_candle_col)

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


def ep_loc(res_df, config, i, np_timeidx, show_detail=True, ep_loc_side=OrderSide.SELL):
    #       Todo : const warning msg manual        #
    #        1. just copy & paste const as string
    #           a. ex) const_ = a < b -> "a < b"
    strat_version = config.strat_version

    # ------- param init ------- #
    open_side = None

    mr_const_cnt = 0
    mr_score = 0
    zone = 'n'

    # ------------------ candle_score ------------------ #
    # ------ candle_score_v0 (1m initial tick 기준임)  ------ #
    if config.loc_set.zone.front_wick_score0 != "None":
        #   i - warning checked     #
        candle_game0(ep_loc_side, res_df, config, i, mr_const_cnt, mr_score, show_detail)

    # ------ candle_score_v1 (previous)  ------ #
    wick_score_arr = np.array([config.loc_set.zone.front_wick_score, config.loc_set.zone.body_score, config.loc_set.zone.back_wick_score])
    assert wick_score_arr.size % 3 == 0, "wick_score_arr.size % 3 == 0"
    if np.sum(wick_score_arr == "None") != wick_score_arr.size:
        # Todo - public 에서 정의 가능할 것
        # ID3_1 기준 - 현재까지, i 의 intmin 은 c_itv_ticks - 1 이기 때문에 아래의 logic 사용 가능함
        if strat_version in ['3_1']:
            prev_hclose_idx_list = [i - config.loc_set.zone.c_itv_ticks * nbar_ for nbar_ in range(len(config.loc_set.zone.front_wick_score))]
        else:
            prev_hclose_idx_list = [i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1 + config.loc_set.zone.c_itv_ticks * nbar_)
                                    for nbar_ in range(len(config.loc_set.zone.front_wick_score))]
        for score_idx, prev_hclose_idx in enumerate(prev_hclose_idx_list):    # Todo - recursive on mr_const & score
            mr_const_cnt, mr_score = candle_game(ep_loc_side, res_df, config, i, score_idx, prev_hclose_idx, mr_const_cnt, mr_score, show_detail)

    # ------ candle_score_v2 (current) ------ #
    # 1. wick_score1 과 동시적용 기능을 위해 병렬 구성
    # 2. body_score 은 > 부호만 허용, else_score 는 양부호 (<, >) 허용 (음수 존재함)
    #   a. front <-> back_score 의 부호는 상반되어야할 것
    wick_score2_arr = np.array([config.loc_set.zone.front_wick_score2, config.loc_set.zone.body_score2, config.loc_set.zone.back_wick_score2])
    if np.sum(wick_score2_arr == "None") != wick_score2_arr.size:
        # => future_data warning 으로 아래와같이 변경함
        #   i - warning checked     #
        shared_ticks = np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1  # Todo - public 에서 정의 가능할 것
        candle_game2(ep_loc_side, res_df, config, i, shared_ticks, mr_const_cnt, mr_score, show_detail)

    # ------------------ spread scheduling ------------------ #
    if config.loc_set.zone.short_spread != "None":
        spread(ep_loc_side, res_df, config, i, mr_const_cnt, mr_score, show_detail)

    # ------------------ dtk ------------------ #
    if config.loc_set.zone.dt_k != "None":
        dtk(ep_loc_side, res_df, config, i, mr_const_cnt, mr_score, show_detail)

    # ------------------ zone rejection  ------------------ #
    if config.loc_set.zone.zone_rejection:
    #       config 로 통제할 수 없는 rejection 은 strat_version 으로 조건문을 나눔 (lvrg_set 과 동일)
    # ------------ by bb ------------ #
        if strat_version in ['v5_2']:
            bb_lower2_close(ep_loc_side, res_df, config, i, mr_const_cnt, mr_score, show_detail)
            bb_base_4h_close(ep_loc_side, res_df, config, i, mr_const_cnt, mr_score, show_detail)

        if strat_version in ['v7_3', '1_1']:
            bb_upper_5m_bb_base_(ep_loc_side, res_df, config, i, mr_const_cnt, mr_score, show_detail)
            bb_upper_5m_ep_(ep_loc_side, res_df, config, i, mr_const_cnt, mr_score, show_detail)
            # Todo - prev_hopen_idx -> public 에서 정의 가능할 것
            #  기존에, +1 을 한 이유는, i 가 0 min 일 경우 prev_hopen 을 구하기 위함임, 이곳의 hopen_idx 개념이 모호함
            # prev_hopen_idx = i - (np_timeidx[
            #                           i] % config.loc_set.zone.c_itv_ticks + 1 + config.loc_set.zone.c_itv_ticks) + config.loc_set.zone.ad_idx
            # dc_upper_5m_bb_upper_15m(ep_loc_side, res_df, config, i, prev_hopen_idx, mr_const_cnt, mr_score, show_detail)

    # ------------ by ema ------------ #
        if strat_version in ['v7_3', '1_1']:
            dc_upper_5m_ema_5m(ep_loc_side, res_df, config, i, mr_const_cnt, mr_score, show_detail)

        if strat_version in ['v5_2', 'v3']:
            close_ema_5m(ep_loc_side, res_df, config, i, mr_const_cnt, mr_score, show_detail)

        if strat_version in ['3']:
            degree(ep_loc_side, res_df, config, i, mr_const_cnt, mr_score, show_detail)
            # hopen_H_ema_5m(ep_loc_side, res_df, config, i, mr_const_cnt, mr_score, show_detail)

    # ------------ by dc ------------ #
    # descending_dc(ep_loc_side, res_df, config, i, mr_const_cnt, mr_score, show_detail)

    # -------------- zoned tr_set - post_work -------------- #
    if config.tr_set.c_ep_gap != "None":
        res_df, zone = zoned_tr_set(ep_loc_side, res_df, config, i, zone, show_detail)

    if mr_const_cnt == mr_score:
        open_side = ep_loc_side

    return res_df, open_side, zone


def short_ep_loc(res_df, config, i, np_timeidx, show_detail=True, ep_loc_side=OrderSide.SELL):   # Todo - temporary
    return ep_loc(res_df, config, i, np_timeidx, show_detail=show_detail, ep_loc_side=OrderSide.SELL)


def long_ep_loc(res_df, config, i, np_timeidx, show_detail=True, ep_loc_side=OrderSide.BUY):   # Todo - temporary
    return ep_loc(res_df, config, i, np_timeidx, show_detail=show_detail, ep_loc_side=OrderSide.BUY)

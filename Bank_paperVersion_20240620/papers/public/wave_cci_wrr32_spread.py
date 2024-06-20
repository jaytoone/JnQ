from funcs.public.indicator import *
from funcs.public.broker import *
from binance_f.model import OrderSide
import logging
from ast import literal_eval

pd.set_option('mode.chained_assignment', None)
sys_log = logging.getLogger()


def lvrg_liqd_set(res_df, config, open_side, ep_, out_, fee, limit_leverage=50):
    selection_id = config.selection_id
    leverage = config.lvrg_set.leverage

    if not pd.isnull(out_) and not config.lvrg_set.static_lvrg:
        # 이 phase 가 정석, 윗 phase 는 결과가 수익 극대화라 사용함
        if open_side == OrderSide.SELL:
            loss = ep_ / out_
        else:
            loss = out_ / ep_

        leverage = config.lvrg_set.target_pct / abs(loss - 1 - (fee + config.trader_set.market_fee))

    # ------------ leverage rejection ------------ #
    # 감당하기 힘든 fluc. 의 경우 진입하지 않음 - dynamic_lvrg 사용 경우
    if leverage < 1 and config.lvrg_set.lvrg_rejection:
        # if config.lvrg_set.leverage >= 1 and config.lvrg_set.lvrg_rejection:
        return None, None

    if not config.lvrg_set.allow_float:
        leverage = int(leverage)

    leverage = min(limit_leverage, max(leverage, 1))

    if open_side == OrderSide.SELL:
        liqd_p = ep_ / (1 + fee + config.trader_set.market_fee - 1 / leverage)
    else:
        liqd_p = ep_ * (1 + fee + config.trader_set.market_fee - 1 / leverage)

    return leverage, liqd_p


def sync_check(df_, config, mode="OPEN", row_slice=True):
    try:
        make_itv_list = [m_itv.replace('m', 'T') for m_itv in literal_eval(config.trader_set.itv_list)]
        row_list = literal_eval(config.trader_set.row_list)
        rec_row_list = literal_eval(config.trader_set.rec_row_list)
        offset_list = literal_eval(config.trader_set.offset_list)

        assert len(make_itv_list) == len(offset_list), "length of itv & offset_list should be equal"
        htf_df_list = [to_htf(df_, itv=itv_, offset=offset_) for itv_idx, (itv_, offset_)
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
            df, df_3T, df_5T, df_15T, df_30T, df_H, df_4H = [df_s.iloc[-row_list[row_idx]:].copy() for row_idx, df_s in
                                                             enumerate(htf_df_list)]
            rec_df, rec_df_3T, rec_df_5T, rec_df_15T, rec_df_30T, rec_df_H, rec_df_4H = [
                df_s.iloc[-rec_row_list[row_idx]:].copy() for row_idx, df_s
                in
                enumerate(htf_df_list)]
        else:
            df, df_3T, df_5T, df_15T, df_30T, df_H, df_4H = htf_df_list
            rec_df, rec_df_3T, rec_df_5T, rec_df_15T, rec_df_30T, rec_df_H, rec_df_4H = htf_df_list

        # --------- add indi. --------- #

        #        1. 필요한 indi. 는 enlist_epouttp & mr_check 보면서 삽입
        #        2. min use_rows 계산을 위해서, tf 별로 gathering 함        #
        # start_0 = time.time()

        # ------ T ------ #
        # df = dc_line(df, None, 'T', dc_period=20)
        # df = bb_line(df, None, 'T')
        #
        # ------ 3T ------ #
        # df = dc_line(df, df_3T, '3T')

        # ------ 5T ------ #
        # h_candle_v3(df, '5T')
        # df = dc_line(df, df_5T, '5T')
        # df = bb_line(df, df_5T, '5T')
        #
        # ------ 15T ------ #
        # h_candle_v3(df, '15T')
        # df = dc_line(df, df_15T, '15T')
        # df = bb_line(df, df_15T, '15T')
        #
        # ------ 30T ------ #
        # df = bb_line(df, df_30T, '30T')
        #
        # ------ H ------ #
        # h_candle_v3(df, 'H')
        # df = dc_line(df, df_H, 'H')

        # ------ 4H ------ #
        # df = bb_line(df, df_4H, '4H')

        # rec_df['rsi_1m'] = rsi(rec_df, 14)  # Todo - recursive, 250 period
        # df = df.join(to_lower_tf_v2(df, rec_df.iloc[:, [-1]], [-1], backing_i=0), how='inner')  # <-- join same_tf manual
        #
        # if order_side in ["OPEN"]:
        #     rec_df_5T['ema_5T'] = ema(rec_df_5T['close'], 195)  # Todo - recursive, 1100 period (5T)
        #     df = df.join(to_lower_tf_v2(df, rec_df_5T, [-1]), how='inner')

    except Exception as e:
        sys_log.error("error in sync_check :", e)
    else:
        return df


def public_indi(res_df, config, np_timeidx, mode="OPEN"):
    selection_id = config.selection_id

    wave_itv1 = config.tr_set.wave_itv1
    wave_itv2 = config.tr_set.wave_itv2
    wave_period1 = config.tr_set.wave_period1
    wave_period2 = config.tr_set.wave_period2
    roll_hl_cnt = 3

    # assert itv_to_number(wave_itv1) > 1  # wave_itv2 == 'T' and
    # ====== public ====== #
    # res_df = wave_range_dcbase_v11_3(res_df, config, over_period=2)

    try:
        # ------------ wave_period1 ------------ #
        if itv_to_number(wave_itv1) > 1:
            offset = '1h' if wave_itv1 != 'D' else '9h'
            htf_df_ = to_htf(res_df, wave_itv1, offset=offset)  # to_htf 는 ohlc, 4개의 col 만 존재 (현재까지)
            htf_df = htf_df_[~pd.isnull(htf_df_.close)]

            htf_df = wave_range_cci_v4_1(htf_df, wave_period1, itv=wave_itv1)

            cols = list(htf_df.columns[4:])  # 15T_ohlc 를 제외한 wave_range_cci_v4 로 추가된 cols, 다 넣어버리기 (추후 혼란 방지)

            valid_co_prime_idx, valid_cu_prime_idx, roll_co_idx_arr, roll_cu_idx_arr = roll_wave_hl_idx_v5(htf_df,
                                                                                                           wave_itv1,
                                                                                                           wave_period1,
                                                                                                           roll_hl_cnt=roll_hl_cnt)

            """ 
            1. wave_bb 의 경우 roll_hl 의 기준이 co <-> cu 변경됨 (cci 와 비교)
            2. wave_bb : high_fill_ -> cu_prime_idx 사용
            """
            htf_df = get_roll_wave_data_v2(htf_df, valid_co_prime_idx, roll_co_idx_arr, 'wave_high_fill_{}{}'.format(wave_itv1, wave_period1), roll_hl_cnt)
            cols += list(htf_df.columns[-roll_hl_cnt:])

            htf_df = get_roll_wave_data_v2(htf_df, valid_cu_prime_idx, roll_cu_idx_arr, 'wave_low_fill_{}{}'.format(wave_itv1, wave_period1), roll_hl_cnt)
            cols += list(htf_df.columns[-roll_hl_cnt:])

            htf_df = wave_range_ratio_v4_2(htf_df, wave_itv1, wave_period1, roll_hl_cnt=roll_hl_cnt)
            cols += list(htf_df.columns[-4:])

            res_df = get_wave_length(res_df, valid_co_prime_idx, valid_cu_prime_idx, roll_co_idx_arr, roll_cu_idx_arr, wave_itv1, wave_period1, roll_hl_cnt=roll_hl_cnt)
            cols += list(htf_df.columns[-4:])

            # ------ 필요한 cols 만 join (htf's idx 정보는 ltf 와 sync. 가 맞지 않음 - join 불가함) ------ #
            res_df.drop(cols, inplace=True, axis=1, errors='ignore')
            res_df = res_df.join(to_lower_tf_v4(res_df, htf_df, cols, backing_i=0, ltf_itv='T').loc[res_df.index], how='inner')

        else:
            res_df = wave_range_cci_v4_1(res_df, wave_period1, itv=wave_itv1)

            valid_co_prime_idx, valid_cu_prime_idx, roll_co_idx_arr, roll_cu_idx_arr = roll_wave_hl_idx_v5(res_df,
                                                                                                           wave_itv1,
                                                                                                           wave_period1,
                                                                                                           roll_hl_cnt=roll_hl_cnt)

            res_df = get_roll_wave_data_v2(res_df, valid_co_prime_idx, roll_co_idx_arr, 'wave_high_fill_{}{}'.format(wave_itv1, wave_period1), roll_hl_cnt)
            res_df = get_roll_wave_data_v2(res_df, valid_cu_prime_idx, roll_cu_idx_arr, 'wave_low_fill_{}{}'.format(wave_itv1, wave_period1), roll_hl_cnt)

            res_df = wave_range_ratio_v4_2(res_df, wave_itv1, wave_period1, roll_hl_cnt=roll_hl_cnt)

            res_df = get_wave_length(res_df, valid_co_prime_idx, valid_cu_prime_idx, roll_co_idx_arr, roll_cu_idx_arr, wave_itv1, wave_period1, roll_hl_cnt=roll_hl_cnt)

        # ------------ wave_period2 ------------ #
        if wave_itv1 != wave_itv2 or wave_period1 != wave_period2:
            if itv_to_number(wave_itv2) > 1:
                offset = '1h' if wave_itv2 != 'D' else '9h'
                htf_df = to_htf(res_df, wave_itv2, offset=offset)
                htf_df = wave_range_cci_v4_1(htf_df, wave_period2, itv=wave_itv2)
                # htf_df = wave_range_bb_v1(htf_df, wave_period2, itv=wave_itv2)

                # cols = list(htf_df.columns[-15:-4])  # except idx col
                cols = list(htf_df.columns[4:])  # 15T_ohlc 를 제외한 wave_range_cci_v4 로 추가된 cols, 다 넣어버리기 (추후 혼란 방지)

                # ------ 필요한 cols 만 join (htf's idx 정보는 ltf 와 sync. 가 맞지 않음 - join 불가함) ------ #
                res_df.drop(cols, inplace=True, axis=1, errors='ignore')
                res_df = res_df.join(to_lower_tf_v4(res_df, htf_df, cols, backing_i=0, ltf_itv='T'), how='inner')  # tf_entry 진입이면, backing_i = 0 가 가능한 것 아닌가.

            else:
                res_df = wave_range_cci_v4_1(res_df, wave_period2, itv=wave_itv2)
                res_df = wave_range_bb_v1(res_df, wave_period2, itv=wave_itv2)

    # valid_co_prime_idx, valid_cu_prime_idx, roll_co_idx_arr, roll_cu_idx_arr = roll_wave_hl_idx_v5(res_df, wave_itv2, wave_period2,
    #                                                                                                roll_hl_cnt=roll_hl_cnt)
    # res_df = get_roll_wave_data_v2(res_df, valid_co_prime_idx, roll_co_idx_arr, 'wave_high_fill_{}{}'.format(wave_itv2, wave_period2),
    #                                roll_hl_cnt)
    # res_df = get_roll_wave_data_v2(res_df, valid_cu_prime_idx, roll_cu_idx_arr, 'wave_low_fill_{}{}'.format(wave_itv2, wave_period2), roll_hl_cnt)
    #
    # res_df = wave_range_ratio_v4_2(res_df, wave_itv2, wave_period2, roll_hl_cnt=roll_hl_cnt)

    # ------ wave_loc_pct (bb) ------ #
    # res_df = wave_loc_pct_v2(res_df, config, 'T', 60)
    # res_df = wave_loc_pct(res_df, config, 'T', 60)

    # future_cols = ['cu_es_15T1', 'co_es_15T1', 'upper_wick_ratio_15T', 'lower_wick_ratio_15T']
    # itv_list = ['15T', '15T', '15T', '15T']
    # res_df = backing_future_data(res_df, future_cols, itv_list)

    # ====== intervaly ====== #
    # ------ 5T ------ #
    # res_df = dc_level(res_df, '5T', 1)
    # res_df = bb_level(res_df, '5T', 1)

    # res_df = st_level(res_df, '5T', 1)

    # ------ 15T ------ #
    # res_df = wick_ratio(res_df, '15T')
    # res_df = dc_level(res_df, '15T', 1)
    # res_df = bb_level(res_df, '15T', 1)
    # res_df = dtk_plot(res_df, dtk_itv2='15T', hhtf_entry=15, use_dtk_line=config.loc_set.zone.use_dtk_line, np_timeidx=np_timeidx)

    # res_df = st_level(res_df, '15T', 1)

    # ------ 30T ------ #
    # res_df = wick_ratio(res_df, '30T')
    # res_df = dc_level(res_df, '30T', 1)
    # res_df = bb_level(res_df, '30T', 1)
    # res_df = st_level(res_df, '30T', 1)

    # ------ H ------ #
    # res_df = wick_ratio(res_df, 'H')
    # res_df = bb_level(res_df, 'H', 1)

    # ------ 4H ------ #
    # res_df = wick_ratio(res_df, '4H')
    # res_df = bb_level(res_df, '4H', 1)

    # res_df['dc_upper_v2'.format(selection_id)] = res_df['high'].rolling(config.loc_set.zone.dc_period).max()   # Todo, consider dc_period
    # res_df['dc_lower_v2'.format(selection_id)] = res_df['low'].rolling(config.loc_set.zone.dc_period).min()

    # res_df['zone_dc_upper_v2'.format(selection_id)] = res_df['high'].rolling(config.loc_set.zone.zone_dc_period).max()   # Todo, consider zone_dc_period
    # res_df['zone_dc_lower_v2'.format(selection_id)] = res_df['low'].rolling(config.loc_set.zone.zone_dc_period).min()

    # if order_side in ["OPEN"]:
    # candle_score_v3(res_df, 'T', unsigned=False)
    # candle_score_v3(res_df, config.loc_set.point1.exp_itv, unsigned=False)

    #     temp indi.    #
    # res_df["ma30_1m"] = res_df['close'].rolling(30).mean()
    # res_df["ma60_1m"] = res_df['close'].rolling(60).mean()
    # res_df = dtk_plot(res_df, dtk_itv2='15T', hhtf_entry=15, use_dtk_line=config.loc_set.zone.use_dtk_line, np_timeidx=np_timeidx)

    except Exception as e:
        sys_log.error("error in public : {}".format(e))
    else:
        return res_df


def expiry_v0(res_df, config, op_idx, e_j, tp_j, np_datas, open_side):
    high, low = np_datas
    selection_id = config.selection_id
    expire = 0

    if config.tr_set.expire_tick != "None":
        if e_j - op_idx >= config.tr_set.expire_tick:
            expire = 1

    if config.tr_set.expire_k1 != "None":
        if open_side == OrderSide.SELL:
            short_tp_1_ = res_df['short_tp_1_{}'.format(selection_id)].to_numpy()  # id 에 따라 dynamic 변수라 이곳에서 numpy 화 진행
            short_tp_gap_ = res_df['short_tp_gap_{}'.format(selection_id)].to_numpy()
            if low[e_j] <= short_tp_1_[tp_j] - short_tp_gap_[tp_j] * config.tr_set.expire_k1:
                expire = 1
        else:
            long_tp_1_ = res_df['long_tp_1_{}'.format(
                selection_id)].to_numpy()  # iloc 이 빠를까, to_numpy() 가 빠를까  # 3.94 ms --> 5.34 ms (iloc)
            long_tp_gap_ = res_df['long_tp_gap_{}'.format(selection_id)].to_numpy()
            if high[e_j] >= long_tp_1_[tp_j] + long_tp_gap_[tp_j] * config.tr_set.expire_k1:
                expire = 1

    return expire


def expiry_tp(res_df, config, op_idx, e_j, tp_j, np_datas, open_side):
    """
    tp, tp_gap 기준 expiry
    """

    high, low = np_datas
    selection_id = config.selection_id
    expire = 0

    if config.tr_set.expire_tick != "None":
        if e_j - op_idx >= config.tr_set.expire_tick:
            expire = 1

    if config.tr_set.expire_k1 != "None":
        if open_side == OrderSide.SELL:
            short_tp_ = res_df['short_tp_{}'.format(selection_id)].to_numpy()  # id 에 따라 dynamic 변수라 이곳에서 numpy 화 진행
            short_tp_gap_ = res_df['short_tp_gap_{}'.format(selection_id)].to_numpy()
            if low[e_j] <= short_tp_[tp_j] + short_tp_gap_[tp_j] * config.tr_set.expire_k1:
                expire = 1
        else:
            long_tp_ = res_df['long_tp_{}'.format(
                selection_id)].to_numpy()  # iloc 이 빠를까, to_numpy() 가 빠를까  # 3.94 ms --> 5.34 ms (iloc)
            long_tp_gap_ = res_df['long_tp_gap_{}'.format(selection_id)].to_numpy()
            if high[e_j] >= long_tp_[tp_j] - long_tp_gap_[tp_j] * config.tr_set.expire_k1:
                expire = 1

    return expire


def expiry_wave(res_df, config, op_idx, e_j, wave1, wave_gap, np_datas, open_side):
    """
    wave_1, wave_gap 기준 expiry
    """

    high, low = np_datas
    selection_id = config.selection_id
    expire = 0

    if config.tr_set.expire_tick != "None":
        if e_j - op_idx >= config.tr_set.expire_tick:
            expire = 1

    if config.tr_set.expire_k2 != "None":
        if open_side == OrderSide.SELL:
            if low[e_j] <= wave1 + wave_gap * config.tr_set.expire_k2:
                expire = 1
        else:
            if high[e_j] >= wave1 - wave_gap * config.tr_set.expire_k2:
                expire = 1

    return expire


def expiry_p1p2(res_df, config, op_idx1, op_idx2, tp1, tp0, tp_gap, np_datas, open_side):
    """
    op_idx1 과 op_idx2 사이의 high / low 통해 expiration survey 라고 보면 됨
    """

    high, low = np_datas
    selection_id = config.selection_id
    expire = 0
    touch_idx = None

    # if config.tr_set.expire_tick != "None":
    #     if e_j - op_idx >= config.tr_set.expire_tick:
    #         expire = 1

    # Todo, p1's tp1, 0 cannot be vectorized
    #   a. expiration 의 조건은 wave1, 0 의 broken
    idx_range = np.arange(op_idx1, op_idx2)
    if config.tr_set.expire_k1 != "None":
        if open_side == OrderSide.SELL:
            touch_idx = np.where((low[op_idx1:op_idx2] <= tp1 + tp_gap * config.tr_set.expire_k1) | \
                                 (high[op_idx1:op_idx2] >= tp0 - tp_gap * config.tr_set.expire_k1),
                                 idx_range, np.nan)
            # if op_idx1 >= 16353:
            #   print("high[16353], tp0 :", high[16353], tp0)
            if np.sum(~np.isnan(touch_idx)) > 0:  # touch 가 존재하면,
                # if low[op_idx1:op_idx2].min() <= tp1 + tp_gap * config.tr_set.expire_k1 or \
                # high[op_idx1:op_idx2].max() >= tp0 - tp_gap * config.tr_set.expire_k1:   # p2_box loc. 이 있어서, op_idx2 + 1 안함
                expire = 1
        else:
            touch_idx = np.where((high[op_idx1:op_idx2] >= tp1 - tp_gap * config.tr_set.expire_k1) | \
                                 (low[op_idx1:op_idx2] <= tp0 + tp_gap * config.tr_set.expire_k1),
                                 idx_range, np.nan)
            if np.sum(~np.isnan(touch_idx)) > 0:
                # if high[op_idx1:op_idx2].max() >= tp1 - tp_gap * config.tr_set.expire_k1 or \
                # low[op_idx1:op_idx2].min() <= tp0 + tp_gap * config.tr_set.expire_k1:
                expire = 1

    return expire, np.nanmin(touch_idx)


def ep_loc_p1_v3(res_df, config, np_timeidx, show_detail=True, ep_loc_side=OrderSide.SELL):

    """
    vectorized calc.
        1. multi-stem 에 따라 dynamic vars.가 입력되기 때문에 class 내부 vars. 로 종속시키지 않음
        2. min & max variables 사용
    """

    # ------- param init ------- #
    selection_id = config.selection_id
    wave_itv1 = config.tr_set.wave_itv1
    wave_period1 = config.tr_set.wave_period1
    c_i = config.trader_set.complete_index

    len_df = len(res_df)
    mr_res = np.ones(len_df)
    zone_arr = np.full(len_df, 'n')

    # ------------ wrr ------------ #
    if config.loc_set.point1.min_wrr_32 != "None":
        if ep_loc_side == OrderSide.SELL:
            cu_wrr_32_ = res_df['cu_wrr_32_{}{}'.format(wave_itv1, wave_period1)].to_numpy()
            mr_res *= cu_wrr_32_ >= config.loc_set.point1.min_wrr_32
            if show_detail:
                sys_log.warning(
                    "cu_wrr_32_ >= config.loc_set.point1.min_wrr_32 : {:.5f} {:.5f} ({})".format(cu_wrr_32_[c_i],
                                                                                                 config.loc_set.point1.min_wrr_32,
                                                                                                 mr_res[c_i]))
        else:
            co_wrr_32_ = res_df['co_wrr_32_{}{}'.format(wave_itv1, wave_period1)].to_numpy()
            mr_res *= co_wrr_32_ >= config.loc_set.point1.min_wrr_32
            if show_detail:
                sys_log.warning(
                    "co_wrr_32_ >= config.loc_set.point1.min_wrr_32 : {:.5f} {:.5f} ({})".format(co_wrr_32_[c_i],
                                                                                                 config.loc_set.point1.min_wrr_32,
                                                                                                 mr_res[c_i]))
    if config.loc_set.point1.max_wrr_32 != "None":
        if ep_loc_side == OrderSide.SELL:
            cu_wrr_32_ = res_df['cu_wrr_32_{}{}'.format(wave_itv1, wave_period1)].to_numpy()
            mr_res *= cu_wrr_32_ <= config.loc_set.point1.max_wrr_32
            if show_detail:
                sys_log.warning(
                    "cu_wrr_32_ <= config.loc_set.point1.max_wrr_32 : {:.5f} {:.5f} ({})".format(cu_wrr_32_[c_i],
                                                                                                 config.loc_set.point1.max_wrr_32,
                                                                                                 mr_res[c_i]))
        else:
            co_wrr_32_ = res_df['co_wrr_32_{}{}'.format(wave_itv1, wave_period1)].to_numpy()
            mr_res *= co_wrr_32_ <= config.loc_set.point1.max_wrr_32
            if show_detail:
                sys_log.warning(
                    "co_wrr_32_ <= config.loc_set.point1.max_wrr_32 : {:.5f} {:.5f} ({})".format(co_wrr_32_[c_i],
                                                                                                 config.loc_set.point1.max_wrr_32,
                                                                                                 mr_res[c_i]))

    # ------------ spread ------------ #
    if config.loc_set.point1.short_min_spread != "None":  # and not config.tr_set.check_hlm:
        if ep_loc_side == OrderSide.SELL:
            short_spread_ = res_df['short_spread_{}'.format(selection_id)].to_numpy()
            mr_res *= short_spread_ >= config.loc_set.point1.short_min_spread
            if show_detail:
                sys_log.warning(
                    "short_spread_ : {:.5f} ({})".format(short_spread_[c_i], mr_res[c_i]))
        else:
            long_spread_ = res_df['long_spread_{}'.format(selection_id)].to_numpy()
            mr_res *= long_spread_ >= config.loc_set.point1.long_min_spread
            if show_detail:
                sys_log.warning(
                    "long_spread_ : {:.5f} ({})".format(long_spread_[c_i], mr_res[c_i]))

    if config.loc_set.point1.short_max_spread != "None":  # and not config.tr_set.check_hlm:
        if ep_loc_side == OrderSide.SELL:
            short_spread_ = res_df['short_spread_{}'.format(selection_id)].to_numpy()
            mr_res *= short_spread_ <= config.loc_set.point1.short_max_spread
            if show_detail:
                sys_log.warning(
                    "short_spread_ : {:.5f} ({})".format(short_spread_[c_i], mr_res[c_i]))
        else:
            long_spread_ = res_df['long_spread_{}'.format(selection_id)].to_numpy()
            mr_res *= long_spread_ <= config.loc_set.point1.long_max_spread
            if show_detail:
                sys_log.warning(
                    "long_spread_ : {:.5f} ({})".format(long_spread_[c_i], mr_res[c_i]))

    # ------------ tr_thresh ------------ #  # vectorize allow only for p1_hhm
    # if config.loc_set.point1.short_tr_thresh != "None":  #  and not config.tr_set.check_hlm:
    #     if ep_loc_side == OrderSide.SELL:
    #         short_tr_ = res_df['short_tr_{}'.format(selection_id)].to_numpy()
    #         mr_res *= short_tr_ >= config.loc_set.point1.short_tr_thresh
    #         # mr_res *= short_tr_ <= config.loc_set.point1.short_tr_thresh + 0.1
    #         if show_detail:
    #             sys_log.warning(
    #                 "short_tr_ >= short_tr_thresh : {:.5f} {:.5f} ({})".format(short_tr_[c_i], config.loc_set.point1.short_tr_thresh, mr_res[c_i]))
    #     else:
    #         long_tr_ = res_df['long_tr_{}'.format(selection_id)].to_numpy()
    #         mr_res *= long_tr_ >= config.loc_set.point1.long_tr_thresh
    #         # mr_res *= long_tr_ <= config.loc_set.point1.long_tr_thresh + 0.1
    #         if show_detail:
    #             sys_log.warning(
    #                 "long_tr_ >= long_tr_thresh : {:.5f} {:.5f} ({})".format(long_tr_[c_i], config.loc_set.point1.long_tr_thresh, mr_res[c_i]))

    # ------------ zone ------------ #
    # ------ config var. 이 등록되지 않은 dur. 은 selection_id 으로 조건문을 나눔 (lvrg_set 과 동일) ------ #
    if config.loc_set.zone1.use_zone:
        # ------------ on_price ------------ #
        # ------ 추세선 리스트 ------ #
        # wave_itv1 = config.tr_set.wave_itv1
        # wave_period1 = config.tr_set.wave_period1

        # wave_high_fill1_ = res_df['wave_high_fill_{}{}'.format(wave_itv1, wave_period1)].to_numpy()
        # wave_low_fill1_ = res_df['wave_low_fill_{}{}'.format(wave_itv1, wave_period1)].to_numpy()

        # short_tp_0_ = res_df['short_tp_0_{}'.format(selection_id)].to_numpy()
        # long_tp_0_ = res_df['long_tp_0_{}'.format(selection_id)].to_numpy()

        # ------ sar ------ #
        #         high_ = res_df['high_15T'].to_numpy()
        #         low_ = res_df['low_15T'].to_numpy()
        #         sar_ = res_df['sar_5T'].to_numpy()

        #         if ep_loc_side == OrderSide.SELL:
        #             mr_res *= high_ < sar_
        #             if show_detail:
        #                 sys_log.warning("high_ < sar_ : {:.5f} {:.5f} ({})".format(high_[c_i], sar_[c_i], mr_res[c_i]))
        #         else:
        #             mr_res *= low_ > sar_
        #             if show_detail:
        #                 sys_log.warning("low_ > sar_ : {:.5f} {:.5f} ({})".format(low_[c_i], sar_[c_i], mr_res[c_i]))

        # ------ dc_base ------ #
        # dc_base_ = res_df['dc_base_T30'].to_numpy()
        # dc_base_T20 = res_df['dc_base_T20'].to_numpy()
        # dc_base_5T20 = res_df['dc_base_5T20'].to_numpy()
        # dc_base_15T20 = res_df['dc_base_15T20'].to_numpy()
        # dc_base_H20 = res_df['dc_base_H20'].to_numpy()
        # dc_base_4H20 = res_df['dc_base_4H20'].to_numpy()

        # if ep_loc_side == OrderSide.SELL:
        #     mr_res *= short_tp_0_ < dc_base_H20
        #     if show_detail:
        #         sys_log.warning("short_tp_0_ < dc_base_H20 : {:.5f} {:.5f} ({})".format(short_tp_0_[c_i], dc_base_H20[c_i], mr_res[c_i]))
        # else:
        #     mr_res *= long_tp_0_ > dc_base_H20
        #     if show_detail:
        #         sys_log.warning("long_tp_0_ > dc_base_H20 : {:.5f} {:.5f} ({})".format(long_tp_0_[c_i], dc_base_H20[c_i], mr_res[c_i]))

        """
        future data phase
        """

    #         close_ = res_df['close_30T'].to_numpy()
    #         f1_close_ = res_df['close_30T'].shift(-30).to_numpy()

    #         if ep_loc_side == OrderSide.SELL:
    #             mr_res *= f1_close_ < close_
    #             if show_detail:
    #                 sys_log.warning("f1_close_ < close_ : {:.5f} {:.5f} ({})".format(f1_close_[c_i], close_[c_i], mr_res[c_i]))
    #         else:
    #             mr_res *= f1_close_ > close_
    #             if show_detail:
    #                 sys_log.warning("f1_close_ < close_ : {:.5f} {:.5f} ({})".format(f1_close_[c_i], close_[c_i], mr_res[c_i]))

    #         cci_ = res_df['cci_30T20'].to_numpy()
    #         f1_cci_ = res_df['cci_30T20'].shift(-30).to_numpy()

    #         if ep_loc_side == OrderSide.SELL:
    #             mr_res *= f1_cci_ < cci_
    #             if show_detail:
    #                 sys_log.warning("f1_cci_ < cci_ : {:.5f} {:.5f} ({})".format(f1_cci_[c_i], cci_[c_i], mr_res[c_i]))
    #         else:
    #             mr_res *= f1_cci_ > cci_
    #             if show_detail:
    #                 sys_log.warning("f1_cci_ < cci_ : {:.5f} {:.5f} ({})".format(f1_cci_[c_i], cci_[c_i], mr_res[c_i]))

    # ------------ outer_price ------------ #
    # ------ cci ------ #
    #         cci_ = res_df['cci_30T20'].to_numpy()
    #         b1_cci_ = res_df['cci_30T20'].shift(30).to_numpy()
    #         # base_value = -100

    #         if ep_loc_side == OrderSide.SELL:
    #             # mr_res *= cci_ < 0
    #             mr_res *= cci_ < b1_cci_
    #             # mr_res *= (cci_ > -100) & (cci_ < -80)
    #             if show_detail:
    #                 sys_log.warning("cci_ < b1_cci_ : {:.5f} {:.5f} ({})".format(cci_[c_i], b1_cci_[c_i], mr_res[c_i]))
    #         else:
    #             # mr_res *= cci_ < 0
    #             mr_res *= cci_ > b1_cci_
    #             # mr_res *= (cci_ > 80) & (cci_ < 100)
    #             if show_detail:
    #                 sys_log.warning("cci_ > b1_cci_ : {:.5f} {:.5f} ({})".format(cci_[c_i], b1_cci_[c_i], mr_res[c_i]))

    # ------ macd ------ #
    # # macd_ = res_df['macd_T535'].to_numpy()
    # macd_ = res_df['macd_hist_T53515'].to_numpy()

    # if ep_loc_side == OrderSide.SELL:
    #   mr_res *= macd_ < 0
    #   if show_detail:
    #     sys_log.warning("macd_ < 0 : {:.5f} {:.5f} ({})".format(macd_[c_i], 0, mr_res[c_i]))
    # else:
    #   mr_res *= macd_ > 0
    #   if show_detail:
    #     sys_log.warning("macd_ > 0 : {:.5f} {:.5f} ({})".format(macd_[c_i], 0, mr_res[c_i]))

    return mr_res, zone_arr  # mr_res 의 True idx 가 open signal


def ep_loc_p2_v3(res_df, config, np_timeidx, show_detail=True, ep_loc_side=OrderSide.SELL):

    # ------- param init ------- #
    selection_id = config.selection_id
    wave_itv2 = config.tr_set.wave_itv2
    wave_period2 = config.tr_set.wave_period2
    c_i = config.trader_set.complete_index

    len_df = len(res_df)
    mr_res = np.ones(len_df)
    zone_arr = np.full(len_df, 'n')

    # ------------ wave_range_ratio ------------ #
    if config.loc_set.point2.min_wrr_32 != "None":
        if ep_loc_side == OrderSide.SELL:
            cu_wrr_32_ = res_df['cu_wrr_32_{}{}'.format(wave_itv2, wave_period2)].to_numpy()
            mr_res *= cu_wrr_32_ >= config.loc_set.point2.min_wrr_32
            if show_detail:
                sys_log.warning(
                    "cu_wrr_32_ >= config.loc_set.point2.min_wrr_32 : {:.5f} {:.5f} ({})".format(cu_wrr_32_[c_i],
                                                                                                 config.loc_set.point2.min_wrr_32,
                                                                                                 mr_res[c_i]))
        else:
            co_wrr_32_ = res_df['co_wrr_32_{}{}'.format(wave_itv2, wave_period2)].to_numpy()
            mr_res *= co_wrr_32_ >= config.loc_set.point2.min_wrr_32
            if show_detail:
                sys_log.warning(
                    "co_wrr_32_ >= config.loc_set.point2.min_wrr_32 : {:.5f} {:.5f} ({})".format(co_wrr_32_[c_i],
                                                                                                 config.loc_set.point2.min_wrr_32,
                                                                                                 mr_res[c_i]))
    if config.loc_set.point2.max_wrr_32 != "None":
        if ep_loc_side == OrderSide.SELL:
            cu_wrr_32_ = res_df['cu_wrr_32_{}{}'.format(wave_itv2, wave_period2)].to_numpy()
            mr_res *= cu_wrr_32_ <= config.loc_set.point2.max_wrr_32
            if show_detail:
                sys_log.warning(
                    "cu_wrr_32_ <= config.loc_set.point2.max_wrr_32 : {:.5f} {:.5f} ({})".format(cu_wrr_32_[c_i],
                                                                                                 config.loc_set.point2.max_wrr_32,
                                                                                                 mr_res[c_i]))
        else:
            co_wrr_32_ = res_df['co_wrr_32_{}{}'.format(wave_itv2, wave_period2)].to_numpy()
            mr_res *= co_wrr_32_ <= config.loc_set.point2.max_wrr_32
            if show_detail:
                sys_log.warning(
                    "co_wrr_32_ <= config.loc_set.point2.max_wrr_32 : {:.5f} {:.5f} ({})".format(co_wrr_32_[c_i],
                                                                                                 config.loc_set.point2.max_wrr_32,
                                                                                                 mr_res[c_i]))

    if config.loc_set.zone2.use_zone:
        # ------------ outer_price ------------ #
        # ------ cci ------ #
        cci_ = res_df['cci_30T20'].to_numpy()
        b1_cci_ = res_df['cci_30T20'].shift(30).to_numpy()
        # base_value = -100

        if ep_loc_side == OrderSide.SELL:
            # mr_res *= cci_ < 0
            mr_res *= cci_ < b1_cci_
            # mr_res *= (cci_ > -100) & (cci_ < -80)
            if show_detail:
                sys_log.warning("cci_ < b1_cci_ : {:.5f} {:.5f} ({})".format(cci_[c_i], b1_cci_[c_i], mr_res[c_i]))
        else:
            # mr_res *= cci_ < 0
            mr_res *= cci_ > b1_cci_
            # mr_res *= (cci_ > 80) & (cci_ < 100)
            if show_detail:
                sys_log.warning("cci_ > b1_cci_ : {:.5f} {:.5f} ({})".format(cci_[c_i], b1_cci_[c_i], mr_res[c_i]))

    return mr_res, zone_arr  # mr_res 의 True idx 가 open signal
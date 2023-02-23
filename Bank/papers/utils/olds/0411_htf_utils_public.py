from funcs.public.indicator import *
from funcs.public.broker import *
import logging
from ast import literal_eval

pd.set_option('mode.chained_assignment',  None)
sys_log = logging.getLogger()


class OrderSide:
    BUY = "BUY"
    SELL = "SELL"
    INVALID = None


def lvrg_set(res_df, config, open_side, ep_, out_, fee, limit_leverage=50):
    strat_version = config.strat_version
    if not pd.isnull(out_) and not config.lvrg_set.static_lvrg:
        if strat_version in ['v3']:
            if open_side == OrderSide.SELL:
                loss = out_ / ep_
            else:
                loss = ep_ / out_
        else:  # 이 phase 가 정석, 윗 phase 는 결과가 수익 극대화라 사용함
            if open_side == OrderSide.SELL:
                loss = ep_ / out_
            else:
                loss = out_ / ep_

        config.lvrg_set.leverage = config.lvrg_set.target_pct / abs(loss - 1 - (fee + config.trader_set.market_fee))

    # ------------ leverage rejection ------------ #
    # 감당하기 힘든 fluc. 의 경우 진입하지 않음 - dynamic_lvrg 사용 경우
    if config.lvrg_set.leverage < 1 and config.lvrg_set.lvrg_rejection:
        # if config.lvrg_set.leverage >= 1 and config.lvrg_set.lvrg_rejection:
        return None

    if not config.lvrg_set.allow_float:
        config.lvrg_set.leverage = int(config.lvrg_set.leverage)

    config.lvrg_set.leverage = min(limit_leverage, max(config.lvrg_set.leverage, 1))

    return config.lvrg_set.leverage


def sync_check(df_, config, order_side="OPEN", row_slice=True):
    try:
        make_itv_list = [m_itv.replace('m', 'T') for m_itv in literal_eval(config.trader_set.itv_list)]
        row_list = literal_eval(config.trader_set.row_list)
        rec_row_list = literal_eval(config.trader_set.rec_row_list)
        offset_list = literal_eval(config.trader_set.offset_list)

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
            df, df_3T, df_5T, df_15T, df_30T, df_H, df_4H = [df_s.iloc[-row_list[row_idx]:].copy() for row_idx, df_s in enumerate(htf_df_list)]
            rec_df, rec_df_3T, rec_df_5T, rec_df_15T, rec_df_30T, rec_df_H, rec_df_4H = [df_s.iloc[-rec_row_list[row_idx]:].copy() for row_idx, df_s
                                                                                         in enumerate(htf_df_list)]
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
        df = dc_line(df, df_5T, '5T')
        df = bb_line(df, df_5T, '5T')
        #
        # ------ 15T ------ #
        df = dc_line(df, df_15T, '15T')
        df = bb_line(df, df_15T, '15T')
        #
        # ------ 30T ------ #
        # df = bb_line(df, df_30T, '30T')
        #
        # ------ H ------ #
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


def public_indi(res_df, config, np_timeidx, order_side="OPEN"):
    strat_version = config.strat_version

    # res_df = wave_range_v7(res_df, config.loc_set.point.p1_period1, config.loc_set.point.p1_period2, ltf_df=None, touch_period=50)
    # if config.loc_set.point.p2_itv1 != "None":
    #     res_df = wave_range_v7(res_df, config.loc_set.point.p2_period1, config.loc_set.point.p2_period2, ltf_df=None, touch_period=50)

    res_df = dc_level(res_df, '5T', 1)
    res_df = bb_level(res_df, '5T', 1)
    # res_df = st_level(res_df, '5T', 1)

    res_df = dc_level(res_df, '15T', 1)
    res_df = bb_level(res_df, '15T', 1)
    res_df = dtk_plot(res_df, dtk_itv2='15T', hhtf_entry=15, use_dtk_line=config.loc_set.zone.use_dtk_line, np_timeidx=np_timeidx)

    # res_df = st_level(res_df, '15T', 1)

    # res_df = dc_level(res_df, '30T', 1)
    # res_df = bb_level(res_df, '30T', 1)
    # res_df = st_level(res_df, '30T', 1)

    # res_df = bb_level(res_df, 'H', 1)

    # res_df = bb_level(res_df, '4H', 1)

    res_df['dc_upper_v2'.format(strat_version)] = res_df['high'].rolling(config.loc_set.zone.dc_period).max()   # Todo, consider dc_period
    res_df['dc_lower_v2'.format(strat_version)] = res_df['low'].rolling(config.loc_set.zone.dc_period).min()

    res_df['zone_dc_upper_v2'.format(strat_version)] = res_df['high'].rolling(config.loc_set.zone.zone_dc_period).max()   # Todo, consider zone_dc_period
    res_df['zone_dc_lower_v2'.format(strat_version)] = res_df['low'].rolling(config.loc_set.zone.zone_dc_period).min()

    if order_side in ["OPEN"]:
        h_candle_v3(res_df, '5T')
        h_candle_v3(res_df, '15T')
        h_candle_v3(res_df, 'H')

        # candle_score_v2(res_df, 'T', unsigned=False)

        # hc_itv = 'H'
        # h_candle_col = ['open_{}'.format(hc_itv), 'high_{}'.format(hc_itv), 'low_{}'.format(hc_itv), 'close_{}'.format(hc_itv)]
        # candle_score_v2(res_df, hc_itv, ohlc_col=h_candle_col, unsigned=False)

    #     temp indi.    #
    # res_df["ma30_1m"] = res_df['close'].rolling(30).mean()
    # res_df["ma60_1m"] = res_df['close'].rolling(60).mean()
    # res_df = dtk_plot(res_df, dtk_itv2='15T', hhtf_entry=15, use_dtk_line=config.loc_set.zone.use_dtk_line, np_timeidx=np_timeidx)

    return res_df


def ep_out_v0(res_df, config, op_idx, e_j, tp_j, np_datas, open_side):
    h, l = np_datas
    strat_version = config.strat_version
    ep_out = 0

    if config.loc_set.zone.ep_out_tick != "None":
        if e_j - op_idx >= config.loc_set.zone.ep_out_tick:
            ep_out = 1

    if config.loc_set.zone.ei_k != "None":
        if open_side == OrderSide.SELL:
            short_tp_1_ = res_df['short_tp_1_{}'.format(strat_version)].to_numpy()  # id 에 따라 dynamic 변수라 이곳에서 numpy 화 진행
            short_tp_gap_ = res_df['short_tp_gap_{}'.format(strat_version)].to_numpy()
            if l[e_j] <= short_tp_1_[tp_j] - short_tp_gap_[tp_j] * config.loc_set.zone.ei_k:
                ep_out = 1
        else:
            long_tp_1_ = res_df['long_tp_1_{}'.format(strat_version)].to_numpy()  # iloc 이 빠를까, to_numpy() 가 빠를까  # 3.94 ms --> 5.34 ms (iloc)
            long_tp_gap_ = res_df['long_tp_gap_{}'.format(strat_version)].to_numpy()
            if h[e_j] >= long_tp_1_[tp_j] + long_tp_gap_[tp_j] * config.loc_set.zone.ei_k:
                ep_out = 1

    return ep_out


def ep_out(res_df, config, op_idx, e_j, tp_j, np_datas, open_side):
    h, l = np_datas
    strat_version = config.strat_version
    ep_out = 0

    if config.loc_set.zone.ep_out_tick != "None":
        if e_j - op_idx >= config.loc_set.zone.ep_out_tick:
            ep_out = 1

    if config.loc_set.zone.ei_k != "None":
        if open_side == OrderSide.SELL:
            short_tp_ = res_df['short_tp_{}'.format(strat_version)].to_numpy()  # id 에 따라 dynamic 변수라 이곳에서 numpy 화 진행
            short_tp_gap_ = res_df['short_tp_gap_{}'.format(strat_version)].to_numpy()
            if l[e_j] <= short_tp_[tp_j] + short_tp_gap_[tp_j] * config.loc_set.zone.ei_k:
                ep_out = 1
        else:
            long_tp_ = res_df['long_tp_{}'.format(strat_version)].to_numpy()  # iloc 이 빠를까, to_numpy() 가 빠를까  # 3.94 ms --> 5.34 ms (iloc)
            long_tp_gap_ = res_df['long_tp_gap_{}'.format(strat_version)].to_numpy()
            if h[e_j] >= long_tp_[tp_j] - long_tp_gap_[tp_j] * config.loc_set.zone.ei_k:
                ep_out = 1

    return ep_out


def ep_out_v2(res_df, config, op_idx, e_j, tp_j, np_datas, open_side):
    h, l = np_datas
    strat_version = config.strat_version
    ep_out = 0

    if config.loc_set.zone.ep_out_tick != "None":
        if e_j - op_idx >= config.loc_set.zone.ep_out_tick:
            ep_out = 1

    if config.loc_set.zone.ei_k != "None":
        if open_side == OrderSide.SELL:
            short_tp_1_ = res_df['short_tp_1_{}'.format(strat_version)].to_numpy()  # id 에 따라 dynamic 변수라 이곳에서 numpy 화 진행
            short_tp_0_ = res_df['short_tp_0_{}'.format(strat_version)].to_numpy()
            short_tp_gap_ = res_df['short_tp_gap_{}'.format(strat_version)].to_numpy()
            if l[e_j] <= short_tp_1_[tp_j] + short_tp_gap_[tp_j] * config.loc_set.zone.ei_k or h[e_j] >= short_tp_0_[tp_j] - short_tp_gap_[
                tp_j] * config.loc_set.zone.ei_k:
                ep_out = 1
        else:
            long_tp_1_ = res_df['long_tp_1_{}'.format(strat_version)].to_numpy()
            long_tp_0_ = res_df['long_tp_0_{}'.format(strat_version)].to_numpy()
            long_tp_gap_ = res_df['long_tp_gap_{}'.format(strat_version)].to_numpy()
            if h[e_j] >= long_tp_1_[tp_j] - long_tp_gap_[tp_j] * config.loc_set.zone.ei_k or l[e_j] <= long_tp_0_[tp_j] + long_tp_gap_[
                tp_j] * config.loc_set.zone.ei_k:
                ep_out = 1

    return ep_out


def ep_loc_point2_v2(res_df, config, i, out_j, side=OrderSide.SELL):
    allow_ep_in = 1
    if config.strat_version in ['v5_2']:
        if side == OrderSide.SELL:
            dc_upper_T = res_df['dc_upper_T'].to_numpy()
            dc_upper_15T = res_df['dc_upper_15T'].to_numpy()
            allow_ep_in *= (dc_upper_T[i - 1] <= dc_upper_15T[i]) & \
                           (dc_upper_15T[i - 1] != dc_upper_15T[i])
        else:
            dc_lower_T = res_df['dc_lower_T'].to_numpy()
            dc_lower_15T = res_df['dc_lower_15T'].to_numpy()
            allow_ep_in *= (dc_lower_T[i - 1] >= dc_lower_15T[i]) & \
                           (dc_lower_15T[i - 1] != dc_lower_15T[i])

    if config.strat_version in ['v3_4']:
        wick_score_list = literal_eval(config.ep_set.point2.wick_score_list)
        wick_scores = [res_df['wick_score_{}'.format(s_itv)].to_numpy() for s_itv in literal_eval(config.ep_set.point2.score_itv_list)]
        close = res_df['close'].to_numpy()
        if side == OrderSide.SELL:
            sup_T = res_df['sup_T'].to_numpy()
            allow_ep_in *= close[i] < sup_T[i - 1]
            if len(wick_score_list) != 0:
                allow_ep_in *= wick_scores[0][i] < -wick_score_list[0]
        else:
            resi_T = res_df['resi_T'].to_numpy()
            allow_ep_in *= close[i] > resi_T[i - 1]
            if len(wick_score_list) != 0:
                allow_ep_in *= wick_scores[0][i] > wick_score_list[0]

    if allow_ep_in:
        out_j = i
    return allow_ep_in, out_j


# vectorized calc.
def ep_loc_v3(res_df, config, np_timeidx, show_detail=True, ep_loc_side=OrderSide.SELL):
    # ------- param init ------- #
    strat_version = config.strat_version
    c_i = config.trader_set.complete_index
    len_df = len(res_df)
    mr_res = np.ones(len_df)
    zone_arr = np.full(len_df, 'n')

    # ------ process 한번에 처리해서 param_check 만 ver. 별로 하면 될 것 ------ #
    #     => public_indi() 가 될 것
    #     1. 사용한 param 정보와 matching 된 data 병렬로 나열 logging 될 것
    tp_fee, out_fee = calc_tp_out_fee_v2(config)

    # ------------ candle_score ------------ #
    wick_score_list = literal_eval(config.loc_set.point.wick_score_list)
    if len(wick_score_list) != 0:
        score_itv_list = literal_eval(config.loc_set.point.score_itv_list)
        # ------ candle_score_v0 (1m initial tick 기준임) ------ #  Todo - higher timeframe 경우 back_data 사용해야함
        for wick_score_, score_itv_ in zip(wick_score_list, score_itv_list):
            wick_score = res_df['wick_score_{}'.format(score_itv_)].to_numpy()
            if ep_loc_side == OrderSide.SELL:
                mr_res *= wick_score <= -wick_score_
                if show_detail:
                    sys_log.warning("wick_score <= -wick_score_ : {:.5f} {:.5f} ({})".format(wick_score[c_i], -wick_score_, mr_res[c_i]))
            else:
                mr_res *= wick_score >= wick_score_
                if show_detail:
                    sys_log.warning("wick_score >= wick_score_ : {:.5f} {:.5f} ({})".format(wick_score[c_i], wick_score_, mr_res[c_i]))

    # -------------- tr_thresh -------------- #
    if config.loc_set.zone.short_tr_thresh != "None":
        if ep_loc_side == OrderSide.SELL:
            short_tr_ = res_df['short_tr_{}'.format(strat_version)].to_numpy()
            # mr_res *= short_tr_ >= config.loc_set.zone.short_tr_thresh
            mr_res *= short_tr_ <= config.loc_set.zone.short_tr_thresh
            if show_detail:
                sys_log.warning(
                    "short_tr_ <= short_tr_thresh : {:.5f} {:.5f} ({})".format(short_tr_[c_i], config.loc_set.zone.short_tr_thresh, mr_res[c_i]))
        else:
            long_tr_ = res_df['long_tr_{}'.format(strat_version)].to_numpy()
            # mr_res *= long_tr_ >= config.loc_set.zone.long_tr_thresh
            mr_res *= long_tr_ <= config.loc_set.zone.long_tr_thresh
            if show_detail:
                sys_log.warning(
                    "long_tr_ <= long_tr_thresh : {:.5f} {:.5f} ({})".format(long_tr_[c_i], config.loc_set.zone.long_tr_thresh, mr_res[c_i]))

    # -------------- spread - independent to tr_set -------------- #
    if config.loc_set.zone.short_spread != "None":
        if strat_version in ['v3']:
            if ep_loc_side == OrderSide.SELL:
                bb_base_5T = res_df['bb_base_5T'].to_numpy()  # to_numpy() 는 ep_loc 에서 진행됨
                bb_lower_5T = res_df['bb_lower_5T'].to_numpy()
                short_spread = (bb_base_5T - bb_lower_5T - tp_fee * bb_base_5T) / (bb_base_5T - bb_lower_5T + out_fee * bb_base_5T)
                mr_res *= short_spread >= config.loc_set.zone.short_spread
                if show_detail:
                    sys_log.warning(
                        "short_spread >= config.loc_set.zone.short_spread : {:.5f} {:.5f} ({})".format(short_spread[c_i], config.loc_set.zone.short_spread, mr_res[c_i]))
            else:
                bb_upper_5T = res_df['bb_upper_5T'].to_numpy()
                dc_lower_5T = res_df['dc_lower_5T'].to_numpy()
                # mr_res *= (bb_base_5T - dc_lower_5T - tp_fee * bb_upper_5T) / (bb_base_5T - dc_lower_5T + out_fee * bb_base_5T) >= config.loc_set.zone.long_spread
                long_spread = (bb_upper_5T - dc_lower_5T - tp_fee * bb_upper_5T) / (bb_upper_5T - dc_lower_5T + out_fee * bb_upper_5T)
                mr_res *= long_spread >= config.loc_set.zone.long_spread
                if show_detail:
                    sys_log.warning(
                        "long_spread >= config.loc_set.zone.long_spread : {:.5f} {:.5f} ({})".format(long_spread[c_i], config.loc_set.zone.long_spread, mr_res[c_i]))
    # ------------ rtc_zone  ------------ #
    # ------ dtk ------ #
    if config.loc_set.zone.dt_k != "None":
        # ------ dc_v2 ------ #
        dc_lower_v2 = res_df['dc_lower_v2'.format(strat_version)].to_numpy()
        short_dtk_1_ = res_df['short_dtk_1_{}'.format(strat_version)].to_numpy() - \
                       res_df['short_dtk_gap_{}'.format(strat_version)].to_numpy() * config.loc_set.zone.dt_k
        dc_upper_v2 = res_df['dc_upper_v2'.format(strat_version)].to_numpy()
        long_dtk_1_ = res_df['long_dtk_1_{}'.format(strat_version)].to_numpy() + \
                      res_df['long_dtk_gap_{}'.format(strat_version)].to_numpy() * config.loc_set.zone.dt_k
        if ep_loc_side == OrderSide.SELL:
            mr_res *= dc_lower_v2 >= short_dtk_1_
            if show_detail:
                sys_log.warning(
                    "dc_lower_v2 >= short_dtk_1_ : {:.5f} {:.5f} ({})".format(dc_lower_v2[c_i], short_dtk_1_[c_i], mr_res[c_i]))
        else:
            mr_res *= dc_upper_v2 <= long_dtk_1_
            if show_detail:
                sys_log.warning(
                    "dc_upper_v2 <= long_dtk_1_ : {:.5f} {:.5f} ({})".format(dc_upper_v2[c_i], long_dtk_1_[c_i], mr_res[c_i]))

    # ------------ zone rejection  ------------ #
    # config 로 통제할 수 없는 rejection 은 strat_version 으로 조건문을 나눔 (lvrg_set 과 동일)
    if config.loc_set.zone.zone_rejection:

        # ------------------ biaser, sr_confirmer ------------------ #
        # ------ 1. mtf_baseline ------ #
        if strat_version in ['4_3', '3_5', '3_51']:
            dc_base_3T = res_df['dc_base_3T'].to_numpy()
            b1_dc_base_3T = res_df['dc_base_3T'].shift(3).to_numpy()
            dc_base_5T = res_df['dc_base_5T'].to_numpy()
            dc_base_15T = res_df['dc_base_15T'].to_numpy()
            dc_base_30T = res_df['dc_base_30T'].to_numpy()
            dc_base_H = res_df['dc_base_H'].to_numpy()
            dc_base_4H = res_df['dc_base_4H'].to_numpy()
            dc_base_D = res_df['dc_base_D'].to_numpy()
            # wave_base_ = res_df['wave_base_{}'.format(config.loc_set.point.tp_itv0)].to_numpy()

            itv, period1, period2 = config.loc_set.point.p1_itv1, config.loc_set.point.p1_period1, config.loc_set.point.p1_period2
            if ep_loc_side == OrderSide.SELL:
                # ------ short_base_ <= dc_base_3T ------ #
                short_base_ = res_df['short_base_{}{}{}'.format(itv, period1, period2)].to_numpy()
                mr_res *= short_base_ <= dc_base_3T
                if show_detail:
                    sys_log.warning("short_base_ <= dc_base_3T : {:.5f} {:.5f} ({})".format(short_base_[c_i], dc_base_3T[c_i], mr_res[c_i]))

                # ------ reject csd ------ #
                # dc_upper_ = res_df['dc_upper_{}{}'.format(itv, period1)].to_numpy()
                # mr_res *= dc_upper_ <= dc_base_3T
                # if show_detail:
                #     sys_log.warning("dc_upper_ <= dc_base_3T : {:.5f} {:.5f} ({})".format(dc_upper_[c_i], dc_base_3T[c_i], mr_res[c_i]))

                # Todo, 부호 조심
                # dc_upper2_ = res_df['dc_upper_{}{}'.format(itv, period2)].to_numpy()
                # mr_res *= dc_upper2_ >= dc_base_H
                # if show_detail:
                #     sys_log.warning("dc_upper2_ >= dc_base_H : {:.5f} {:.5f} ({})".format(dc_upper2_[c_i], dc_base_H[c_i], mr_res[c_i]))

                # dc_lower2_ = res_df['dc_lower_{}{}'.format(itv, period2)].to_numpy()
                # mr_res *= dc_lower2_ >= dc_base_H
                # if show_detail:
                #     sys_log.warning("dc_lower2_ >= dc_base_H : {:.5f} {:.5f} ({})".format(dc_lower2_[c_i], dc_base_H[c_i], mr_res[c_i]))

                    # ------ consecutive base ascender ------ #
                # ------ 1. roll_min ------ #
                dc_base_3T_rollmin = res_df['dc_base_3T'].rolling(config.loc_set.zone.base_roll_period).min().to_numpy()
                mr_res *= dc_base_3T_rollmin == dc_base_3T
                if show_detail:
                    sys_log.warning(
                        "dc_base_3T_rollmin == dc_base_3T : {:.5f} {:.5f} ({})".format(dc_base_3T_rollmin[c_i], dc_base_3T[c_i], mr_res[c_i]))
            else:
                # ------ long_base >= dc_base_3T ------ #
                long_base_ = res_df['long_base_{}{}{}'.format(itv, period1, period2)].to_numpy()
                mr_res *= long_base_ >= dc_base_3T
                if show_detail:
                    sys_log.warning("long_base_ >= dc_base_3T : {:.5f} {:.5f} ({})".format(long_base_[c_i], dc_base_3T[c_i], mr_res[c_i]))

                    # ------ reject csd ------ #
                # dc_lower_ = res_df['dc_lower_{}{}'.format(itv, period1)].to_numpy()
                # mr_res *= dc_lower_ >= dc_base_3T
                # if show_detail:
                #     sys_log.warning("dc_lower_ >= dc_base_3T : {:.5f} {:.5f} ({})".format(dc_lower_[c_i], dc_base_3T[c_i], mr_res[c_i]))

                dc_lower2_ = res_df['dc_lower_{}{}'.format(itv, period2)].to_numpy()
                mr_res *= dc_lower2_ >= dc_base_H
                if show_detail:
                    sys_log.warning("dc_lower2_ >= dc_base_H : {:.5f} {:.5f} ({})".format(dc_lower2_[c_i], dc_base_H[c_i], mr_res[c_i]))

                    # ------ alignment ------ #
                # mr_res *= (dc_base_3T > dc_base_5T) & (dc_base_5T > dc_base_15T) & (dc_base_15T > dc_base_30T)

                # ------ consecutive base ascender ------ #
                # ------ 1. roll_max ------ #
                dc_base_3T_rollmax = res_df['dc_base_3T'].rolling(config.loc_set.zone.base_roll_period).max().to_numpy()
                mr_res *= dc_base_3T_rollmax == dc_base_3T
                if show_detail:
                    sys_log.warning(
                        "dc_base_3T_rollmax == dc_base_3T : {:.5f} {:.5f} ({})".format(dc_base_3T_rollmax[c_i], dc_base_3T[c_i], mr_res[c_i]))

                # ------ 2. roll_max_v2 - ascender  ------ #
                # dc_base_3T_ascend = (res_df['dc_base_3T'] >= res_df['dc_base_3T'].shift(3)).rolling(config.loc_set.zone.base_roll_period).sum().to_numpy()
                # # mr_res *= dc_base_3T_ascend == config.loc_set.zone.base_roll_period
                # mr_res *= dc_base_3T_ascend != config.loc_set.zone.base_roll_period
                # if show_detail:
                #     sys_log.warning("dc_base_3T_ascend == config.loc_set.zone.base_roll_period : {:.5f} {:.5f} ({})".format(dc_base_3T_ascend[c_i], config.loc_set.zone.base_roll_period, mr_res[c_i]))

        # ------------ 2. imbalance_ratio ------------ #
        if config.loc_set.zone.ir != "None":
            itv = config.loc_set.point.tf_entry
            itv_num = to_itvnum(itv)
            if ep_loc_side == OrderSide.SELL:
                short_ir_ = res_df['short_ir_{}'.format(itv)].to_numpy()
                # short_ir_ = res_df['short_ir_{}'.format(itv)].shift(itv_num).to_numpy()

                # mr_res *= short_ir_ >= config.loc_set.zone.ir     # greater
                mr_res *= short_ir_ <= config.loc_set.zone.ir  # lesser
                if show_detail:
                    sys_log.warning(
                        "short_ir_ <= config.loc_set.zone.ir : {:.5f} {:.5f} ({})".format(short_ir_[c_i], config.loc_set.zone.ir, mr_res[c_i]))
            else:
                long_ir_ = res_df['long_ir_{}'.format(itv)].to_numpy()
                # long_ir_ = res_df['long_ir_{}'.format(itv)].shift(itv_num).to_numpy()

                # mr_res *= long_ir_ >= config.loc_set.zone.ir
                mr_res *= long_ir_ <= config.loc_set.zone.ir
                if show_detail:
                    sys_log.warning(
                        "long_ir_ <= config.loc_set.zone.ir : {:.5f} {:.5f} ({})".format(long_ir_[c_i], config.loc_set.zone.ir, mr_res[c_i]))

        # ------------ 3. body_rel_ratio ------------ #
        if config.loc_set.zone.brr != "None":
            body_rel_ratio_ = res_df['body_rel_ratio_{}'.format(config.loc_set.point.tf_entry)].to_numpy()
            mr_res *= body_rel_ratio_ >= config.loc_set.zone.brr

        # ------ dc_base ------ #
        # if strat_version in ['4']:  # 'v3_3', 'v3_4',
        #   hc_itv = '5T'
        #   dc_itv = '5T'
        #   itv_num = to_itvnum(hc_itv)
        #   close_ = res_df['close_{}'.format(hc_itv)].shift(itv_num).to_numpy()   # 따라서 future_data 사용시, shifting 필요함
        #   if ep_loc_side == OrderSide.SELL:
        #     dc_lower_ = res_df['dc_lower_%s' % dc_itv].shift(itv_num).to_numpy()
        #     mr_res *= close_ < dc_lower_
        #   else:
        #     dc_upper_ = res_df['dc_upper_%s' % dc_itv].shift(itv_num).to_numpy()
        #     mr_res *= close_ > dc_upper_

        # ------ ema ------ #
        # if strat_version in ['v5_2']: # 'v3'
        #   ema_5T = res_df['ema_5T'].to_numpy()
        #   if ep_loc_side == OrderSide.SELL:
        #     mr_res *= close < ema_5T
        #   else:
        #     mr_res *= close > ema_5T

        # ------------------ swing_middle ------------------ #
        # ------------ 1. envelope ------------ #

        # ------ a. dc ------ #
        # ep_loc check 기준 idx 가 entry 기준이라는 걸 명심
        if strat_version in ['v3_2']:
            hc_itv = '15T'
            dc_itv = '15T'
            shift_num = [0, to_itvnum(hc_itv)]
            div_res = [1, 0]
            for itv_num, res in zip(shift_num, div_res):
                close_ = res_df['close_{}'.format(hc_itv)].shift(itv_num).to_numpy()  # close_bar timein 사용하는 경우, 특수로 shift(0) 사용가능
                if ep_loc_side == OrderSide.SELL:
                    dc_lower_ = res_df['dc_lower_%s' % dc_itv].shift(itv_num).to_numpy()
                    mr_res *= (close_ < dc_lower_) == res
                else:
                    dc_upper_ = res_df['dc_upper_%s' % dc_itv].shift(itv_num).to_numpy()
                    mr_res *= (close_ > dc_upper_) == res

        # ------ b. bb ------ #
        # close = res_df['close'].to_numpy()

        # if strat_version in ['v3_3']:
        #   open = res_df['open'].to_numpy()
        #   if ep_loc_side == OrderSide.SELL:
        #     bb_lower_1m = res_df['bb_lower_1m'].to_numpy()
        #     # mr_res *= close <= bb_lower_1m
        #     mr_res *= open <= bb_lower_1m
        #   else:
        #     bb_upper_1m = res_df['bb_upper_1m'].to_numpy()
        #     # mr_res *= close >= bb_upper_1m
        #     mr_res *= open >= bb_upper_1m

        if strat_version in ['4_1']:
            if ep_loc_side == OrderSide.SELL:
                bb_lower_15T = res_df['bb_lower_15T'].to_numpy()
                short_ep_ = res_df['short_ep_{}'.format(strat_version)].to_numpy()
                mr_res *= bb_lower_15T >= short_ep_
            else:
                bb_upper_15T = res_df['bb_upper_15T'].to_numpy()
                long_ep_ = res_df['long_ep_{}'.format(strat_version)].to_numpy()
                mr_res *= bb_upper_15T <= long_ep_

        # if strat_version in ['v5_2']:
        #   bb_upper2_ = res_df['bb_upper2_%s' % config.loc_set.zone.bbz_itv].to_numpy()
        #   bb_lower2_ = res_df['bb_lower2_%s' % config.loc_set.zone.bbz_itv].to_numpy()
        #   if ep_loc_side == OrderSide.SELL:
        #     mr_res *= bb_upper2_ < close
        #   else:
        #     mr_res *= bb_lower2_ > close

        # ------------ 2. degree ------------ #
        # ------ a. norm_body_ratio ------ #
        if config.loc_set.zone.abs_ratio != "None":
            itv = config.loc_set.point.tf_entry
            abs_ratio_ = res_df['abs_ratio_{}'.format(itv)].to_numpy()
            mr_res *= abs_ratio_ >= config.loc_set.zone.abs_ratio
            # mr_res *= abs_ratio_ <= config.loc_set.zone.abs_ratio

        # degree_list = literal_eval(config.loc_set.zone.degree_list)
        # if len(degree_list) != 0:
        # # if strat_version in ['v3_3', 'v3_4']:
        #   norm_close_15 = res_df['norm_close_15'].to_numpy()   # -> 이거 뭘로 만들었는지 불분명함,,
        #   b1_norm_close_15 = res_df['norm_close_15'].shift(15).to_numpy()

        #   if ep_loc_side == OrderSide.SELL:
        #     mr_res *= norm_close_15 <= -degree_list[0]
        #     # mr_res *= b1_norm_close_15 <= -degree_list[1]
        #   else:
        #     mr_res *= norm_close_15 >= degree_list[0]
        #     # mr_res *= b1_norm_close_15 >= degree_list[1]

        # ------ b. dc ------ #
        # if strat_version in ['v3_3']:
        #   if ep_loc_side == OrderSide.SELL:
        #     dc_lower_ = res_df['dc_lower_1m'].to_numpy()
        #     b1_dc_lower_ = res_df['dc_lower_1m'].shift(1).to_numpy()
        #     mr_res *= dc_lower_ < b1_dc_lower_
        #   else:
        #     dc_upper_ = res_df['dc_upper_1m'].to_numpy()
        #     b1_dc_upper_ = res_df['dc_upper_1m'].shift(1).to_numpy()
        #     mr_res *= dc_upper_ > b1_dc_upper_

        # ------ c. sar ------ #
        # if strat_version in ['v3_3']:
        # sar_uptrend_3T = res_df['sar_uptrend_3T'].to_numpy()
        # if ep_loc_side == OrderSide.SELL:
        #   mr_res *= sar_uptrend_3T == 0
        # else:
        #   mr_res *= sar_uptrend_3T == 1

    # -------------- zoned tr_set - post_work -------------- #
    if config.tr_set.c_ep_gap != "None" or config.tr_set.t_out_gap != "None":
        #       by zone_dtk       #
        #         c_zone        #
        zone_dc_upper_v2_ = res_df['zone_dc_upper_v2'.format(strat_version)].to_numpy()
        long_dtk_plot_1 = res_df['long_dtk_plot_1'].to_numpy() + res_df['long_dtk_plot_gap'].to_numpy() * config.loc_set.zone.zone_dt_k
        zone_dc_lower_v2_ = res_df['zone_dc_lower_v2'.format(strat_version)].to_numpy()
        short_dtk_plot_1 = res_df['short_dtk_plot_1'].to_numpy() - res_df['short_dtk_plot_gap'].to_numpy() * config.loc_set.zone.zone_dt_k
        if ep_loc_side == OrderSide.SELL:
            zone_res = zone_dc_upper_v2_ > long_dtk_plot_1  # mr_res 와는 별개임
            pos = 'short'
        else:
            zone_res = zone_dc_lower_v2_ < short_dtk_plot_1
            pos = 'long'

        # static 여부에 따른 vectorized adj. - dynamic 은 vectorize 불가
        if config.ep_set.static_ep and config.tr_set.c_ep_gap != "None":
            res_df['{}_ep_{}'.format(pos, strat_version)][zone_res] = res_df['{}_ep2_{}'.format(pos, strat_version)][zone_res]
        if config.out_set.static_out and config.tr_set.t_out_gap != "None":
            res_df['{}_out_{}'.format(pos, strat_version)][~zone_res] = res_df['{}_out2_{}'.format(pos, strat_version)][
                ~zone_res]  # t_zone 에 대한 out2 setting
        zone_arr = np.where(zone_res == 1, 'c', 't')

    return mr_res, zone_arr  # mr_res 의 True idx 가 open signal
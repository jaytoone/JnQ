from funcs.funcs_indicator import *
from funcs.funcs_trader import *
import logging
from ast import literal_eval

sys_log = logging.getLogger()


class OrderSide:
    BUY = "BUY"
    SELL = "SELL"
    INVALID = None


def lvrg_set(res_df, config, open_side, ep_, out_, fee, limit_leverage=50):
    strat_version = config.strat_version
    if not pd.isnull(out_) and not config.lvrg_set.static_lvrg:
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

    # ------------ leverage rejection ------------ #
    #       Todo - return None ? -> 1 (일단 임시로 수정함)
    if config.lvrg_set.leverage < 1 and config.lvrg_set.lvrg_rejection:
        return 1
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

    except Exception as e:
        sys_log.error("error in sync_check :", e)
    else:
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

        # sys_log.warning("~ h_wick_score elapsed time : {}".format(time.time() - start_0))
        # print("wick_score() ~ h_candle() elapsed time : {}".format(time.time() - start_0))

        h_candle_col = ['hopen_{}'.format(h_c_intv2), 'hhigh_{}'.format(h_c_intv2), 'hlow_{}'.format(h_c_intv2),
                        'hclose_{}'.format(h_c_intv2)]

        res_df['h_wick_score'], res_df['h_body_score'] = candle_score(res_df, ohlc_col=h_candle_col, unsigned=False)

    #     temp indi.    #
    # res_df["ma30_1m"] = res_df['close'].rolling(30).mean()
    # res_df["ma60_1m"] = res_df['close'].rolling(60).mean()
    # res_df = dtk_plot(res_df, dtk_itv2='15m', hhtf_entry=15, use_dtk_line=config.loc_set.zone.use_dtk_line, np_timeidx=np_timeidx)

    return res_df


def ep_loc_point2(res_df, config, i, out_j, side=OrderSide.SELL):
  allow_ep_in = 0
  if config.strat_version in ['v5_2']:
    if side == OrderSide.SELL:            
      if (res_df['dc_upper_1m'].iloc[i - 1] <= res_df['dc_upper_15m'].iloc[i]) & \
              (res_df['dc_upper_15m'].iloc[i - 1] != res_df['dc_upper_15m'].iloc[i]):
        allow_ep_in = 1
    else:
      if (res_df['dc_lower_1m'].iloc[i - 1] >= res_df['dc_lower_15m'].iloc[i]) & \
              (res_df['dc_lower_15m'].iloc[i - 1] != res_df['dc_lower_15m'].iloc[i]):
        allow_ep_in = 1

  if allow_ep_in:
    out_j = i
  return allow_ep_in, out_j


# vectorized calc.
def ep_loc_v2(res_df, config, np_timeidx, show_detail=True, ep_loc_side=OrderSide.SELL):
    strat_version = config.strat_version

    # ------- param init ------- #
    open_side = None
    len_df = len(res_df)
    mr_res = np.ones(len_df)
    zone_arr = np.full(len_df, 'n')

    # ------ process 한번에 처리해서 param_check 만 ver. 별로 하면 될 것 ------ #
    #     => public_indi() 가 될 것
    #     1. 사용한 param 정보와 matching 된 data 병렬로 나열 logging 될 것
    #     2. binarize 할 것 => short 와 long 은 같은 조건식 사용할 수 있게됨 - 보류 (생각보다 상반된 조건식이 많음)
    #       a. 조건식이 다른 경우는 ?
    tp_fee, out_fee = calc_tp_out_fee(config)

    # -------------- candle_score -------------- #
    if config.loc_set.point.wick_score != "None":
      # -------------- candle_score_v0 (1m initial tick 기준임)  -------------- #
      if strat_version in ['v5_2', '1_1']:
        wick_score = res_df['wick_score'].to_numpy()
        if ep_loc_side == OrderSide.SELL:
          mr_res *= wick_score <= -config.loc_set.point.wick_score
        else:
          mr_res *= wick_score >= config.loc_set.point.wick_score

    # -------------- spread scheduling -------------- #
    if config.loc_set.zone.short_spread != "None":
      bb_base_5m = res_df['bb_base_5m'].to_numpy()  # to_numpy() 는 ep_loc 에서 진행됨
      bb_lower_5m = res_df['bb_lower_5m'].to_numpy()
      bb_upper_5m = res_df['bb_upper_5m'].to_numpy()
      dc_lower_5m = res_df['dc_lower_5m'].to_numpy()
      if ep_loc_side == OrderSide.SELL:
        mr_res *= (bb_base_5m - bb_lower_5m - tp_fee * bb_base_5m) / (bb_base_5m - bb_lower_5m + out_fee * bb_base_5m) >= config.loc_set.zone.short_spread
      else:
        mr_res *= (bb_base_5m - dc_lower_5m - tp_fee * bb_upper_5m) / (bb_base_5m - dc_lower_5m + out_fee * bb_base_5m) >= config.loc_set.zone.short_spread

    # -------------- dtk -------------- #
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
      else:
        mr_res *= dc_upper_v2 <= long_dtk_1_

    # -------------- zone rejection  -------------- #
    if config.loc_set.zone.zone_rejection:
        #       config 로 통제할 수 없는 rejection 은 strat_version 으로 조건문을 나눔 (lvrg_set 과 동일)
        # ------------ by bb ------------ #
        # ------ bb & close ------ #
        close = res_df['close'].to_numpy()

        if strat_version in ['v5_2']:
          bb_upper2_ = res_df['bb_upper2_%s' % config.loc_set.zone.bbz_itv].to_numpy()
          bb_lower2_ = res_df['bb_lower2_%s' % config.loc_set.zone.bbz_itv].to_numpy()
          if ep_loc_side == OrderSide.SELL:
            mr_res *= bb_upper2_ < close
          else:
            mr_res *= bb_lower2_ > close

        # ------ close, bb & ema ------ #
        if strat_version in ['v5_2']:
          ema_5m = res_df['ema_5m'].to_numpy()
          if ep_loc_side == OrderSide.SELL:
            mr_res *= close < ema_5m
          else:
            mr_res *= close > ema_5m

    # -------------- zoned tr_set - post_work -------------- #
    if config.tr_set.c_ep_gap != "None":
        #       by zone_dtk       #
        #         c_zone        #
        zone_dc_upper_v2_ = res_df['zone_dc_upper_v2'.format(strat_version)].to_numpy()
        long_dtk_plot_1 = res_df['long_dtk_plot_1'].to_numpy() + res_df['long_dtk_plot_gap'].to_numpy() * config.loc_set.zone.zone_dt_k
        zone_dc_lower_v2_ = res_df['zone_dc_lower_v2'.format(strat_version)].to_numpy()
        short_dtk_plot_1 = res_df['short_dtk_plot_1'].to_numpy() - res_df['short_dtk_plot_gap'].to_numpy() * config.loc_set.zone.zone_dt_k
        if ep_loc_side == OrderSide.SELL:
          zone_res = zone_dc_upper_v2_ > long_dtk_plot_1  # mr_res 와는 별개임
          # 지금의 생각은, dynamic 이여도, 그대로 ep2 를 입히면 될 것으로 보이는데, 어차피 zone_res 가 True 인 곳은 모두 c_zone 이니까
          res_df['short_ep_{}'.format(strat_version)][zone_res] = res_df['short_ep2_{}'.format(strat_version)][zone_res]
          res_df['short_out_{}'.format(strat_version)][~zone_res] = res_df['short_out2_{}'.format(strat_version)][~zone_res]  # t_zone 에 대한 out2 setting   
        else:
          zone_res = zone_dc_lower_v2_ < short_dtk_plot_1
          res_df['long_ep_{}'.format(strat_version)] = res_df['long_ep2_{}'.format(strat_version)]
          res_df['long_out_{}'.format(strat_version)][~zone_res] = res_df['long_out2_{}'.format(strat_version)][~zone_res]  # t_zone 에 대한 out2 setting        
        zone_arr = np.where(zone_res == 1, 'c', 't')

    return mr_res, zone_arr  # mr_res 의 True idx 가 open signal
from funcs.olds.funcs_indicator_candlescore import *
from funcs.public.broker import *
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
    df = bb_line(df, None, '1m')
    df = bb_line(df, third_df, '5m')
    df = dc_line(df, third_df, '5m')
    df = bb_line(df, fourth_df, '15m')
    df = dc_line(df, fourth_df, '15m')

    df['rsi_1m'] = rsi(df, 14)

    if order_side in ["OPEN"]:

        third_df['ema_5m'] = ema(third_df['close'], 200)
        df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf_v2(df, third_df, [-1]), columns=['ema_5m']))

    return df


def public_indi(res_df, np_timeidx, order_side="OPEN"):

    res_df = bb_level(res_df, '5m', 1)
    res_df = dc_level(res_df, '5m', 1)
    res_df = bb_level(res_df, '15m', 1)
    res_df = dc_level(res_df, '15m', 1)
    # res_df = bb_level(res_df, '30m', 1)
    # res_df = dc_level(res_df, '30m', 1)

    # res_df = st_level(res_df, '5m', 1)
    # res_df = st_level(res_df, '15m', 1)
    # res_df = st_level(res_df, '30m', 1)

    if order_side in ["OPEN"]:

        start_0 = time.time()

        res_df["candle_ratio"], res_df['body_ratio'] = candle_ratio(res_df, unsigned=False)

        print("~ candle_ratio() elapsed time : {}".format(time.time() - start_0))

        start_0 = time.time()

        # h_c_intv1 = 15
        # h_c_intv2 = 60
        # res_df = h_candle(res_df, h_c_intv1)
        # res_df = h_candle(res_df, h_c_intv2)
        
        h_c_intv1 = '15T'
        h_c_intv2 = 'H'
        res_df = h_candle_v2(res_df, h_c_intv1)
        res_df = h_candle_v2(res_df, h_c_intv2)

        # sys_log3.warning("~ h_candle_ratio elapsed time : {}".format(time.time() - start_0))
        print("candle_ratio() ~ h_candle() elapsed time : {}".format(time.time() - start_0))

        h_candle_col = ['hopen_{}'.format(h_c_intv2), 'hhigh_{}'.format(h_c_intv2), 'hlow_{}'.format(h_c_intv2), 'hclose_{}'.format(h_c_intv2)]

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
        pass
    else:
      return 0, out_j # input out_j could be initial_i    
    
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

  if config.strat_version =='v5_2' and allow_ep_in == 0:

    if (res_df['dc_lower_1m'].iloc[i - 1] >= res_df['dc_lower_15m'].iloc[i]) & \
        (res_df['dc_lower_15m'].iloc[i - 1] != res_df['dc_lower_15m'].iloc[i]):
        pass
    else:
      return 0, out_j # input out_j could be initial_i

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


def short_ep_loc(res_df, config, i, np_timeidx, show_detail=True):

    strat_version = config.strat_version

    # ------- param init ------- #
    open_side = None

    # mr_const_cnt = 0
    # mr_score = 0
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

        if show_detail:
          sys_log3.warning("candle_ratio_0 : {}".format(candle_ratio_))

        # if candle_ratio_ >= config.loc_set.point.candle_ratio:
        if candle_ratio_ <= -config.loc_set.point.candle_ratio:
          pass

        else:
          return res_df, open_side, zone


      # -------------- candle_ratio_v1 (previous)  -------------- #
      if strat_version in ['v7_3', '2_2', 'v3']:

        prev_hclose_idx = i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1)

        if prev_hclose_idx >= 0:
          
          h_candle_ratio_ = res_df['h_candle_ratio'].iloc[prev_hclose_idx]
          h_body_ratio_ = res_df['h_body_ratio'].iloc[prev_hclose_idx]

          if strat_version in ['v7_3']:
            
            if show_detail:
                sys_log3.warning("candle_ratio1 : {}".format(h_candle_ratio_ + h_body_ratio_/100))
            
            if h_candle_ratio_ + h_body_ratio_/100 <= -config.loc_set.point.candle_ratio:
                pass
            else:
              return res_df, open_side, zone   
          
          elif strat_version in ['2_2', 'v3', '1_1']:
            
            if show_detail:
                sys_log3.warning("candle_ratio1 : {}".format(h_candle_ratio_))

            
            if h_candle_ratio_ <= -config.loc_set.point.candle_ratio:
                pass
            else:
              return res_df, open_side, zone   


            if config.loc_set.point.body_ratio != "None":
              
              if show_detail:
                  sys_log3.warning("body_ratio1 : {}".format(h_body_ratio_))

              
              if h_body_ratio_ >= config.loc_set.point.body_ratio:
                  pass
              else:
                return res_df, open_side, zone   


    if config.loc_set.point.candle_ratio2 != "None":

        #     candle_ratio_v2 (current)     #
      prev_hclose_idx = i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1)

      if prev_hclose_idx >= -1:
        hc_res_df = res_df.iloc[prev_hclose_idx + 1:i + 1].copy()
        ho = hc_res_df['open'].iloc[0]
        hh = hc_res_df['high'].max()
        hl = hc_res_df['low'].min()
        hc = hc_res_df['close'].iloc[-1]


        if strat_version in ['1_3', '1_1']:
          score, body_score = candle_score(ho, hh, hl, hc, updown=None, unsigned=False)
        else:
          score, body_score = candle_score(ho, hh, hl, ho, updown=None, unsigned=False)

        if show_detail:
          sys_log3.warning("candle_ratio2 : {}".format(score))

        
        if score <= -config.loc_set.point.candle_ratio2:
          pass
        else:
          return res_df, open_side, zone   

        if config.loc_set.point.body_ratio2 != "None":

          if show_detail:
            sys_log3.warning("body_ratio2 : {}".format(body_score))

          
          if ho > hc and body_score >= config.loc_set.point.body_ratio2:
            pass
          else:
            return res_df, open_side, zone   

    
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

        if show_detail:
            sys_log3.warning("spread : {}".format(spread))

        if spread >= config.loc_set.zone.short_spread:
            pass
        else:
          return res_df, open_side, zone   


    # -------------- dtk -------------- #
    if config.loc_set.zone.dt_k != "None":

        
        # if res_df['dc_lower_%s' % config.loc_set.zone.dtk_dc_itv].iloc[i] >= res_df['short_rtc_1'].iloc[i] - res_df['h_short_rtc_gap'].iloc[i] * config.loc_set.zone.dt_k:
        #     dtk_v1 & v2 platform     #
        if config.loc_set.zone.dtk_dc_itv != "None":
            dc = res_df['dc_lower_%s' % config.loc_set.zone.dtk_dc_itv].iloc[i]
            dt_k = res_df['short_dtk_1_{}'.format(strat_version)].iloc[i] - \
                  res_df['short_dtk_gap_{}'.format(strat_version)].iloc[i] * config.loc_set.zone.dt_k
                  
            if show_detail:
                sys_log3.warning("dc : {}".format(dc))
                sys_log3.warning("dt_k : {}".format(dt_k))

            if dc >= dt_k:
                pass
            else:
              return res_df, open_side, zone   

                #     dc_v2   #
        else:
            dc = res_df['dc_lower_v2_{}'.format(strat_version)].iloc[i]
            dt_k = res_df['short_dtk_1_{}'.format(strat_version)].iloc[i] - \
                  res_df['short_dtk_gap_{}'.format(strat_version)].iloc[i] * config.loc_set.zone.dt_k

            if show_detail:
                sys_log3.warning("dc : {}".format(dc))
                sys_log3.warning("dt_k : {}".format(dt_k))

            if dc >= dt_k:
                # if res_df['dc_lower_v2_{}'.format(strat_version)].iloc[i] >= res_df['short_dtk_1_{}'.format(strat_version)].iloc[i] - res_df['short_dtk_gap_{}'.format(strat_version)].iloc[i] * config.loc_set.zone.dt_k and \
                # res_df['dc_upper_v2_{}'.format(strat_version)].iloc[i] <= res_df['long_dtk_1_{}'.format(strat_version)].iloc[i] + res_df['long_dtk_gap_{}'.format(strat_version)].iloc[i] * config.loc_set.zone.dt_k:
                pass
            else:
              return res_df, open_side, zone   

            
      # -------------- candle_dt_k -------------- #
    # 
    # # if res_df['dc_lower_1m'].iloc[i] >= res_df['hclose_60'].iloc[i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1)]:
    # if res_df['dc_lower_1m'].iloc[i] >= res_df['hlow_60'].iloc[i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1)]:
    #   pass        

    # 
    # if res_df['dc_upper_1m'].iloc[i] <= res_df['hhigh_60'].iloc[i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1)]:
    # # if res_df['dc_upper_1m'].iloc[i] <= res_df['hopen_60'].iloc[i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1)]:
    #   pass  


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

          if show_detail:
            sys_log3.warning("bb_upper2_ & close : {} {}".format(bb_, close_))

          # if res_df['bb_upper3_%s' % config.loc_set.zone.bbz_itv].iloc[i] > res_df['close'].iloc[i] > res_df['bb_upper2_%s' % config.loc_set.zone.bbz_itv].iloc[i]:
          if close_ > bb_:
              pass
          else:
            return res_df, open_side, zone   

          #     bb & bb   #           
        if strat_version in ['v7_3', '1_1']:

          bb_ = res_df['bb_upper_5m'].iloc[i]
          bb_1 = res_df['bb_base_%s' % config.loc_set.zone.bbz_itv].iloc[i]
          
          if show_detail:
              sys_log3.warning("bb_upper_5m & bb_base_ : {} {}".format(bb_, bb_1))

          # if res_df['bb_upper_1m'].iloc[i] < res_df['bb_lower_%s' % config.loc_set.zone.bbz_itv].iloc[i]:
          if bb_ < bb_1:
            pass                
          else:
            return res_df, open_side, zone   

            #     bb & ep   #          
          ep_ = res_df['short_ep_{}'.format(strat_version)].iloc[i]
          bb_ = res_df['bb_upper_5m'].iloc[i]

          if show_detail:
            sys_log3.warning("bb_upper_5m & short_ep_ : {} {}".format(bb_, ep_))

          if ep_ < bb_:
              pass                
          else:
            return res_df, open_side, zone   

            #     bb & dc   #
          
          # if res_df['bb_base_%s' % config.loc_set.zone.bbz_itv].iloc[i] <= res_df['dc_upper_1m'].iloc[i] <= res_df['bb_upper_%s' % config.loc_set.zone.bbz_itv].iloc[i]:
          
          prev_hopen_idx = i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1 + config.loc_set.zone.c_itv_ticks) + config.loc_set.zone.ad_idx

          if prev_hopen_idx >= 0:
            # if res_df['dc_upper_5m'].iloc[prev_hopen_idx] < res_df['bb_upper_15m'].iloc[i]:

            dc_ = res_df['dc_upper_5m'].iloc[prev_hopen_idx]
            bb_ = res_df['bb_upper_15m'].iloc[prev_hopen_idx]

            if show_detail:
                sys_log3.warning("bb_upper_15m & dc_upper_5m : {} {}".format(bb_, dc_))

            if dc_ < bb_:
              pass                
            else:
              return res_df, open_side, zone   

          # --------- by ema --------- # 

          #    dc & ema   #
        if strat_version in ['v7_3', '1_1']:
          
          dc_ = res_df['dc_upper_5m'].iloc[i]
          ema_ = res_df['ema_5m'].iloc[i]
          
          if show_detail:
              sys_log3.warning("dc_upper_5m & ema_5m : {} {}".format(dc_, ema_))

          # if res_df['bb_upper_15m'].iloc[i] < res_df['ema_5m'].iloc[i]:
          if dc_ < ema_:
            pass                
          else:
            return res_df, open_side, zone   



          #    close, bb & ema   #
        if strat_version in ['v5_2', 'v3']:
          
          close_ = res_df['close'].iloc[i]
          ema_ = res_df['ema_5m'].iloc[i]
          bb_ = res_df['bb_upper_30m'].iloc[i]
          
          if show_detail:
            sys_log3.warning("close & ema_5m : {} {}".format(close_, ema_))
            
          # if res_df['short_ep'].iloc[i] < res_df['ema_5m'].iloc[i]:
          if close_ < ema_:
              pass                
          else:
            return res_df, open_side, zone   

          # if ema_ < bb_:
          #   pass
          # else:
          #   return res_df, open_side, zone   

        # --------- by st --------- # 
        # if strat_version in ['v5_2']:
        #   
        #   if res_df['close'].iloc[i] < res_df['st_base_5m'].iloc[i]:
        #   # if res_df['close'].iloc[i] > res_df['st_base_5m'].iloc[i]:
        #       pass

        #       if show_detail:
        #         sys_log3.warning("close & st : {} {}".format(bb_, bb_1))

        # --------- by dc --------- # 
        
          #     descending dc    #
        # 
        # if res_df['dc_lower_5m'].iloc[i] <= res_df['dc_lower_5m'].iloc[i - 50 : i].min():
        #   pass

        # --------- by candle --------- #
        # if strat_version in ['2_2']:

        #   prev_hclose_idx = i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1)
        #   if prev_hclose_idx >= 0:
            
        #     # if res_df['short_ep_{}'.format(strat_version)].iloc[i] <= res_df['hclose_60'].iloc[prev_hclose_idx)]:
        #     if res_df['close'].iloc[i] <= res_df['hclose_60'].iloc[prev_hclose_idx]:
        #         pass
                
        #     else:
        #       return res_df, open_side, zone   

        # --------- by macd --------- #
        # 
        # if res_df['ma30_1m'].iloc[i] < res_df['ma60_1m'].iloc[i]:
        #     pass


        # --------- by zone_dtk --------- #
        # 
        # if res_df['zone_dc_upper_v2_{}'.format(strat_version)].iloc[i] < res_df['long_dtk_plot_1'].iloc[i] + res_df['long_dtk_plot_gap'].iloc[
        #     i] * config.loc_set.zone.zone_dt_k:
        #   pass

    # -------------- zoned tr_set - post_Work -------------- #
    if config.tr_set.c_ep_gap != "None":

        #       by bb       # 
        # if res_df['close'].iloc[i] > res_df['bb_lower_%s' % config.loc_set.zone.bbz_itv].iloc[i]:

        #       by zone_dtk       #

        #         c_zone        #
        zone_dc_ =  res_df['zone_dc_upper_v2_{}'.format(strat_version)].iloc[i]
        l_dtk_ = res_df['long_dtk_plot_1'].iloc[i] + res_df['long_dtk_plot_gap'].iloc[i] * config.loc_set.zone.zone_dt_k

        if show_detail:
          sys_log3.warning("zone_dc_upper_v2_ & long_dtk_plot_ : {} {}".format(zone_dc_, l_dtk_))

        if zone_dc_ > l_dtk_:
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

    return res_df, OrderSide.SELL, zone


def long_ep_loc(res_df, config, i, np_timeidx, show_detail=True):

    strat_version = config.strat_version

    # ------- param init ------- #
    open_side = None

    # mr_const_cnt = 0
    # mr_score = 0
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

        if show_detail:
            sys_log3.warning("candle_ratio_0 : {}".format(candle_ratio_))

        if candle_ratio_ >= config.loc_set.point.candle_ratio:
          pass
                
        else:
          return res_df, open_side, zone   

      # -------------- candle_ratio_v1 (previous)  -------------- #
      if strat_version in ['v7_3', '2_2', 'v3']:

        prev_hclose_idx = i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1)
        
        if prev_hclose_idx >= 0:

          h_candle_ratio_ = res_df['h_candle_ratio'].iloc[prev_hclose_idx]
          h_body_ratio_ = res_df['h_body_ratio'].iloc[prev_hclose_idx]

          if strat_version in ['v7_3']:

            if show_detail:
              sys_log3.warning("candle_ratio1 : {}".format(h_candle_ratio_ + h_body_ratio_/100))

            
            if h_candle_ratio_ + h_body_ratio_/100 >= config.loc_set.point.candle_ratio:
              pass
                
            else:
              return res_df, open_side, zone   

          elif strat_version in ['2_2', 'v3', '1_1']:
          
            if show_detail:
              sys_log3.warning("candle_ratio1 : {}".format(h_candle_ratio_))    

            
            if h_candle_ratio_ >= config.loc_set.point.candle_ratio:
              pass
                
            else:
              return res_df, open_side, zone   
    

            if config.loc_set.point.body_ratio != "None":
              
              if show_detail:
                  sys_log3.warning("body_ratio1 : {}".format(h_body_ratio_))

              
              if h_body_ratio_ >= config.loc_set.point.body_ratio:
                  pass
                
              else:
                return res_df, open_side, zone   


    if config.loc_set.point.candle_ratio2 != "None":

      #     candle_ratio_v2 (current)     #
      prev_hclose_idx = i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1)

      if prev_hclose_idx >= -1:

        hc_res_df = res_df.iloc[prev_hclose_idx + 1:i + 1].copy()
        ho = hc_res_df['open'].iloc[0]
        hc = hc_res_df['close'].iloc[-1]
        hh = hc_res_df['high'].max()
        hl = hc_res_df['low'].min()

        if strat_version in ['1_3', '1_1']:
          score, body_score = candle_score(ho, hh, hl, hc, updown=None, unsigned=False)
        else:
          score, body_score = candle_score(ho, hh, hl, ho, updown=None, unsigned=False)
        
        if show_detail:
          sys_log3.warning("candle_ratio2 : {}".format(score))

        
        if score >= config.loc_set.point.candle_ratio2:
          pass
                
        else:
          return res_df, open_side, zone   

        # print("candle_ratio2 : {} {} .format(bb_, bb_1)!")

        if config.loc_set.point.body_ratio2 != "None":
          
          if show_detail:
            sys_log3.warning("body_ratio2 : {}".format(body_score))

          
          if ho < hc and body_score >= config.loc_set.point.body_ratio2:
            pass
                
          else:
            return res_df, open_side, zone   


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
        
        if show_detail:
            sys_log3.warning("spread : {}".format(spread))

        if spread >= config.loc_set.zone.long_spread:
            pass
        else:
          return res_df, open_side, zone   


    # -------------- dtk -------------- #
    if config.loc_set.zone.dt_k != "None":

        
        # if res_df['dc_upper_%s' % config.loc_set.zone.dtk_dc_itv].iloc[i] <= res_df['long_rtc_1'].iloc[i] + res_df['long_rtc_gap'].iloc[i] * config.loc_set.zone.dt_k:
        #     dtk_v1 & v2 platform    #
        if config.loc_set.zone.dtk_dc_itv != "None":
            dc = res_df['dc_upper_%s' % config.loc_set.zone.dtk_dc_itv].iloc[i]
            dt_k = res_df['long_dtk_1_{}'.format(strat_version)].iloc[i] + \
                  res_df['long_dtk_gap_{}'.format(strat_version)].iloc[i] * config.loc_set.zone.dt_k
                  
            if show_detail:
                sys_log3.warning("dc : {}".format(dc))
                sys_log3.warning("dt_k : {}".format(dt_k))

            if dc <= dt_k:
                pass
            else:
              return res_df, open_side, zone  

        else:
            #     dc_v2     #
            dc = res_df['dc_upper_v2_{}'.format(strat_version)].iloc[i]
            dt_k = res_df['long_dtk_1_{}'.format(strat_version)].iloc[i] + \
                  res_df['long_dtk_gap_{}'.format(strat_version)].iloc[i] * config.loc_set.zone.dt_k

            if show_detail:
                sys_log3.warning("dc : {}".format(dc))
                sys_log3.warning("dt_k : {}".format(dt_k))

            if dc <= dt_k:
                # if res_df['dc_upper_v2_{}'.format(strat_version)].iloc[i] >= res_df['long_dtk_1_{}'.format(strat_version)].iloc[i] + res_df['long_dtk_gap_{}'.format(strat_version)].iloc[i] * config.loc_set.zone.dt_k:

                # if res_df['dc_upper_v2_{}'.format(strat_version)].iloc[i] <= res_df['long_dtk_1_{}'.format(strat_version)].iloc[i] + res_df['long_dtk_gap_{}'.format(strat_version)].iloc[i] * config.loc_set.zone.dt_k and \
                #   res_df['dc_lower_v2_{}'.format(strat_version)].iloc[i] >= res_df['short_dtk_1_{}'.format(strat_version)].iloc[i] - res_df['short_dtk_gap_{}'.format(strat_version)].iloc[i] * config.loc_set.zone.dt_k:

                pass
            else:
              return res_df, open_side, zone  

      # -------------- candle_dt_k -------------- #
    # 
    # # if res_df['dc_upper_1m'].iloc[i] <= res_df['hclose_60'].iloc[i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1)]:
    # if res_df['dc_upper_1m'].iloc[i] <= res_df['hhigh_60'].iloc[i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1)]:
    #   pass  

    # 
    # if res_df['dc_lower_1m'].iloc[i] >= res_df['hlow_60'].iloc[i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1)]:
    # # if res_df['dc_lower_1m'].iloc[i] >= res_df['hopen_60'].iloc[i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1)]:
    #   pass  

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

          if show_detail:
            sys_log3.warning("bb_lower2_ & close : {} {}".format(bb_, close_))

          if close_ < bb_:
          # if res_df['bb_lower3_%s' % config.loc_set.zone.bbz_itv].iloc[i] < res_df['close'].iloc[i] < res_df['bb_lower2_%s' % config.loc_set.zone.bbz_itv].iloc[i]:
          # if res_df['close'].iloc[i] < res_df['bb_lower3_%s' % config.loc_set.zone.bbz_itv].iloc[i]:
              pass
          else:
            return res_df, open_side, zone   

          #     bb & bb   #
        if strat_version in ['v7_3', '1_1']:

          
          bb_ = res_df['bb_lower_5m'].iloc[i]
          bb_1 = res_df['bb_base_%s' % config.loc_set.zone.bbz_itv].iloc[i]

          if show_detail:
              sys_log3.warning("bb_lower_5m & bb_base_ : {} {}".format(bb_, bb_1))

          if bb_ > bb_1:
          # if res_df['bb_lower_1m'].iloc[i] > res_df['bb_upper_%s' % config.loc_set.zone.bbz_itv].iloc[i]:            
              pass
          else:
            return res_df, open_side, zone   

            #     bb & ep   #
          ep_ = res_df['long_ep_{}'.format(strat_version)].iloc[i]
          bb_ = res_df['bb_lower_5m'].iloc[i]
          
          if show_detail:
            sys_log3.warning("bb_lower_5m & long_ep_ : {} {}".format(bb_, ep_))

          if ep_ > bb_:
              pass
          else:
            return res_df, open_side, zone   
          
            #     bb & dc   #
          
          # if res_df['bb_base_%s' % config.loc_set.zone.bbz_itv].iloc[i] >= res_df['dc_lower_1m'].iloc[i] >= res_df['bb_lower_%s' % config.loc_set.zone.bbz_itv].iloc[i]:

          prev_hopen_idx = i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1 + config.loc_set.zone.c_itv_ticks) + config.loc_set.zone.ad_idx
          
          if prev_hopen_idx >= 0:            
            # if res_df['dc_lower_5m'].iloc[prev_hopen_idx] > res_df['bb_lower_15m'].iloc[i]:

            dc_ = res_df['dc_lower_5m'].iloc[prev_hopen_idx]
            bb_ = res_df['bb_lower_15m'].iloc[prev_hopen_idx]

            if show_detail:
                sys_log3.warning("bb_lower_15m & dc_lower_5m : {} {}".format(bb_, dc_))

            if dc_ > bb_:
              pass
            else:
              return res_df, open_side, zone   

        # --------- by ema --------- # 

          #     dc & ema   #
        if strat_version in ['v7_3', '1_1']:
          
          dc_ = res_df['dc_lower_5m'].iloc[i]
          ema_ = res_df['ema_5m'].iloc[i]

          if show_detail:
              sys_log3.warning("dc_lower_5m & ema_5m : {} {}".format(dc_, ema_))

          if dc_ > ema_:
            pass
          else:
            return res_df, open_side, zone   

          #     close, bb & ema     #
        if strat_version in ['v5_2', 'v3']:
          
          close_ = res_df['close'].iloc[i]
          ema_ = res_df['ema_5m'].iloc[i]
          bb_ = res_df['bb_lower_30m'].iloc[i]

          if show_detail:
              sys_log3.warning("close & ema_5m : {} {}".format(close_, ema_))

          # if  res_df['long_ep'].iloc[i] > ema_:
          if close_ > ema_:
              pass
          else:
            return res_df, open_side, zone   
            
          # if show_detail:
          #     sys_log3.warning("bb_lower_30m & ema_5m : {} {}".format(bb_, ema_))

          # if ema_ > bb_:
          #   pass
          # else:
          #   return res_df, open_side, zone   

        # if strat_version in ["2_2"]:

        #   
        #   # if  res_df['long_ep'].iloc[i] > res_df['ema_5m'].iloc[i]:
        #   if res_df['close'].iloc[i] < res_df['ema_5m'].iloc[i]:
        #       pass

        #       if show_detail:
        #         sys_log3.warning("close & ema : {} {}".format(bb_, bb_1))

        # --------- by st --------- # 
        # if strat_version in ['v5_2']:
        #   
        #   if res_df['close'].iloc[i] > res_df['st_base_5m'].iloc[i]:
        #   # if res_df['close'].iloc[i] < res_df['st_base_15m'].iloc[i]:
        #       pass

        #       if show_detail:
        #         sys_log3.warning("close & st : {} {}".format(bb_, bb_1))
          
        # --------- by dc --------- # 

          #     ascending dc    #
        # 
        # if res_df['dc_upper_5m'].iloc[i] >= res_df['dc_upper_5m'].iloc[i - 50 : i].max():
        #   pass          

        # --------- by candle --------- #
        # if strat_version in ['2_2']:

        #   prev_hclose_idx = i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1)
        #   if prev_hclose_idx >= 0:
            
        #     if res_df['close'].iloc[i] >= res_df['hclose_60'].iloc[prev_hclose_idx]:
        #         pass
        #     else:
        #       return res_df, open_side, zone   
        
        # --------- by macd --------- #
        # 
        # if res_df['ma30_1m'].iloc[i] > res_df['ma60_1m'].iloc[i]:
        #     pass

        # --------- by zone_dtk --------- #
        # 
        # if res_df['zone_dc_lower_v2_{}'.format(strat_version)].iloc[i] > res_df['short_dtk_plot_1'].iloc[i] - res_df['short_dtk_plot_gap'].iloc[i] * config.loc_set.zone.zone_dt_k:
        #   pass

    # -------------- zoned tr_set - post_work -------------- #
    if config.tr_set.c_ep_gap != "None":
        #       by bb       # 
        # if res_df['close'].iloc[i] < res_df['bb_upper_%s' % config.loc_set.zone.bbz_itv].iloc[i]:

        #       by zone_dtk       #

        #         c_zone        #
        zone_dc_ =  res_df['zone_dc_lower_v2_{}'.format(strat_version)].iloc[i]
        l_dtk_ = res_df['short_dtk_plot_1'].iloc[i] - res_df['short_dtk_plot_gap'].iloc[i] * config.loc_set.zone.zone_dt_k
        
        if show_detail:
          sys_log3.warning("zone_dc_lower_v2_ & short_dtk_plot : {} {}".format(zone_dc_, l_dtk_))

        if zone_dc_ < l_dtk_:

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
            #   pass

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

    return res_df, OrderSide.BUY, zone


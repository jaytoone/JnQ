from funcs.funcs_duration import *
# from funcs.funcs_idep import get_col_idxs

#       Todo        #
#        1. future_data warning
#        2. lookback indi. warning - index error
#        3. ep_loc 의 i 는 close 기준임 -> 즉, last_index = -2 warning


def lvrg_set(res_df, config, open_side, ep_, out_, fee, limit_leverage=50):
    strat_version = config.strat_version
    if not pd.isnull(out_):
        if open_side == OrderSide.SELL:
            if strat_version in ['v3']:
                config.lvrg_set.leverage = config.lvrg_set.target_pct / (
                        out_ / ep_ - 1 - (fee + config.trader_set.market_fee))
                #     zone 에 따른 c_ep_gap 를 고려 (loss 완화 방향) / 윗 줄은 수익 극대화 방향
                # config.lvrg_set.leverage = config.lvrg_set.target_pct / (out_ / res_df['short_ep_org'].iloc[ep_j] - 1 - (fee + config.trader_set.market_fee))
            elif strat_version in ['v5_2', 'v7_3']:
                config.lvrg_set.leverage = config.lvrg_set.target_pct / abs(
                    ep_ / out_ - 1 - (fee + config.trader_set.market_fee))
                # config.lvrg_set.leverage = config.lvrg_set.target_pct / abs(res_df['short_ep_org'].iloc[ep_j] / out_ - 1 - (fee + config.trader_set.market_fee))
        else:
            #   윗 phase 는 min_pr 의 오차가 커짐
            if strat_version in ['v3']:
                config.lvrg_set.leverage = config.lvrg_set.target_pct / (
                        ep_ / out_ - 1 - (fee + config.trader_set.market_fee))
                # config.lvrg_set.leverage = config.lvrg_set.target_pct / (res_df['long_ep_org'].iloc[ep_j] / out_ - 1 - (fee + config.trader_set.market_fee))
            elif strat_version in ['v5_2', 'v7_3']:
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


def sync_check(df_, config, order_side="OPEN", div_rec_row=True):
    try:
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
        # --------- slicing (in trader phase only) --------- #
        #    --> latency 영향도가 높은 곳은 이곳
        #   recursive 가 아닌 indi. 의 latency 를 고려한 if cond.
        if div_rec_row:
            df, df_5T, df_15T, df_30T, df_4H = [df_s.iloc[-row_list[row_idx]:].copy() for row_idx, df_s in enumerate(htf_df_list)]
            rec_df, rec_df_5T, rec_df_15T, rec_df_30T, rec_df_4H = [df_s.iloc[-rec_row_list[row_idx]:].copy() for row_idx, df_s in enumerate(htf_df_list)]
        else:
            df, df_5T, df_15T, df_30T, df_4H = htf_df_list
            rec_df, rec_df_5T, rec_df_15T, rec_df_30T, rec_df_4H = htf_df_list

        # --------- add indi. --------- #

        #        1. 필요한 indi. 는 enlist_epouttp & mr_check 보면서 삽입
        #        2. min use_rows 계산을 위해서, tf 별로 gathering 함        #

    except Exception as e:
        sys_log.error("error in sync_check :", e)
    else:
        return df


def public_indi(res_df, config, np_timeidx, order_side="OPEN"):
    #       Todo        #
    #        1. future data warning => annot. lb_period
    try:
        if order_side in ["OPEN"]:
            h_c_itv_list = ['15T']
            _ = [h_candle_v2(res_df, h_c_itv) for h_c_itv in h_c_itv_list]

            res_df = add_hscores_nb(res_df, h_c_itv_list)   # Todo - lb_period 15
            res_df = add_hrel_habs(res_df, '15T')   # Todo - lb_period 120

            # ------ data pre_proc ------ #
            h_c_itv = '15T'
            divider = 10    # Todo - currently, just one public_divider allowed
            res_df[score_cols(h_c_itv)] = res_df[score_cols(h_c_itv)] // divider
            res_df[h_ratio_cols(h_c_itv)] = res_df[h_ratio_cols(h_c_itv)] // divider

    except Exception as e:
        sys_log.error("error in public_indi :", e)

    return res_df


def ep_loc(res_df, config, i, np_timeidx, show_detail=True, ep_loc_side=OrderSide.SELL):
    # ------- param init ------- #
    open_side = None
    zone = 'n'
    strat_version = config.strat_version

    #        1. just copy & paste const as string
    #           a. ex) const_ = a < b -> "a < b"
    #       Todo : const warning msg manual        #
    #        2. 이곳에서는 앞으로, feature_arr 형식으로 접근할 것 - point_wise
    try:
        # ------- candle_score ------- #
        h_c_itv = '15T'
        ep_loc_data = res_df[score_cols(h_c_itv)].to_numpy()[[i]]   # .shape[-1] 을 위해 [[i]] 형식 채택함
        fws, bs, _ = np.split(ep_loc_data, ep_loc_data.shape[-1], axis=1)

        ep_loc_data = res_df[score_cols(h_c_itv)].to_numpy()[[i - to_itvnum(h_c_itv)]]
        fws2, bs2, _ = np.split(ep_loc_data, ep_loc_data.shape[-1], axis=1)

        ep_loc_data = res_df[h_ratio_cols(h_c_itv)].to_numpy()[[i]]
        hrel, habs = np.split(ep_loc_data, ep_loc_data.shape[-1], axis=1)

        feature_arr = np.array([fws, bs, fws2, bs2, hrel, habs]).reshape(-1,)
        # feature_arr = np.array([-10, -1, 9, 9, 9, 4])
        sys_log.warning("[fws, bs, fws2, bs2, hrel, habs] : {}".format(feature_arr))

        if ep_loc_side == OrderSide.SELL:
            load_arr = load_bin(config.loc_set.zone.short_bin_path)
        else:
            load_arr = load_bin(config.loc_set.zone.long_bin_path)
        assert len(load_arr.shape) == 2, "assert len(load_arr.shape) == 2"

        if any(np.equal(feature_arr, load_arr).all(axis=1)):
            open_side = ep_loc_side

    except Exception as e:
        sys_log.error("error in ep_loc :", e)

    return res_df, open_side, zone


def ep_loc_point2(res_df, config, i, out_j, allow_ep_in, side=OrderSide.SELL):
    try:
        if side == OrderSide.SELL:
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
        else:
            if config.strat_version == 'v5_2' and allow_ep_in == 0:
                if (res_df['dc_lower_1m'].iloc[i - 1] >= res_df['dc_lower_15m'].iloc[i]) & \
                        (res_df['dc_lower_15m'].iloc[i - 1] != res_df['dc_lower_15m'].iloc[i]):
                    pass
                else:
                    return 0, out_j  # input out_j could be initial_i

    except Exception as e:
        sys_log.error("error in ep_loc_point2 :", e)

    return 1, out_j  # allow_ep_in, out_j (for idep)


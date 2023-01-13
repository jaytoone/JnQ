import pandas as pd
import numpy as np
import logging

def enlist_tr(res_df, config, np_timeidx, mode='OPEN', show_detail=True):
    # ================== enlist wave_unit ================== #
    selection_id = config.selection_id

    len_df = len(res_df)
    len_df_range = np.arange(len_df)

    # if config.tr_set.check_hlm == 2:  # 동일한 param 으로도 p2_hlm 시도를 충분히 할 수 있음 (csdbox 와 같은)
    #   assert not (wave_itv1 == wave_itv2 and wave_period1 == wave_period2)

    # ------------ get wave_features ------------ #
    wave_itv1 = config.tr_set.wave_itv1
    wave_period1 = config.tr_set.wave_period1
    wave_itv2 = config.tr_set.wave_itv2
    wave_period2 = config.tr_set.wave_period2
    roll_hl_cnt = 3

    roll_highs1 = [res_df['wave_high_fill_{}{}_-{}'.format(wave_itv1, wave_period1, cnt_ + 1)].to_numpy() for cnt_ in reversed(range(roll_hl_cnt))]
    roll_lows1 = [res_df['wave_low_fill_{}{}_-{}'.format(wave_itv1, wave_period1, cnt_ + 1)].to_numpy() for cnt_ in reversed(range(roll_hl_cnt))]

    wave_high_fill1_ = res_df['wave_high_fill_{}{}'.format(wave_itv1, wave_period1)].to_numpy()
    wave_low_fill1_ = res_df['wave_low_fill_{}{}'.format(wave_itv1, wave_period1)].to_numpy()

    roll_highs2 = [res_df['wave_high_fill_{}{}_-{}'.format(wave_itv2, wave_period2, cnt_ + 1)].to_numpy() for cnt_ in reversed(range(roll_hl_cnt))]
    roll_lows2 = [res_df['wave_low_fill_{}{}_-{}'.format(wave_itv2, wave_period2, cnt_ + 1)].to_numpy() for cnt_ in reversed(range(roll_hl_cnt))]

    wave_high_fill2_ = res_df['wave_high_fill_{}{}'.format(wave_itv2, wave_period2)].to_numpy()
    wave_low_fill2_ = res_df['wave_low_fill_{}{}'.format(wave_itv2, wave_period2)].to_numpy()

    # res_df['short_wave_spread_fill_{}{}'.format(wave_itv1, wave_period1)] = roll_highs1[-1] / wave_low_fill1_
    # res_df['long_wave_spread_fill_{}{}'.format(wave_itv1, wave_period1)] = wave_high_fill1_ / roll_lows1[-1]
    res_df['short_wave_spread_fill_{}{}'.format(wave_itv1, wave_period1)] = wave_high_fill1_ / wave_low_fill1_
    res_df['long_wave_spread_fill_{}{}'.format(wave_itv1, wave_period1)] = wave_high_fill1_ / wave_low_fill1_

    # ------------ enlist tr_unit ------------ #
    # cu's roll_high_[:, -1] = prev_high & cu's roll_low_[:, -1] = current_low
    # co's roll_low_[:, -1] = prev_low & co's roll_high_[:, -1] = current_high
    res_df['short_tp_1_{}'.format(selection_id)] = wave_low_fill1_  # wave_low_fill_ b2_low_5T
    res_df['short_tp_0_{}'.format(selection_id)] = roll_highs1[-1]  # roll_high_[:, -2] wave_high_fill_
    res_df['long_tp_1_{}'.format(selection_id)] = wave_high_fill1_  # wave_high_fill_ b2_high_5T
    res_df['long_tp_0_{}'.format(selection_id)] = roll_lows1[-1]  # roll_low_[:, -2]  wave_low_fill_

    res_df['short_ep1_1_{}'.format(selection_id)] = wave_low_fill1_  # b2_low_5T
    res_df['short_ep1_0_{}'.format(selection_id)] = wave_high_fill1_  # roll_high_[:, -2]
    res_df['long_ep1_1_{}'.format(selection_id)] = wave_high_fill1_  # b2_high_5T
    res_df['long_ep1_0_{}'.format(selection_id)] = wave_low_fill1_  # roll_low_[:, -2]

    res_df['short_out_1_{}'.format(selection_id)] = wave_low_fill2_  # wave_low_fill1_   # wave_low_fill2_   # b2_low_5T
    res_df['short_out_0_{}'.format(selection_id)] = roll_highs2[-1]  # roll_highs1[-1]   # roll_highs2[-1]  # roll_high_[:, -2]
    res_df['long_out_1_{}'.format(selection_id)] = wave_high_fill2_  # wave_high_fill1_   # wave_high_fill2_   # b2_high_5T
    res_df['long_out_0_{}'.format(selection_id)] = roll_lows2[-1]  # roll_lows1[-1]   # roll_lows2[-1]    # roll_low_[:, -2]

    res_df['short_ep2_1_{}'.format(selection_id)] = wave_low_fill2_  # b2_low_5T
    res_df['short_ep2_0_{}'.format(selection_id)] = wave_high_fill2_  # roll_high_[:, -2]
    res_df['long_ep2_1_{}'.format(selection_id)] = wave_high_fill2_  # b2_high_5T
    res_df['long_ep2_0_{}'.format(selection_id)] = wave_low_fill2_  # roll_low_[:, -2]

    # --- inversion --- #
    if config.pos_set.short_inversion or config.pos_set.long_inversion:
        res_df.rename({short_tp_1_: long_tp_1_, long_tp_1_: short_tp_1_}, axis=1, inplace=True)
        res_df.rename({short_tp_0_: long_tp_0_, long_tp_0_: short_tp_0_}, axis=1, inplace=True)
        res_df.rename({short_epout_1_: long_epout_1_, long_epout_1_: short_epout_1_}, axis=1, inplace=True)
        res_df.rename({short_epout_0_: long_epout_0_, long_epout_0_: short_epout_0_}, axis=1, inplace=True)

    res_df['short_tp_gap_{}'.format(selection_id)] = abs(res_df['short_tp_1_{}'.format(selection_id)] - res_df['short_tp_0_{}'.format(selection_id)])
    res_df['long_tp_gap_{}'.format(selection_id)] = abs(res_df['long_tp_1_{}'.format(selection_id)] - res_df['long_tp_0_{}'.format(selection_id)])
    res_df['short_ep1_gap_{}'.format(selection_id)] = abs(
        res_df['short_ep1_1_{}'.format(selection_id)] - res_df['short_ep1_0_{}'.format(selection_id)])
    res_df['long_ep1_gap_{}'.format(selection_id)] = abs(res_df['long_ep1_1_{}'.format(selection_id)] - res_df['long_ep1_0_{}'.format(selection_id)])

    res_df['short_out_gap_{}'.format(selection_id)] = abs(
        res_df['short_out_1_{}'.format(selection_id)] - res_df['short_out_0_{}'.format(selection_id)])
    res_df['long_out_gap_{}'.format(selection_id)] = abs(res_df['long_out_1_{}'.format(selection_id)] - res_df['long_out_0_{}'.format(selection_id)])
    res_df['short_ep2_gap_{}'.format(selection_id)] = abs(
        res_df['short_ep2_1_{}'.format(selection_id)] - res_df['short_ep2_0_{}'.format(selection_id)])
    res_df['long_ep2_gap_{}'.format(selection_id)] = abs(res_df['long_ep2_1_{}'.format(selection_id)] - res_df['long_ep2_0_{}'.format(selection_id)])

    data_cols = ['open', 'high', 'low', 'close']  # Todo - public_indi 이전에 해야할지도 모름 # 'close', 'haopen', 'hahigh', 'halow', 'haclose'
    open, high, low, close = [res_df[col_].to_numpy() for col_ in data_cols]

    # ================== point ================== #
    short_open_res1 = np.ones(len_df)  # .astype(object)
    long_open_res1 = np.ones(len_df)  # .astype(object)
    short_open_res2 = np.ones(len_df)  # .astype(object)
    long_open_res2 = np.ones(len_df)  # .astype(object)

    # ------------ wave_point ------------ #
    notnan_cu = ~pd.isnull(res_df['wave_cu_{}{}'.format(wave_itv1, wave_period1)].to_numpy())  # isnull for object
    notnan_co = ~pd.isnull(res_df['wave_co_{}{}'.format(wave_itv1, wave_period1)].to_numpy())

    short_open_res1 *= res_df['wave_cu_{}{}'.format(wave_itv1, wave_period1)].to_numpy().astype(bool) * notnan_cu  # object로 변환되는 경우에 대응해, bool 로 재정의
    long_open_res1 *= res_df['wave_co_{}{}'.format(wave_itv1, wave_period1)].to_numpy().astype(bool) * notnan_co  # np.nan = bool type 으로 True 임..
    short_open_res2 *= res_df['wave_cu_{}{}'.format(wave_itv2, wave_period2)].to_numpy().astype(bool) * notnan_cu
    long_open_res2 *= res_df['wave_co_{}{}'.format(wave_itv2, wave_period2)].to_numpy().astype(bool) * notnan_co

    if show_detail:
        sys_log.warning("wave_point")
        sys_log.warning("np.sum(short_open_res1 == 1) : {}".format(np.sum(short_open_res1 == 1)))
        sys_log.warning("np.sum(long_open_res1 == 1) : {}".format(np.sum(long_open_res1 == 1)))
        sys_log.warning("np.sum(short_open_res2 == 1) : {}".format(np.sum(short_open_res2 == 1)))
        sys_log.warning("np.sum(long_open_res2 == 1) : {}".format(np.sum(long_open_res2 == 1)))

        # ------ reject update_hl ------ #
    notnan_update_low_cu = ~pd.isnull(res_df['wave_update_low_cu_bool_{}{}'.format(wave_itv1, wave_period1)].to_numpy())
    notnan_update_high_co = ~pd.isnull(res_df['wave_update_high_co_bool_{}{}'.format(wave_itv1, wave_period1)].to_numpy())

    short_open_res1 *= ~(res_df['wave_update_low_cu_bool_{}{}'.format(wave_itv1, wave_period1)].to_numpy().astype(bool)) * notnan_update_low_cu
    long_open_res1 *= ~(res_df['wave_update_high_co_bool_{}{}'.format(wave_itv1, wave_period1)].to_numpy().astype(bool)) * notnan_update_high_co
    short_open_res2 *= ~(res_df['wave_update_low_cu_bool_{}{}'.format(wave_itv2, wave_period2)].to_numpy().astype(bool)) * notnan_update_low_cu
    long_open_res2 *= ~(res_df['wave_update_high_co_bool_{}{}'.format(wave_itv2, wave_period2)].to_numpy().astype(bool)) * notnan_update_high_co

    # short_open_res1 *= ~res_df['wave_update_low_cu_bool_{}{}'.format(wave_itv1, wave_period1)].to_numpy()
    # long_open_res1 *= ~res_df['wave_update_high_co_bool_{}{}'.format(wave_itv1, wave_period1)].to_numpy()
    # short_open_res2 *= ~res_df['wave_update_low_cu_bool_{}{}'.format(wave_itv2, wave_period2)].to_numpy()
    # long_open_res2 *= ~res_df['wave_update_high_co_bool_{}{}'.format(wave_itv2, wave_period2)].to_numpy()

    if show_detail:
        sys_log.warning("reject update_hl")
        sys_log.warning("np.sum(short_open_res1 == 1) : {}".format(np.sum(short_open_res1 == 1)))
        sys_log.warning("np.sum(long_open_res1 == 1) : {}".format(np.sum(long_open_res1 == 1)))
        sys_log.warning("np.sum(short_open_res2 == 1) : {}".format(np.sum(short_open_res2 == 1)))
        sys_log.warning("np.sum(long_open_res2 == 1) : {}".format(np.sum(long_open_res2 == 1)))

        # ------ wave_itv ------ #
    if wave_itv1 != 'T':
        wave_itv1_num = to_itvnum(wave_itv1)
        short_open_res1 *= np_timeidx % wave_itv1_num == (wave_itv1_num - 1)
        long_open_res1 *= np_timeidx % wave_itv1_num == (wave_itv1_num - 1)

        if show_detail:
            sys_log.warning("wave_itv1")
            sys_log.warning("np.sum(short_open_res1 == 1) : {}".format(np.sum(short_open_res1 == 1)))
            sys_log.warning("np.sum(long_open_res1 == 1) : {}".format(np.sum(long_open_res1 == 1)))

    if wave_itv2 != 'T':
        wave_itv2_num = to_itvnum(wave_itv2)
        short_open_res2 *= np_timeidx % wave_itv2_num == (wave_itv2_num - 1)
        long_open_res2 *= np_timeidx % wave_itv2_num == (wave_itv2_num - 1)

        if show_detail:
            sys_log.warning("wave_itv2")
            sys_log.warning("np.sum(short_open_res2 == 1) : {}".format(np.sum(short_open_res2 == 1)))
            sys_log.warning("np.sum(long_open_res2 == 1) : {}".format(np.sum(long_open_res2 == 1)))

    # ------ wave_mm ------ #
    wave_high_terms_cnt_fill1_ = res_df['wave_high_terms_cnt_fill_{}{}'.format(wave_itv1, wave_period1)].to_numpy()
    wave_low_terms_cnt_fill1_ = res_df['wave_low_terms_cnt_fill_{}{}'.format(wave_itv1, wave_period1)].to_numpy()

    short_open_res1 *= (wave_high_terms_cnt_fill1_ > config.tr_set.wave_greater2) & (wave_low_terms_cnt_fill1_ > config.tr_set.wave_greater1)
    long_open_res1 *= (wave_low_terms_cnt_fill1_ > config.tr_set.wave_greater2) & (wave_high_terms_cnt_fill1_ > config.tr_set.wave_greater1)

    wave_high_terms_cnt_fill2_ = res_df['wave_high_terms_cnt_fill_{}{}'.format(wave_itv2, wave_period2)].to_numpy()
    wave_low_terms_cnt_fill2_ = res_df['wave_low_terms_cnt_fill_{}{}'.format(wave_itv2, wave_period2)].to_numpy()

    short_open_res2 *= (wave_high_terms_cnt_fill2_ > config.tr_set.wave_greater2) & (wave_low_terms_cnt_fill2_ > config.tr_set.wave_greater1)
    long_open_res2 *= (wave_low_terms_cnt_fill2_ > config.tr_set.wave_greater2) & (wave_high_terms_cnt_fill2_ > config.tr_set.wave_greater1)

    if show_detail:
        sys_log.warning("wave_mm")
        sys_log.warning("np.sum(short_open_res1 == 1) : {}".format(np.sum(short_open_res1 == 1)))
        sys_log.warning("np.sum(long_open_res1 == 1) : {}".format(np.sum(long_open_res1 == 1)))
        sys_log.warning("np.sum(short_open_res2 == 1) : {}".format(np.sum(short_open_res2 == 1)))
        sys_log.warning("np.sum(long_open_res2 == 1) : {}".format(np.sum(long_open_res2 == 1)))

    # ------ wave_length ------ #
    if config.tr_set.wave_length1 != "None":
        short_wave_length_fill_ = res_df['short_wave_length_fill_{}{}'.format(wave_itv1, wave_period1)].to_numpy()
        long_wave_length_fill_ = res_df['long_wave_length_fill_{}{}'.format(wave_itv1, wave_period1)].to_numpy()

        short_open_res1 *= short_wave_length_fill_ >= config.tr_set.wave_length1
        long_open_res1 *= long_wave_length_fill_ >= config.tr_set.wave_length1

        if show_detail:
            sys_log.warning("wave_length")
            sys_log.warning("np.sum(short_open_res1 == 1) : {}".format(np.sum(short_open_res1 == 1)))
            sys_log.warning("np.sum(long_open_res1 == 1) : {}".format(np.sum(long_open_res1 == 1)))

    # ------ wave_spread ------ #
    if config.tr_set.wave_spread1 != "None":
        short_wave_spread_fill = res_df['short_wave_spread_fill_{}{}'.format(wave_itv1, wave_period1)].to_numpy()
        long_wave_spread_fill = res_df['long_wave_spread_fill_{}{}'.format(wave_itv1, wave_period1)].to_numpy()

        short_open_res1 *= short_wave_spread_fill >= config.tr_set.wave_spread1
        long_open_res1 *= long_wave_spread_fill >= config.tr_set.wave_spread1

        if show_detail:
            sys_log.warning("wave_spread")
            sys_log.warning("np.sum(short_open_res1 == 1) : {}".format(np.sum(short_open_res1 == 1)))
            sys_log.warning("np.sum(long_open_res1 == 1) : {}".format(np.sum(long_open_res1 == 1)))

            # ------------ 추세선 리스트 ------------ #
    # ------ ma & prev_low ------ #
    # ma_period = 100

    # short_prev_ma_ = res_df['short_ma_T{}_-1'.format(ma_period)].to_numpy()
    # long_prev_ma_ = res_df['long_ma_T{}_-1'.format(ma_period)].to_numpy()

    # short_open_res1 *= short_prev_ma_ > roll_highs1[-1]  # Todo, index sync. 요망
    # long_open_res1 *= long_prev_ma_ < roll_lows1[-1]

    # if show_detail:
    #   sys_log.warning("ma & prev_low")
    #   sys_log.warning("np.sum(short_open_res1 == 1) : {}".format(np.sum(short_open_res1 == 1)))
    #   sys_log.warning("np.sum(long_open_res1 == 1) : {}".format(np.sum(long_open_res1 == 1)))

    # ================== tr_set ================== #
    # ------------ tp ------------ #
    tpg = config.tr_set.tp_gap
    res_df['short_tp_{}'.format(selection_id)] = res_df['short_tp_1_{}'.format(selection_id)].to_numpy() - res_df[
        'short_tp_gap_{}'.format(selection_id)].to_numpy() * tpg
    res_df['long_tp_{}'.format(selection_id)] = res_df['long_tp_1_{}'.format(selection_id)].to_numpy() + res_df[
        'long_tp_gap_{}'.format(selection_id)].to_numpy() * tpg
    # res_df['short_tp_{}'.format(selection_id)] = short_tp_1 - short_epout_gap * tpg
    # res_df['long_tp_{}'.format(selection_id)] = long_tp_1 + long_epout_gap * tpg

    # ------ limit_ep ------ #
    if config.ep_set.entry_type == "LIMIT":
        epg1 = config.tr_set.ep_gap1
        epg2 = config.tr_set.ep_gap2

        # ------ epbox as epg ------ #
        res_df['short_ep1_{}'.format(selection_id)] = res_df['short_ep1_0_{}'.format(selection_id)].to_numpy() + res_df[
            'short_ep1_gap_{}'.format(selection_id)].to_numpy() * epg1
        res_df['long_ep1_{}'.format(selection_id)] = res_df['long_ep1_0_{}'.format(selection_id)].to_numpy() - res_df[
            'long_ep1_gap_{}'.format(selection_id)].to_numpy() * epg1
        res_df['short_ep2_{}'.format(selection_id)] = res_df['short_ep2_0_{}'.format(selection_id)].to_numpy() + res_df[
            'short_ep2_gap_{}'.format(selection_id)].to_numpy() * epg2
        res_df['long_ep2_{}'.format(selection_id)] = res_df['long_ep2_0_{}'.format(selection_id)].to_numpy() - res_df[
            'long_ep2_gap_{}'.format(selection_id)].to_numpy() * epg2

        # ------ tpbox as epg ------ #
        # p1_hlm 을 위해선, tp_0 를 기준할 수 없음 --> ep1 & ep2 를 기준으로 진행
        # res_df['short_ep1_{}'.format(selection_id)] = res_df['short_ep1_0_{}'.format(selection_id)].to_numpy() + res_df['short_tp_gap_{}'.format(selection_id)].to_numpy() * epg1  # fibonacci 고려하면, tp / out gap 기준이 맞지 않을까
        # res_df['long_ep1_{}'.format(selection_id)] = res_df['long_ep1_0_{}'.format(selection_id)].to_numpy() - res_df['long_tp_gap_{}'.format(selection_id)].to_numpy() * epg1
        # res_df['short_ep2_{}'.format(selection_id)] = res_df['short_ep2_0_{}'.format(selection_id)].to_numpy() + res_df['short_out_gap_{}'.format(selection_id)].to_numpy() * epg2
        # res_df['long_ep2_{}'.format(selection_id)] = res_df['long_ep2_0_{}'.format(selection_id)].to_numpy() - res_df['long_out_gap_{}'.format(selection_id)].to_numpy() * epg2

        # ------ fibo_ep ------ #
        # res_df['short_ep1_{}'.format(selection_id)] = res_df['short_tp_0_{}'.format(selection_id)].to_numpy() + res_df['short_tp_gap_{}'.format(selection_id)].to_numpy() * epg1  # fibonacci 고려하면, tp / out gap 기준이 맞지 않을까
        # res_df['long_ep1_{}'.format(selection_id)] = res_df['long_tp_0_{}'.format(selection_id)].to_numpy() - res_df['long_tp_gap_{}'.format(selection_id)].to_numpy() * epg1

    # ------ market_ep ------ #
    else:
        res_df['short_ep1_{}'.format(selection_id)] = close
        res_df['long_ep1_{}'.format(selection_id)] = close
        res_df['short_ep2_{}'.format(selection_id)] = close
        res_df['long_ep2_{}'.format(selection_id)] = close

    # ------------ out ------------ #
    outg = config.tr_set.out_gap
    # res_df['short_out_{}'.format(selection_id)] = short_tp_0 + short_tp_gap * outg            # 1. for hhm check -> 규칙성과 wave_range 기반 거래 기준의 hhm 확인
    # res_df['long_out_{}'.format(selection_id)] = long_tp_0 - long_tp_gap * outg

    if config.tr_set.check_hlm in [0, 2]:  # for p1_hhm, p2_hlm
        res_df['short_out_{}'.format(selection_id)] = res_df['short_out_0_{}'.format(selection_id)].to_numpy() + res_df[
            'short_out_gap_{}'.format(selection_id)].to_numpy() * outg
        res_df['long_out_{}'.format(selection_id)] = res_df['long_out_0_{}'.format(selection_id)].to_numpy() - res_df[
            'long_out_gap_{}'.format(selection_id)].to_numpy() * outg
        # res_df['short_out_{}'.format(selection_id)] = res_df['short_ep2_0_{}'.format(selection_id)].to_numpy() + res_df['short_ep2_gap_{}'.format(selection_id)].to_numpy() * outg   # p2's ep_box 를 out 으로 사용한다?
        # res_df['long_out_{}'.format(selection_id)] = res_df['long_ep2_0_{}'.format(selection_id)].to_numpy() - res_df['long_ep2_gap_{}'.format(selection_id)].to_numpy() * outg

    else:  # for p1_hlm
        # ------ irregular - next_fibo ------ #
        co_wrr_32_ = res_df['co_wrr_32_{}{}'.format(wave_itv1, wave_period1)].to_numpy()
        cu_wrr_32_ = res_df['cu_wrr_32_{}{}'.format(wave_itv1, wave_period1)].to_numpy()

        short_next_fibo_outg = np.vectorize(get_next_fibo_gap)(cu_wrr_32_)
        long_next_fibo_outg = np.vectorize(get_next_fibo_gap)(co_wrr_32_)

        res_df['short_out_{}'.format(selection_id)] = res_df['short_tp_0_{}'.format(selection_id)].to_numpy() + res_df[
            'short_tp_gap_{}'.format(selection_id)].to_numpy() * short_next_fibo_outg  # ep 와 마찬가지로, tpg 기준 가능
        res_df['long_out_{}'.format(selection_id)] = res_df['long_tp_0_{}'.format(selection_id)].to_numpy() - res_df[
            'long_tp_gap_{}'.format(selection_id)].to_numpy() * long_next_fibo_outg

        # ------ ep1box as outg ------ #
        # res_df['short_out_{}'.format(selection_id)] = res_df['short_ep1_0_{}'.format(selection_id)].to_numpy() + res_df['short_ep1_gap_{}'.format(selection_id)].to_numpy() * outg
        # res_df['long_out_{}'.format(selection_id)] = res_df['long_ep1_0_{}'.format(selection_id)].to_numpy() - res_df['long_ep1_gap_{}'.format(selection_id)].to_numpy() * outg

        # ------ ep1box as outg for 1_tr ------ #
        # res_df['short_out_{}'.format(selection_id)] = res_df['short_ep1_0_{}'.format(selection_id)].to_numpy() + res_df['short_ep1_gap_{}'.format(selection_id)].to_numpy() * outg
        # res_df['long_out_{}'.format(selection_id)] = res_df['long_ep1_0_{}'.format(selection_id)].to_numpy() - res_df['long_ep1_gap_{}'.format(selection_id)].to_numpy() * outg

        # ------ 1_tr - auto_calculation ------ #
        # res_df['short_out_{}'.format(selection_id)] = res_df['short_ep1_{}'.format(selection_id)] + (res_df['short_ep1_{}'.format(selection_id)].to_numpy() - res_df['short_tp_{}'.format(selection_id)].to_numpy())
        # res_df['long_out_{}'.format(selection_id)] = res_df['long_ep1_{}'.format(selection_id)].to_numpy() - (res_df['long_tp_{}'.format(selection_id)].to_numpy() - res_df['long_ep1_{}'.format(selection_id)].to_numpy())

        # ------ tpbox as outg ------ #
        # res_df['short_out_{}'.format(selection_id)] = res_df['short_ep1_0_{}'.format(selection_id)].to_numpy() + res_df['short_tp_gap_{}'.format(selection_id)].to_numpy() * outg  # ep 와 마찬가지로, tpg 기준 가능
        # res_df['long_out_{}'.format(selection_id)] = res_df['long_ep1_0_{}'.format(selection_id)].to_numpy() - res_df['long_tp_gap_{}'.format(selection_id)].to_numpy() * outg

        # ------ fibo_out ------ #
        # res_df['short_out_{}'.format(selection_id)] = res_df['short_tp_0_{}'.format(selection_id)].to_numpy() + res_df['short_tp_gap_{}'.format(selection_id)].to_numpy() * outg  # ep 와 마찬가지로, tpg 기준 가능
        # res_df['long_out_{}'.format(selection_id)] = res_df['long_tp_0_{}'.format(selection_id)].to_numpy() - res_df['long_tp_gap_{}'.format(selection_id)].to_numpy() * outg

    # ------------ point validation ------------ # - vecto. 로 미리 거를 수 있는걸 거르면 좋을 것
    short_tp_ = res_df['short_tp_{}'.format(selection_id)].to_numpy()
    short_ep1_ = res_df['short_ep1_{}'.format(selection_id)].to_numpy()
    short_ep2_ = res_df['short_ep2_{}'.format(selection_id)].to_numpy()
    short_out_ = res_df['short_out_{}'.format(selection_id)].to_numpy()

    long_tp_ = res_df['long_tp_{}'.format(selection_id)].to_numpy()
    long_ep1_ = res_df['long_ep1_{}'.format(selection_id)].to_numpy()
    long_ep2_ = res_df['long_ep2_{}'.format(selection_id)].to_numpy()
    long_out_ = res_df['long_out_{}'.format(selection_id)].to_numpy()

    short_open_res1 *= (
                short_tp_ < short_ep1_)  # (short_ep_ < short_out_)  # tr_set validation reject nan data & 정상 거래 위한 tp > ep / --> p2_box location (cannot be vectorized)
    # short_open_res1 *= close < short_ep1_   # reject entry open_execution
    short_open_res1 *= close < res_df[
        'short_ep1_0_{}'.format(selection_id)].to_numpy()  # reject hl_out open_execution -> close always < ep1_0 at wave_p1
    # short_out_  res_df['short_tp_0_{}'.format(selection_id)].to_numpy() res_df['short_ep1_0_{}'.format(selection_id)].to_numpy()

    long_open_res1 *= (long_tp_ > long_ep1_)  # (long_ep_ > long_out_)  # (long_tp_ > long_ep_) # tr_set validation
    # long_open_res1 *= close > long_ep1_  # reject entry open_execution
    long_open_res1 *= close > res_df['long_ep1_0_{}'.format(selection_id)].to_numpy()  # reject hl_out open_execution
    # long_out_ res_df['long_tp_0_{}'.format(selection_id)].to_numpy() res_df['long_ep1_0_{}'.format(selection_id)].to_numpy()

    short_open_res2 *= (short_ep2_ < short_out_)  # tr_set validation (short_tp_ < short_ep_) # --> p2_box location (cannot be vectorized)
    short_open_res2 *= close < short_out_  # reject hl_out open_execution

    long_open_res2 *= (long_ep2_ > long_out_)  # tr_set validation (long_tp_ > long_ep_) &   # p2's ep & out can be vectorized
    long_open_res2 *= close > long_out_  # reject hl_out open_execution

    res_df['short_open1_{}'.format(selection_id)] = short_open_res1 * (not config.pos_set.short_ban)
    res_df['long_open1_{}'.format(selection_id)] = long_open_res1 * (not config.pos_set.long_ban)
    # print("res_df['long_open1_{}'.format(selection_id)].to_numpy() :", res_df['long_open1_{}'.format(selection_id)].to_numpy())
    res_df['short_open2_{}'.format(selection_id)] = short_open_res2
    res_df['long_open2_{}'.format(selection_id)] = long_open_res2

    if show_detail:
        sys_log.warning("point validation")
        sys_log.warning("np.sum(short_open_res1 == 1) : {}".format(np.sum(short_open_res1 == 1)))
        sys_log.warning("np.sum(long_open_res1 == 1) : {}".format(np.sum(long_open_res1 == 1)))
        sys_log.warning("np.sum(short_open_res2 == 1) : {}".format(np.sum(short_open_res2 == 1)))
        sys_log.warning("np.sum(long_open_res2 == 1) : {}".format(np.sum(long_open_res2 == 1)))

    # ------------ higher_high momentum ------------ #
    # wb_tpg = config.tr_set.wb_tp_gap
    # wb_outg = config.tr_set.wb_out_gap
    # res_df['short_wave_1_{}'.format(selection_id)] = res_df['short_tp_1_{}'.format(selection_id)].to_numpy() - res_df['short_tp_gap_{}'.format(selection_id)].to_numpy() * wb_tpg
    # res_df['long_wave_1_{}'.format(selection_id)] = res_df['long_tp_1_{}'.format(selection_id)].to_numpy() + res_df['long_tp_gap_{}'.format(selection_id)].to_numpy() * wb_tpg

    # res_df['short_wave_0_{}'.format(selection_id)] = res_df['short_tp_0_{}'.format(selection_id)].to_numpy() + res_df['short_tp_gap_{}'.format(selection_id)].to_numpy() * wb_outg        # hhm check
    # res_df['long_wave_0_{}'.format(selection_id)] = res_df['long_tp_0_{}'.format(selection_id)].to_numpy() - res_df['long_tp_gap_{}'.format(selection_id)].to_numpy() * wb_outg
    # res_df['short_wave_0_{}'.format(selection_id)] = short_epout_0 + short_epout_gap * wb_outg
    # res_df['long_wave_0_{}'.format(selection_id)] = long_epout_0 - long_epout_gap * wb_outg

    # ------ tr ------ #
    if not config.tr_set.check_hlm:
        res_df['short_tr_{}'.format(selection_id)] = abs(
            (short_ep1_ / short_tp_ - config.trader_set.limit_fee - 1) / (short_ep1_ / short_out_ - config.trader_set.market_fee - 1))
        res_df['long_tr_{}'.format(selection_id)] = abs(
            (long_tp_ / long_ep1_ - config.trader_set.limit_fee - 1) / (long_out_ / long_ep1_ - config.trader_set.market_fee - 1))
    else:
        res_df['short_tr_{}'.format(selection_id)] = np.nan
        res_df['long_tr_{}'.format(selection_id)] = np.nan

    # ------ zoned_ep ------ #
    # if config.tr_set.c_ep_gap != "None":
    #     # res_df['short_ep_org_{}'.format(selection_id)] = res_df['short_ep_{}'.format(selection_id)].copy()
    #     # res_df['long_ep_org_{}'.format(selection_id)] = res_df['long_ep_{}'.format(selection_id)].copy()
    #     res_df['short_ep2_{}'.format(selection_id)] = short_epout_1 + short_epout_gap * config.tr_set.c_ep_gap
    #     res_df['long_ep2_{}'.format(selection_id)] = long_epout_1 - long_epout_gap * config.tr_set.c_ep_gap

    # # ------ zoned_out ------ #
    # if config.tr_set.t_out_gap != "None":
    #     # res_df['short_out_org_{}'.format(selection_id)] = res_df['short_out_{}'.format(selection_id)].copy()
    #     # res_df['long_out_org_{}'.format(selection_id)] = res_df['long_out_{}'.format(selection_id)].copy()
    #     res_df['short_out2_{}'.format(selection_id)] = short_epout_0 + short_epout_gap * config.tr_set.t_out_gap
    #     res_df['long_out2_{}'.format(selection_id)] = long_epout_0 - long_epout_gap * config.tr_set.t_out_gap

    return res_df

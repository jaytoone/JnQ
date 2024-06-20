import pandas as pd
import numpy as np
import logging
from funcs.public.broker import itv_to_number

sys_log = logging.getLogger()


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
    tc_period = config.tr_set.tc_period
    roll_hl_cnt = 3

    roll_highs1 = [res_df['wave_high_fill_{}{}_-{}'.format(wave_itv1, wave_period1, cnt_ + 1)].to_numpy() for cnt_ in reversed(range(roll_hl_cnt))]
    roll_lows1 = [res_df['wave_low_fill_{}{}_-{}'.format(wave_itv1, wave_period1, cnt_ + 1)].to_numpy() for cnt_ in reversed(range(roll_hl_cnt))]

    wave_high_fill1_ = res_df['wave_high_fill_{}{}'.format(wave_itv1, wave_period1)].to_numpy()
    wave_low_fill1_ = res_df['wave_low_fill_{}{}'.format(wave_itv1, wave_period1)].to_numpy()

    #     roll_highs2 = [res_df['wave_high_fill_{}{}_-{}'.format(wave_itv2, wave_period2, cnt_ + 1)].to_numpy() for cnt_ in reversed(range(roll_hl_cnt))]
    #     roll_lows2 = [res_df['wave_low_fill_{}{}_-{}'.format(wave_itv2, wave_period2, cnt_ + 1)].to_numpy() for cnt_ in reversed(range(roll_hl_cnt))]

    #     wave_high_fill2_ = res_df['wave_high_fill_{}{}'.format(wave_itv2, wave_period2)].to_numpy()
    #     wave_low_fill2_ = res_df['wave_low_fill_{}{}'.format(wave_itv2, wave_period2)].to_numpy()

    # res_df['short_wave_spread_fill_{}{}'.format(wave_itv1, wave_period1)] = roll_highs1[-1] / wave_low_fill1_
    # res_df['long_wave_spread_fill_{}{}'.format(wave_itv1, wave_period1)] = wave_high_fill1_ / roll_lows1[-1]
    # res_df['short_wave_spread_fill_{}{}'.format(wave_itv1, wave_period1)] = wave_high_fill1_ / wave_low_fill1_
    # res_df['long_wave_spread_fill_{}{}'.format(wave_itv1, wave_period1)] = wave_high_fill1_ / wave_low_fill1_

    # itvnum = itv_to_number(wave_itv1)
    # itvnum2 = itvnum * 2

    # high_ = res_df['high_{}'.format(wave_itv1)].to_numpy()
    # low_ = res_df['low_{}'.format(wave_itv1)].to_numpy()

    # b1_close_ = res_df['close_{}'.format(wave_itv1)].shift(itvnum).to_numpy()
    # b1_open_ = res_df['open_{}'.format(wave_itv1)].shift(itvnum).to_numpy()
    # b1_high_ = res_df['high_{}'.format(wave_itv1)].shift(itvnum).to_numpy()
    # b1_low_ = res_df['low_{}'.format(wave_itv1)].shift(itvnum).to_numpy()
    # b2_high_ = res_df['high_{}'.format(wave_itv1)].shift(itvnum2).to_numpy()
    # b2_low_ = res_df['low_{}'.format(wave_itv1)].shift(itvnum2).to_numpy()

    # max_high_ = np.maximum(b1_high_, b2_high_)
    # min_low_ = np.minimum(b1_low_, b2_low_)

    # res_df['b1_close_{}'.format(wave_itv1)] = b1_close_
    # res_df['b1_open_{}'.format(wave_itv1)] = b1_open_
    # res_df['b1_high_{}'.format(wave_itv1)] = b1_high_
    # res_df['b1_low_{}'.format(wave_itv1)] = b1_low_
    # res_df['b2_high_{}'.format(wave_itv1)] = b2_high_
    # res_df['b2_low_{}'.format(wave_itv1)] = b2_low_
    # res_df['max_high_{}'.format(wave_itv1)] = max_high_
    # res_df['min_low_{}'.format(wave_itv1)] = min_low_

    # ------------ enlist tr_unit ------------ #
    # cu's roll_high_[:, -1] = prev_high & cu's roll_low_[:, -1] = current_low
    # co's roll_low_[:, -1] = prev_low & co's roll_high_[:, -1] = current_high
    # 2 사용하는 이유 : tp_1 을 p2_box 기준으로 설정하가 위함 --> enex_pairing_v4 function 과 호환되지 않음
    res_df['short_tp_1_{}'.format(selection_id)] = wave_low_fill1_  # wave_low_fill1_ b2_low_5T
    res_df['short_tp_0_{}'.format(selection_id)] = roll_highs1[-1]  # roll_highs1[-1] wave_high_fill1_
    res_df['long_tp_1_{}'.format(selection_id)] = wave_high_fill1_  # wave_high_fill1_ b2_high_5T
    res_df['long_tp_0_{}'.format(selection_id)] = roll_lows1[-1]  # roll_lows1[-1]  wave_low_fill1_

    res_df['short_ep1_1_{}'.format(selection_id)] = wave_low_fill1_  # wave_low_fill1_   # b2_low_5T
    res_df['short_ep1_0_{}'.format(selection_id)] = wave_high_fill1_  # wave_high_fill1_
    res_df['long_ep1_1_{}'.format(selection_id)] = wave_high_fill1_  # wave_high_fill1_   # b2_high_5T
    res_df['long_ep1_0_{}'.format(selection_id)] = wave_low_fill1_  # wave_low_fill1_

    # --> p2's ep use p1's ep
    res_df['short_ep2_1_{}'.format(selection_id)] = wave_low_fill1_  # wave_low_fill2_   # b2_low_5T
    res_df['short_ep2_0_{}'.format(selection_id)] = wave_high_fill1_  # wave_high_fill2_
    res_df['long_ep2_1_{}'.format(selection_id)] = wave_high_fill1_  # wave_high_fill2_   # b2_high_5T
    res_df['long_ep2_0_{}'.format(selection_id)] = wave_low_fill1_  # wave_low_fill2_

    # --> out use p1's low, (allow prev_low as out for p1_hhm only)
    res_df['short_out_1_{}'.format(selection_id)] = wave_low_fill1_  # wave_low_fill1_   # wave_low_fill2_   # b2_low_5T
    res_df['short_out_0_{}'.format(selection_id)] = roll_highs1[-1]  # roll_highs1[-1] if not config.tr_set.check_hlm else wave_high_fill1_   # roll_highs2[-1]  # roll_high_[:, -2]
    res_df['long_out_1_{}'.format(selection_id)] = wave_high_fill1_  # wave_high_fill1_   # wave_high_fill2_   # b2_high_5T
    res_df['long_out_0_{}'.format(selection_id)] = roll_lows1[-1]  # roll_lows1[-1] if not config.tr_set.check_hlm else wave_low_fill1_   # roll_lows2[-1]

    res_df['short_tp_gap_{}'.format(selection_id)] = abs(res_df['short_tp_1_{}'.format(selection_id)] - res_df['short_tp_0_{}'.format(selection_id)])
    res_df['long_tp_gap_{}'.format(selection_id)] = abs(res_df['long_tp_1_{}'.format(selection_id)] - res_df['long_tp_0_{}'.format(selection_id)])
    res_df['short_ep1_gap_{}'.format(selection_id)] = abs(res_df['short_ep1_1_{}'.format(selection_id)] - res_df['short_ep1_0_{}'.format(selection_id)])
    res_df['long_ep1_gap_{}'.format(selection_id)] = abs(res_df['long_ep1_1_{}'.format(selection_id)] - res_df['long_ep1_0_{}'.format(selection_id)])

    res_df['short_out_gap_{}'.format(selection_id)] = abs(res_df['short_out_1_{}'.format(selection_id)] - res_df['short_out_0_{}'.format(selection_id)])
    res_df['long_out_gap_{}'.format(selection_id)] = abs(res_df['long_out_1_{}'.format(selection_id)] - res_df['long_out_0_{}'.format(selection_id)])
    res_df['short_ep2_gap_{}'.format(selection_id)] = abs(res_df['short_ep2_1_{}'.format(selection_id)] - res_df['short_ep2_0_{}'.format(selection_id)])
    res_df['long_ep2_gap_{}'.format(selection_id)] = abs(res_df['long_ep2_1_{}'.format(selection_id)] - res_df['long_ep2_0_{}'.format(selection_id)])

    res_df['short_spread_{}'.format(selection_id)] = 1 + (res_df['short_tp_0_{}'.format(selection_id)].to_numpy() / res_df['short_tp_1_{}'.format(selection_id)].to_numpy() - 1) / 2
    res_df['long_spread_{}'.format(selection_id)] = 1 + (res_df['long_tp_1_{}'.format(selection_id)].to_numpy() / res_df['long_tp_0_{}'.format(selection_id)].to_numpy() - 1) / 2

    # Todo - public_indi 이전에 해야할지도 모름 # 'close', 'haopen', 'hahigh', 'halow', 'haclose'
    open, high, low, close = [res_df[col_].to_numpy() for col_ in ['open', 'high', 'low', 'close']]

    # ================== tr_set ================== #
    # ------------ tp ------------ #
    tpg = config.tr_set.tp_gap

    # ------ magnetic level ------ #
    #     cu_wrr_32_ = res_df['cu_wrr_32_{}{}'.format(wave_itv1, wave_period1)].to_numpy()
    #     co_wrr_32_ = res_df['co_wrr_32_{}{}'.format(wave_itv1, wave_period1)].to_numpy()

    #     short_magnetic_tpg = np.vectorize(get_next_wave_level)(cu_wrr_32_, 'tp')
    #     long_magnetic_tpg = np.vectorize(get_next_wave_level)(co_wrr_32_, 'tp')

    #     res_df['short_tp_{}'.format(selection_id)] = res_df['short_tp_1_{}'.format(selection_id)].to_numpy() - res_df['short_tp_gap_{}'.format(selection_id)].to_numpy() * short_magnetic_tpg  # ep 와 마찬가지로, tpg 기준 가능
    #     res_df['long_tp_{}'.format(selection_id)] = res_df['long_tp_1_{}'.format(selection_id)].to_numpy() + res_df['long_tp_gap_{}'.format(selection_id)].to_numpy() * long_magnetic_tpg

    # ------ 기준 : tp_1, gap : tp_box ------ #
    res_df['short_tp_{}'.format(selection_id)] = res_df['short_tp_1_{}'.format(selection_id)].to_numpy() - res_df['short_tp_gap_{}'.format(selection_id)].to_numpy() * tpg
    res_df['long_tp_{}'.format(selection_id)] = res_df['long_tp_1_{}'.format(selection_id)].to_numpy() + res_df['long_tp_gap_{}'.format(selection_id)].to_numpy() * tpg

    # res_df['short_tp_{}'.format(selection_id)] = res_df['short_out_1_{}'.format(selection_id)].to_numpy() - res_df[
    #     'short_out_gap_{}'.format(selection_id)].to_numpy() * tpg
    # res_df['long_tp_{}'.format(selection_id)] = res_df['long_out_1_{}'.format(selection_id)].to_numpy() + res_df[
    #     'long_out_gap_{}'.format(selection_id)].to_numpy() * tpg

    # res_df['short_tp_{}'.format(selection_id)] = short_tp_1 - short_epout_gap * tpg
    # res_df['long_tp_{}'.format(selection_id)] = long_tp_1 + long_epout_gap * tpg

    # ------ limit_ep1 ------ #
    if config.ep_set.entry_type == "LIMIT":
        epg1 = config.tr_set.ep_gap1

        # ------ magnetic level ------ #
        # cu_wrr_32_ = res_df['cu_wrr_32_{}{}'.format(wave_itv1, wave_period1)].to_numpy()
        # co_wrr_32_ = res_df['co_wrr_32_{}{}'.format(wave_itv1, wave_period1)].to_numpy()

        #         short_magnetic_outg = np.vectorize(get_next_wave_level)(cu_wrr_32_)
        #         long_magnetic_outg = np.vectorize(get_next_wave_level)(co_wrr_32_)

        #         res_df['short_ep1_{}'.format(selection_id)] = res_df['short_tp_1_{}'.format(selection_id)].to_numpy() - res_df['short_tp_gap_{}'.format(selection_id)].to_numpy() * short_magnetic_outg  # ep 와 마찬가지로, tpg 기준 가능
        #         res_df['long_ep1_{}'.format(selection_id)] = res_df['long_tp_1_{}'.format(selection_id)].to_numpy() + res_df['long_tp_gap_{}'.format(selection_id)].to_numpy() * long_magnetic_outg

        # ------ 기준 : ep1_0, gap : ep1_box ------ #
        res_df['short_ep1_{}'.format(selection_id)] = res_df['short_ep1_0_{}'.format(selection_id)].to_numpy() + res_df['short_ep1_gap_{}'.format(selection_id)].to_numpy() * epg1
        res_df['long_ep1_{}'.format(selection_id)] = res_df['long_ep1_0_{}'.format(selection_id)].to_numpy() - res_df['long_ep1_gap_{}'.format(selection_id)].to_numpy() * epg1

        # ------ 기준 : ep1_0, gap : tp_box ------ #
        # p1_hlm 을 위해선, tp_0 를 기준할 수 없음 --> ep1 & ep2 를 기준으로 진행
        # res_df['short_ep1_{}'.format(selection_id)] = res_df['short_ep1_0_{}'.format(selection_id)].to_numpy() + res_df['short_tp_gap_{}'.format(selection_id)].to_numpy() * epg1  # fibonacci 고려하면, tp / out gap 기준이 맞지 않을까
        # res_df['long_ep1_{}'.format(selection_id)] = res_df['long_ep1_0_{}'.format(selection_id)].to_numpy() - res_df['long_tp_gap_{}'.format(selection_id)].to_numpy() * epg1

        # ------ [ fibo_ep ] = 기준 : tp_1, gap : tp_box ------ #
        # res_df['short_ep1_{}'.format(selection_id)] = res_df['short_tp_1_{}'.format(selection_id)].to_numpy() - res_df['short_tp_gap_{}'.format(selection_id)].to_numpy() * epg1  # fibonacci 고려하면, tp / out gap 기준이 맞지 않을까
        # res_df['long_ep1_{}'.format(selection_id)] = res_df['long_tp_1_{}'.format(selection_id)].to_numpy() + res_df['long_tp_gap_{}'.format(selection_id)].to_numpy() * epg1

        # ------ market_ep1 ------ #
    else:
        res_df['short_ep1_{}'.format(selection_id)] = close
        res_df['long_ep1_{}'.format(selection_id)] = close

    # ------ limit_ep2 ------ #
    if config.ep_set.point2.entry_type == "LIMIT":
        epg2 = config.tr_set.ep_gap2

        # ------ magnetic level ------ #
        # cu_wrr_32_ = res_df['cu_wrr_32_{}{}'.format(wave_itv1, wave_period1)].to_numpy()
        # co_wrr_32_ = res_df['co_wrr_32_{}{}'.format(wave_itv1, wave_period1)].to_numpy()

        #         short_magnetic_outg = np.vectorize(get_next_wave_level)(cu_wrr_32_)
        #         long_magnetic_outg = np.vectorize(get_next_wave_level)(co_wrr_32_)

        #         res_df['short_ep2_{}'.format(selection_id)] = res_df['short_tp_1_{}'.format(selection_id)].to_numpy() - res_df['short_tp_gap_{}'.format(selection_id)].to_numpy() * short_magnetic_outg  # ep 와 마찬가지로, tpg 기준 가능
        #         res_df['long_ep2_{}'.format(selection_id)] = res_df['long_tp_1_{}'.format(selection_id)].to_numpy() + res_df['long_tp_gap_{}'.format(selection_id)].to_numpy() * long_magnetic_outg

        # ------ 기준 : ep2_0, gap : ep2_box ------ #
        res_df['short_ep2_{}'.format(selection_id)] = res_df['short_ep2_0_{}'.format(selection_id)].to_numpy() + res_df['short_ep2_gap_{}'.format(selection_id)].to_numpy() * epg2
        res_df['long_ep2_{}'.format(selection_id)] = res_df['long_ep2_0_{}'.format(selection_id)].to_numpy() - res_df['long_ep2_gap_{}'.format(selection_id)].to_numpy() * epg2

        # ------ 기준 : ep2_0, gap : out_box ------ # --> ep1 & ep2 를 기준으로 진행
        # res_df['short_ep2_{}'.format(selection_id)] = res_df['short_ep2_0_{}'.format(selection_id)].to_numpy() + res_df['short_out_gap_{}'.format(selection_id)].to_numpy() * epg2
        # res_df['long_ep2_{}'.format(selection_id)] = res_df['long_ep2_0_{}'.format(selection_id)].to_numpy() - res_df['long_out_gap_{}'.format(selection_id)].to_numpy() * epg2

        # ------ [ fibo_ep ] = 기준 : tp_1, gap : tp_box ------ #
        # res_df['short_ep2_{}'.format(selection_id)] = res_df['short_tp_1_{}'.format(selection_id)].to_numpy() - res_df['short_tp_gap_{}'.format(selection_id)].to_numpy() * epg2  # fibonacci 고려하면, tp / out gap 기준이 맞지 않을까
        # res_df['long_ep2_{}'.format(selection_id)] = res_df['long_tp_1_{}'.format(selection_id)].to_numpy() + res_df['long_tp_gap_{}'.format(selection_id)].to_numpy() * epg2

        # ------ [ fibo_ep ]  = 기준 : out_0, gap : out_box ------ #
        # res_df['short_ep2_{}'.format(selection_id)] = res_df['short_out_0_{}'.format(selection_id)].to_numpy() + res_df[
        #     'short_out_gap_{}'.format(selection_id)].to_numpy() * epg2  # fibonacci 고려하면, tp / out gap 기준이 맞지 않을까
        # res_df['long_ep2_{}'.format(selection_id)] = res_df['long_out_0_{}'.format(selection_id)].to_numpy() - res_df[
        #     'long_out_gap_{}'.format(selection_id)].to_numpy() * epg2

        # ------ market_ep2 ------ #
    else:
        res_df['short_ep2_{}'.format(selection_id)] = close
        res_df['long_ep2_{}'.format(selection_id)] = close

    # ------------ out ------------ #
    outg = config.tr_set.out_gap
    # res_df['short_out_{}'.format(selection_id)] = short_tp_0 + short_tp_gap * outg            # 1. for hhm check -> 규칙성과 wave_range 기반 거래 기준의 hhm 확인
    # res_df['long_out_{}'.format(selection_id)] = long_tp_0 - long_tp_gap * outg

    if config.tr_set.check_hlm == 0:

        # cu_wrr_32_ = res_df['cu_wrr_32_{}{}'.format(wave_itv1, wave_period1)].to_numpy()
        # co_wrr_32_ = res_df['co_wrr_32_{}{}'.format(wave_itv1, wave_period1)].to_numpy()

        # ------ magnetic level ------ #
        #         short_magnetic_outg = np.vectorize(get_next_wave_level)(cu_wrr_32_, 'out')
        #         long_magnetic_outg = np.vectorize(get_next_wave_level)(co_wrr_32_, 'out')

        #         res_df['short_out_{}'.format(selection_id)] = res_df['short_tp_1_{}'.format(selection_id)].to_numpy() - res_df['short_tp_gap_{}'.format(selection_id)].to_numpy() * short_magnetic_outg  # ep 와 마찬가지로, tpg 기준 가능
        #         res_df['long_out_{}'.format(selection_id)] = res_df['long_tp_1_{}'.format(selection_id)].to_numpy() + res_df['long_tp_gap_{}'.format(selection_id)].to_numpy() * long_magnetic_outg

        # ------ 기준 : out_0, gap : out_box ------ #
        res_df['short_out_{}'.format(selection_id)] = res_df['short_out_0_{}'.format(selection_id)].to_numpy() + res_df['short_out_gap_{}'.format(selection_id)].to_numpy() * outg
        res_df['long_out_{}'.format(selection_id)] = res_df['long_out_0_{}'.format(selection_id)].to_numpy() - res_df['long_out_gap_{}'.format(selection_id)].to_numpy() * outg

        # ------ 기준 : tp_0, gap : tp_box ------ #
        # res_df['short_out_{}'.format(selection_id)] = res_df['short_tp_0_{}'.format(selection_id)].to_numpy() + res_df['short_tp_gap_{}'.format(selection_id)].to_numpy() * outg
        # res_df['long_out_{}'.format(selection_id)] = res_df['long_tp_0_{}'.format(selection_id)].to_numpy() - res_df['long_tp_gap_{}'.format(selection_id)].to_numpy() * outg

        # ------ temporary refix for flexible p1_hhm ------ #
        # res_df['short_tp_0_{}'.format(selection_id)] = res_df['short_out_{}'.format(selection_id)]
        # res_df['long_tp_0_{}'.format(selection_id)] = res_df['long_out_{}'.format(selection_id)]

        # ------ 기준 : ep1_0, gap : ep1_box ------ #
        # res_df['short_out_{}'.format(selection_id)] = res_df['short_ep1_0_{}'.format(selection_id)].to_numpy() + res_df['short_ep1_gap_{}'.format(selection_id)].to_numpy() * outg
        # res_df['long_out_{}'.format(selection_id)] = res_df['long_ep1_0_{}'.format(selection_id)].to_numpy() - res_df['long_ep1_gap_{}'.format(selection_id)].to_numpy() * outg

    elif config.tr_set.check_hlm == 1:  # for p1_hlm
        # ------ ep1box as outg ------ #
        # res_df['short_out_{}'.format(selection_id)] = res_df['short_ep1_0_{}'.format(selection_id)].to_numpy() + res_df['short_ep1_gap_{}'.format(selection_id)].to_numpy() * outg
        # res_df['long_out_{}'.format(selection_id)] = res_df['long_ep1_0_{}'.format(selection_id)].to_numpy() - res_df['long_ep1_gap_{}'.format(selection_id)].to_numpy() * outg

        # ------ 1_tr - ep1box as outg ------ #
        # res_df['short_out_{}'.format(selection_id)] = res_df['short_ep1_0_{}'.format(selection_id)].to_numpy() + res_df['short_ep1_gap_{}'.format(selection_id)].to_numpy() * outg
        # res_df['long_out_{}'.format(selection_id)] = res_df['long_ep1_0_{}'.format(selection_id)].to_numpy() - res_df['long_ep1_gap_{}'.format(selection_id)].to_numpy() * outg

        # ------ 1_tr - auto_calculation by ep1 ------ #
        # res_df['short_out_{}'.format(selection_id)] = res_df['short_ep1_{}'.format(selection_id)] + (res_df['short_ep1_{}'.format(selection_id)].to_numpy() - res_df['short_tp_{}'.format(selection_id)].to_numpy())
        # res_df['long_out_{}'.format(selection_id)] = res_df['long_ep1_{}'.format(selection_id)].to_numpy() - (res_df['long_tp_{}'.format(selection_id)].to_numpy() - res_df['long_ep1_{}'.format(selection_id)].to_numpy())

        # ------ tpbox as outg ------ #
        # res_df['short_out_{}'.format(selection_id)] = res_df['short_ep1_0_{}'.format(selection_id)].to_numpy() + res_df['short_tp_gap_{}'.format(selection_id)].to_numpy() * outg  # ep 와 마찬가지로, tpg 기준 가능
        # res_df['long_out_{}'.format(selection_id)] = res_df['long_ep1_0_{}'.format(selection_id)].to_numpy() - res_df['long_tp_gap_{}'.format(selection_id)].to_numpy() * outg

        # ------ [fibo_out] = 기준 : tp_0, gap = tp_box ------ #
        res_df['short_out_{}'.format(selection_id)] = res_df['short_tp_0_{}'.format(selection_id)].to_numpy() + res_df[
            'short_tp_gap_{}'.format(selection_id)].to_numpy() * outg  # ep 와 마찬가지로, tpg 기준 가능
        res_df['long_out_{}'.format(selection_id)] = res_df['long_tp_0_{}'.format(selection_id)].to_numpy() - res_df['long_tp_gap_{}'.format(selection_id)].to_numpy() * outg

    else:  # p2_hlm
        # ------ 기준 : out_0, gap : out_box ------ #
        res_df['short_out_{}'.format(selection_id)] = res_df['short_out_0_{}'.format(selection_id)].to_numpy() + res_df['short_out_gap_{}'.format(selection_id)].to_numpy() * outg
        res_df['long_out_{}'.format(selection_id)] = res_df['long_out_0_{}'.format(selection_id)].to_numpy() - res_df['long_out_gap_{}'.format(selection_id)].to_numpy() * outg

        # res_df['short_out_{}'.format(selection_id)] = res_df['short_tp_0_{}'.format(selection_id)].to_numpy() + res_df['short_tp_gap_{}'.format(selection_id)].to_numpy() * outg
        # res_df['long_out_{}'.format(selection_id)] = res_df['long_tp_0_{}'.format(selection_id)].to_numpy() - res_df['long_tp_gap_{}'.format(selection_id)].to_numpy() * outg

        # ------ ep2box as out ------ #
        # res_df['short_out_{}'.format(selection_id)] = res_df['short_ep2_0_{}'.format(selection_id)].to_numpy() + res_df['short_ep2_gap_{}'.format(selection_id)].to_numpy() * outg   # p2's ep_box 를 out 으로 사용한다?
        # res_df['long_out_{}'.format(selection_id)] = res_df['long_ep2_0_{}'.format(selection_id)].to_numpy() - res_df['long_ep2_gap_{}'.format(selection_id)].to_numpy() * outg

    if mode == "OPEN":

        # ================== point ================== #
        short_open_res1 = np.ones(len_df)  # .astype(object)
        long_open_res1 = np.ones(len_df)  # .astype(object)
        short_open_res2 = np.ones(len_df)  # .astype(object)
        long_open_res2 = np.ones(len_df)  # .astype(object)

        # ------------ wave_point ------------ #
        # notnan_short_tc = ~pd.isnull(res_df['short_tc_{}{}'.format(wave_itv1, wave_period1)].to_numpy())  # isnull for object
        # notnan_long_tc = ~pd.isnull(res_df['long_tc_{}{}'.format(wave_itv1, wave_period1)].to_numpy())  # isnull for object

        notnan_cu = ~pd.isnull(res_df['wave_cu_{}{}'.format(wave_itv1, wave_period1)].to_numpy())  # isnull for object
        notnan_co = ~pd.isnull(res_df['wave_co_{}{}'.format(wave_itv1, wave_period1)].to_numpy())
        # notnan_cu2 = ~pd.isnull(res_df['wave_cu_{}{}'.format(wave_itv2, wave_period2)].to_numpy())  # isnull for object
        # notnan_co2 = ~pd.isnull(res_df['wave_co_{}{}'.format(wave_itv2, wave_period2)].to_numpy())

        short_open_res1 *= res_df['wave_cu_{}{}'.format(wave_itv1, wave_period1)].to_numpy().astype(
            bool) * notnan_cu  # object로 변환되는 경우에 대응해, bool 로 재정의
        long_open_res1 *= res_df['wave_co_{}{}'.format(wave_itv1, wave_period1)].to_numpy().astype(
            bool) * notnan_co  # np.nan = bool type 으로 True 임..
        # short_open_res1 *= res_df['short_tc_{}{}'.format(wave_itv1, wave_period1)].to_numpy().astype(bool) * notnan_short_tc
        # long_open_res1 *= res_df['long_tc_{}{}'.format(wave_itv1, wave_period1)].to_numpy().astype(bool) * notnan_long_tc

        # short_open_res2 *= res_df['wave_cu_{}{}'.format(wave_itv2, wave_period2)].to_numpy().astype(bool) * notnan_cu2  # object로 변환되는 경우에 대응해, bool 로 재정의
        # long_open_res2 *= res_df['wave_co_{}{}'.format(wave_itv2, wave_period2)].to_numpy().astype(bool) * notnan_co2  # np.nan = bool type 으로 True 임..
        # short_open_res2 *= res_df['short_tc_{}{}'.format(wave_itv2, tc_period)].to_numpy()
        # long_open_res2 *= res_df['long_tc_{}{}'.format(wave_itv2, tc_period)].to_numpy()

        # sys_log.warning => for pretty logging, 개행 문자를 추가하지 않고 이쁘게 log_file 에만 write 하는 법.
        if show_detail:
            sys_log.warning("wave_point")
            sys_log.warning("np.sum(short_open_res1 == 1) : {}".format(np.sum(short_open_res1 == 1)))
            sys_log.warning("np.sum(long_open_res1 == 1) : {}".format(np.sum(long_open_res1 == 1)))
            # sys_log.warning("np.sum(short_open_res2 == 1) : {}".format(np.sum(short_open_res2 == 1)))
            # sys_log.warning("np.sum(long_open_res2 == 1) : {}".format(np.sum(long_open_res2 == 1)))

        # ------ reject wave_update_hl ------ #
        notnan_update_low_cu = ~pd.isnull(res_df['wave_update_low_cu_bool_{}{}'.format(wave_itv1, wave_period1)].to_numpy())
        notnan_update_high_co = ~pd.isnull(
            res_df['wave_update_high_co_bool_{}{}'.format(wave_itv1, wave_period1)].to_numpy())
        # notnan_update_low_cu2 = ~pd.isnull(res_df['wave_update_low_cu_bool_{}{}'.format(wave_itv2, wave_period2)].to_numpy())
        # notnan_update_high_co2 = ~pd.isnull(res_df['wave_update_high_co_bool_{}{}'.format(wave_itv2, wave_period2)].to_numpy())

        short_open_res1 *= ~(res_df['wave_update_low_cu_bool_{}{}'.format(wave_itv1, wave_period1)].to_numpy().astype(
            bool)) * notnan_update_low_cu
        long_open_res1 *= ~(res_df['wave_update_high_co_bool_{}{}'.format(wave_itv1, wave_period1)].to_numpy().astype(
            bool)) * notnan_update_high_co
        # short_open_res2 *= ~(res_df['wave_update_low_cu_bool_{}{}'.format(wave_itv2, wave_period2)].to_numpy().astype(bool)) * notnan_update_low_cu2
        # long_open_res2 *= ~(res_df['wave_update_high_co_bool_{}{}'.format(wave_itv2, wave_period2)].to_numpy().astype(bool)) * notnan_update_high_co2

        if show_detail:
            sys_log.warning("reject update_hl")
            sys_log.warning("np.sum(short_open_res1 == 1) : {}".format(np.sum(short_open_res1 == 1)))
            sys_log.warning("np.sum(long_open_res1 == 1) : {}".format(np.sum(long_open_res1 == 1)))
        #     sys_log.warning("np.sum(short_open_res2 == 1) : {}".format(np.sum(short_open_res2 == 1)))
        #     sys_log.warning("np.sum(long_open_res2 == 1) : {}".format(np.sum(long_open_res2 == 1)))

        # ------ wave_itv ------ #
        if wave_itv1 != 'T':
            wave_itv1_num = itv_to_number(wave_itv1)
            short_open_res1 *= np_timeidx % wave_itv1_num == (wave_itv1_num - 1)
            long_open_res1 *= np_timeidx % wave_itv1_num == (wave_itv1_num - 1)

            if show_detail:
                sys_log.warning("wave_itv1")
                sys_log.warning("np.sum(short_open_res1 == 1) : {}".format(np.sum(short_open_res1 == 1)))
                sys_log.warning("np.sum(long_open_res1 == 1) : {}".format(np.sum(long_open_res1 == 1)))

        if wave_itv2 != 'T':
            wave_itv2_num = itv_to_number(wave_itv2)
            short_open_res2 *= np_timeidx % wave_itv2_num == (wave_itv2_num - 1)
            long_open_res2 *= np_timeidx % wave_itv2_num == (wave_itv2_num - 1)

            if show_detail:
                sys_log.warning("wave_itv2")
                sys_log.warning("np.sum(short_open_res2 == 1) : {}".format(np.sum(short_open_res2 == 1)))
                sys_log.warning("np.sum(long_open_res2 == 1) : {}".format(np.sum(long_open_res2 == 1)))

        """ wave conditions """
        # ------ wave_mm ------ #
        wave_high_terms_cnt_fill1_ = res_df['wave_high_terms_cnt_fill_{}{}'.format(wave_itv1, wave_period1)].to_numpy()
        wave_low_terms_cnt_fill1_ = res_df['wave_low_terms_cnt_fill_{}{}'.format(wave_itv1, wave_period1)].to_numpy()

        short_open_res1 *= (wave_high_terms_cnt_fill1_ > config.tr_set.wave_greater2) & (
                wave_low_terms_cnt_fill1_ > config.tr_set.wave_greater1)
        long_open_res1 *= (wave_low_terms_cnt_fill1_ > config.tr_set.wave_greater2) & (
                wave_high_terms_cnt_fill1_ > config.tr_set.wave_greater1)

        # wave_high_terms_cnt_fill2_ = res_df['wave_high_terms_cnt_fill_{}{}'.format(wave_itv2, wave_period2)].to_numpy()
        # wave_low_terms_cnt_fill2_ = res_df['wave_low_terms_cnt_fill_{}{}'.format(wave_itv2, wave_period2)].to_numpy()

        # short_open_res2 *= (wave_high_terms_cnt_fill2_ > config.tr_set.wave_greater2) & (wave_low_terms_cnt_fill2_ > config.tr_set.wave_greater1)
        # long_open_res2 *= (wave_low_terms_cnt_fill2_ > config.tr_set.wave_greater2) & (wave_high_terms_cnt_fill2_ > config.tr_set.wave_greater1)

        if show_detail:
            sys_log.warning("wave_mm")
            sys_log.warning("np.sum(short_open_res1 == 1) : {}".format(np.sum(short_open_res1 == 1)))
            sys_log.warning("np.sum(long_open_res1 == 1) : {}".format(np.sum(long_open_res1 == 1)))
            # sys_log.warning("np.sum(short_open_res2 == 1) : {}".format(np.sum(short_open_res2 == 1)))
            # sys_log.warning("np.sum(long_open_res2 == 1) : {}".format(np.sum(long_open_res2 == 1)))

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

        #     # ------ wave_spread ------ #
        #     if config.tr_set.wave_spread1 != "None":
        #       short_wave_spread_fill = res_df['short_wave_spread_fill_{}{}'.format(wave_itv1, wave_period1)].to_numpy()
        #       long_wave_spread_fill = res_df['long_wave_spread_fill_{}{}'.format(wave_itv1, wave_period1)].to_numpy()

        #       short_open_res1 *= short_wave_spread_fill >= config.tr_set.wave_spread1
        #       long_open_res1 *= long_wave_spread_fill >= config.tr_set.wave_spread1

        #       if show_detail:
        #         sys_log.warning("wave_spread")
        #         sys_log.warning("np.sum(short_open_res1 == 1) : {}".format(np.sum(short_open_res1 == 1)))
        #         sys_log.warning("np.sum(long_open_res1 == 1) : {}".format(np.sum(long_open_res1 == 1)))

        #     # ------ wave_time_ratio ------ #
        #     if config.tr_set.wave_time_ratio1 != "None":
        #       short_wave_time_ratio = res_df['short_wave_time_ratio_{}{}'.format(wave_itv1, wave_period1)].to_numpy()
        #       long_wave_time_ratio = res_df['long_wave_time_ratio_{}{}'.format(wave_itv1, wave_period1)].to_numpy()

        #       short_open_res1 *= short_wave_time_ratio >= config.tr_set.wave_time_ratio1
        #       long_open_res1 *= long_wave_time_ratio >= config.tr_set.wave_time_ratio1

        #       if show_detail:
        #         sys_log.warning("wave_time_ratio")
        #         sys_log.warning("np.sum(short_open_res1 == 1) : {}".format(np.sum(short_open_res1 == 1)))
        #         sys_log.warning("np.sum(long_open_res1 == 1) : {}".format(np.sum(long_open_res1 == 1)))

        # """ bb conditions """
        # ------ wave prime_idx : 연속된 wave_point 에 대해서 prime_idx 만 허용함 (wave_bb 에서 파생됨) ------ #
        #     short_open_res1 *= res_df['wave_cu_idx_{}{}'.format(wave_itv1, wave_period1)] == res_df['wave_cu_prime_idx_fill_{}{}'.format(wave_itv1, wave_period1)]
        #     long_open_res1 *= res_df['wave_co_idx_{}{}'.format(wave_itv1, wave_period1)] == res_df['wave_co_prime_idx_fill_{}{}'.format(wave_itv1, wave_period1)]

        #     if show_detail:
        #         sys_log.warning("wave prime_idx")
        #         sys_log.warning("np.sum(short_open_res1 == 1) : {}".format(np.sum(short_open_res1 == 1)))
        #         sys_log.warning("np.sum(long_open_res1 == 1) : {}".format(np.sum(long_open_res1 == 1)))

        # ------ inner_triangle ------ #
        #     # short_open_res1 *= (roll_highs1[-2] > roll_highs1[-1]) & (roll_lows1[-2] < roll_lows1[-1])
        #     # long_open_res1 *= (roll_lows1[-2] < roll_lows1[-1]) & (roll_highs1[-2] > roll_highs1[-1])
        #     short_open_res1 *= (roll_highs1[-2] > roll_highs1[-1])
        #     long_open_res1 *= (roll_lows1[-2] < roll_lows1[-1])
        #     # short_open_res1 *= (roll_lows1[-2] < roll_lows1[-1])
        #     # long_open_res1 *= (roll_highs1[-2] > roll_highs1[-1])

        #     if show_detail:
        #         sys_log.warning("inner_triangle")
        #         sys_log.warning("np.sum(short_open_res1 == 1) : {}".format(np.sum(short_open_res1 == 1)))
        #         sys_log.warning("np.sum(long_open_res1 == 1) : {}".format(np.sum(long_open_res1 == 1)))

        # ------------ point validation ------------ # - vecto. 로 미리 거를 수 있는걸 거르면 좋을 것
        short_tp_ = res_df['short_tp_{}'.format(selection_id)].to_numpy()
        short_ep1_ = res_df['short_ep1_{}'.format(selection_id)].to_numpy()
        short_ep2_ = res_df['short_ep2_{}'.format(selection_id)].to_numpy()
        short_out_ = res_df['short_out_{}'.format(selection_id)].to_numpy()

        long_tp_ = res_df['long_tp_{}'.format(selection_id)].to_numpy()
        long_ep1_ = res_df['long_ep1_{}'.format(selection_id)].to_numpy()
        long_ep2_ = res_df['long_ep2_{}'.format(selection_id)].to_numpy()
        long_out_ = res_df['long_out_{}'.format(selection_id)].to_numpy()

        # ------ p1 point_validation ------ # : csd 상황상 p1 point validation 진행하지 않음.
        short_open_res1 *= (short_tp_ < short_ep1_) & (
                short_ep1_ < short_out_)  # tr_set validation reject nan data & 정상 거래 위한 tp > ep / --> p2_box location (cannot be vectorized)
        # short_open_res1 *= close < short_ep1_   # reject entry open_execution
        short_open_res1 *= close < short_out_  # res_df['short_ep1_0_{}'.format(selection_id)].to_numpy()   # reject hl_out open_execution -> close always < ep1_0 at wave_p1
        # short_out_  res_df['short_tp_0_{}'.format(selection_id)].to_numpy() res_df['short_ep1_0_{}'.format(selection_id)].to_numpy()

        long_open_res1 *= (long_tp_ > long_ep1_) & (long_ep1_ > long_out_)  # (long_tp_ > long_ep_) # tr_set validation
        # long_open_res1 *= close > long_ep1_  # reject entry open_execution
        long_open_res1 *= close > long_out_  # res_df['long_ep1_0_{}'.format(selection_id)].to_numpy()  # reject hl_out open_execution
        # long_out_ res_df['long_tp_0_{}'.format(selection_id)].to_numpy() res_df['long_ep1_0_{}'.format(selection_id)].to_numpy()

        # ------ p2 point_validation ------ # deprecated => now, executed in en_ex_pairing() function.
        # short_open_res2 *= (short_ep2_ < short_out_) # tr_set validation (short_tp_ < short_ep_) # --> p2_box location (cannot be vectorized)
        # short_open_res2 *= close < short_out_    # reject hl_out open_execution
        # long_open_res2 *= (long_ep2_ > long_out_)  # tr_set validation (long_tp_ > long_ep_) &   # p2's ep & out cannot be vectorized
        # long_open_res2 *= close > long_out_    # reject hl_out open_execution

        res_df['short_open1_{}'.format(selection_id)] = short_open_res1 * (not config.pos_set.short_ban)
        res_df['long_open1_{}'.format(selection_id)] = long_open_res1 * (not config.pos_set.long_ban)

        res_df['short_open2_{}'.format(selection_id)] = short_open_res2
        res_df['long_open2_{}'.format(selection_id)] = long_open_res2

        if show_detail:
            sys_log.warning("point validation")
            sys_log.warning("np.sum(short_open_res1 == 1) : {}".format(np.sum(short_open_res1 == 1)))
            sys_log.warning("np.sum(long_open_res1 == 1) : {}".format(np.sum(long_open_res1 == 1)))
            # sys_log.warning("np.sum(short_open_res2 == 1) : {}".format(np.sum(short_open_res2 == 1)))
            # sys_log.warning("np.sum(long_open_res2 == 1) : {}".format(np.sum(long_open_res2 == 1)))

        # ------ tr ------ #
        if config.tr_set.check_hlm == 2:
            res_df['short_tr_{}'.format(selection_id)] = np.nan
            res_df['long_tr_{}'.format(selection_id)] = np.nan
        else:
            res_df['short_tr_{}'.format(selection_id)] = abs(
                (short_ep1_ / short_tp_ - config.trader_set.limit_fee - 1) / (
                        short_ep1_ / short_out_ - config.trader_set.market_fee - 1))  # 이게 맞음, loss 의 분모 > 분자 & profit 의 분모 < 분자
            res_df['long_tr_{}'.format(selection_id)] = abs(
                (long_tp_ / long_ep1_ - config.trader_set.limit_fee - 1) / (
                        long_out_ / long_ep1_ - config.trader_set.market_fee - 1))

    return res_df

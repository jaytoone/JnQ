import pandas as pd
import numpy as np
import time
from funcs.funcs_idep import *

class OrderSide:    # 추후 위치 옮길 것 - colab 에 binance_file 종속할 수 없어 이곳에 임시적으로 선언함
    BUY = "BUY"
    SELL = "SELL"
    INVALID = None

def get_res_v9(res_df, open_info_df_list, ohlc_list, config_list, np_timeidx, funcs, inversion=False, test_ratio=0.3, plot_is=True, signi=False, show_detail=False):
  # ------------ make open_info_list ------------ #
  open_idx1, open_idx2 = [open_info_df.index.to_numpy() for open_info_df in open_info_df_list]
  len_df = len(res_df)

  sample_len = int(len_df * (1 - test_ratio))
  sample_idx1 = (open_idx1 < sample_len) == plot_is  # in / out sample plot 여부
  sample_open_idx1 = open_idx1[sample_idx1]
  sample_idx2 = (open_idx2 < sample_len) == plot_is  # in / out sample plot 여부

  # ------------ open_info_list 기준 = p1 ------------ #
  sample_open_info_df1, sample_open_info_df2 = [df_[idx_] for df_, idx_ in zip(open_info_df_list, [sample_idx1, sample_idx2])]
  open_info1 = [sample_open_info_df1[col_].to_numpy() for col_ in sample_open_info_df1.columns]

  if config_list[0].tr_set.check_hlm in [0, 1]:   # 여기서 open_info 자동화하더라도, utils info 는 직접 실행해주어야함
    sample_open_idx2 = sample_open_idx1
    open_info2 = open_info1
  else:
    sample_open_idx2 = open_idx2[sample_idx2]
    open_info2 = [sample_open_info_df2[col_].to_numpy() for col_ in sample_open_info_df2.columns]

  # ------------ get paired_res ------------ #
  start_0 = time.time()
  paired_res = en_ex_pairing_v9_4(res_df, [sample_open_idx1, sample_open_idx2], [open_info1, open_info2], ohlc_list, config_list, np_timeidx, funcs, show_detail)
  # valid_openi_arr, pair_idx_arr, pair_price_arr, lvrg_arr, fee_arr, tpout_arr = paired_res
  print("en_ex_pairing elapsed time :", time.time() - start_0)  #  0.37 --> 0.3660471439361572 --> 0.21(lesser if)

  # ------------ idep_plot ------------ #
  start_0 = time.time()
  high, low = ohlc_list[1:3]
  res = idep_plot_v16_2(res_df, len_df, config_list[0], high, low, sample_open_info_df1, paired_res, inversion=inversion, sample_ratio=1 - test_ratio, signi=signi)
  print("idep_plot elapsed time :", time.time() - start_0)   # 1.40452 (v6) 1.4311 (v5)

  return res

def get_res_v8(res_df, open_info_df_list, ohlc_list, config_list, np_timeidx, funcs, check_hlm=0, inversion=False, test_ratio=0.3, plot_is=True, signi=False, show_detail=False):
  # ------------ make open_info_list ------------ #
  open_idx1, open_idx2 = [open_info_df.index.to_numpy() for open_info_df in open_info_df_list]
  len_df = len(res_df)

  sample_len = int(len_df * (1 - test_ratio))
  sample_idx1 = (open_idx1 < sample_len) == plot_is  # in / out sample plot 여부
  sample_open_idx1 = open_idx1[sample_idx1]
  sample_idx2 = (open_idx2 < sample_len) == plot_is  # in / out sample plot 여부

  # ------------ open_info_list 기준 = p1 ------------ #
  sample_open_info_df1, sample_open_info_df2 = [df_[idx_] for df_, idx_ in zip(open_info_df_list, [sample_idx1, sample_idx2])]
  open_info1 = [sample_open_info_df1[col_].to_numpy() for col_ in sample_open_info_df1.columns]

  if check_hlm in [0, 1]:   # 여기서 open_info 자동화하더라도, utils info 는 직접 실행해주어야함
    sample_open_idx2 = sample_open_idx1
    open_info2 = open_info1
  else:
    sample_open_idx2 = open_idx2[sample_idx2]
    open_info2 = [sample_open_info_df2[col_].to_numpy() for col_ in sample_open_info_df2.columns]

  # ------------ get paired_res ------------ #
  start_0 = time.time()
  paired_res = en_ex_pairing_v8(res_df, [sample_open_idx1, sample_open_idx2], [open_info1, open_info2], ohlc_list, config_list, np_timeidx, funcs, check_hlm, show_detail)
  # valid_openi_arr, pair_idx_arr, pair_price_arr, lvrg_arr, fee_arr, tpout_arr = paired_res
  print("en_ex_pairing elapsed time :", time.time() - start_0)  #  0.37 --> 0.3660471439361572 --> 0.21(lesser if)

  # ------------ idep_plot ------------ #
  start_0 = time.time()
  high, low = ohlc_list[1:3]
  res = idep_plot_v16(res_df, len_df, config_list[0], high, low, sample_open_info_df1, paired_res, inversion=inversion, sample_ratio=1 - test_ratio, signi=signi)
  print("idep_plot elapsed time :", time.time() - start_0)   # 1.40452 (v6) 1.4311 (v5)

  return res

def get_res_v7(res_df, open_info_df_list, ohlc_list, config_list, np_timeidx, funcs, check_net_hhm=False, inversion=False, test_ratio=0.3, plot_is=True, signi=False, show_detail=False):
  # ------------ make open_info_list ------------ #
  open_idx1, open_idx2 = [open_info_df.index.to_numpy() for open_info_df in open_info_df_list]
  len_df = len(res_df)

  sample_len = int(len_df * (1 - test_ratio))
  sample_idx1 = (open_idx1 < sample_len) == plot_is  # in / out sample plot 여부
  sample_open_idx1 = open_idx1[sample_idx1]
  sample_idx2 = (open_idx2 < sample_len) == plot_is  # in / out sample plot 여부

  # ------------ open_info_list 기준 = p1 ------------ #
  sample_open_info_df1, sample_open_info_df2 = [df_[idx_] for df_, idx_ in zip(open_info_df_list, [sample_idx1, sample_idx2])]
  open_info1 = [sample_open_info_df1[col_].to_numpy() for col_ in sample_open_info_df1.columns]

  if check_net_hhm:   # 여기서 open_info 자동화하더라도, utils info 는 직접 실행해주어야함
    sample_open_idx2 = sample_open_idx1
    open_info2 = open_info1
  else:
    sample_open_idx2 = open_idx2[sample_idx2]
    open_info2 = [sample_open_info_df2[col_].to_numpy() for col_ in sample_open_info_df2.columns]

  # ------------ get paired_res ------------ #
  start_0 = time.time()
  paired_res = en_ex_pairing_v7(res_df, [sample_open_idx1, sample_open_idx2], [open_info1, open_info2], ohlc_list, config_list, np_timeidx, funcs, check_net_hhm, show_detail)
  # valid_openi_arr, pair_idx_arr, pair_price_arr, lvrg_arr, fee_arr, tpout_arr = paired_res
  print("en_ex_pairing elapsed time :", time.time() - start_0)  #  0.37 --> 0.3660471439361572 --> 0.21(lesser if)

  # ------------ idep_plot ------------ #
  start_0 = time.time()
  high, low = ohlc_list[1:3]
  res = idep_plot_v16(res_df, len_df, config_list[0], high, low, sample_open_info_df1, paired_res, inversion=inversion, sample_ratio=1 - test_ratio, signi=signi)
  print("idep_plot elapsed time :", time.time() - start_0)   # 1.40452 (v6) 1.4311 (v5)

  return res
def get_res_v6(res_df, open_info_df_list, ohlc_list, config_list, np_timeidx, funcs, inversion=False, test_ratio=0.3, plot_is=True, signi=False):
  # ------------ make open_info_list ------------ #
  open_idx1, open_idx2 = [open_info_df.index.to_numpy() for open_info_df in open_info_df_list]
  len_df = len(res_df)

  sample_len = int(len_df * (1 - test_ratio))
  sample_idx1 = (open_idx1 < sample_len) == plot_is  # in / out sample plot 여부
  sample_idx2 = (open_idx2 < sample_len) == plot_is  # in / out sample plot 여부

  sample_open_idx1 = open_idx1[sample_idx1]
  sample_open_idx2 = open_idx2[sample_idx2]

  # ------------ open_info_list 기준 = p1 ------------ #
  sample_open_info_df1 = open_info_df_list[0][sample_idx1]
  sample_open_info_df1, sample_open_info_df2 = [df_[idx_] for df_, idx_ in zip(open_info_df_list, [sample_idx1, sample_idx2])]
  open_info1 = [sample_open_info_df1[col_].to_numpy() for col_ in sample_open_info_df1.columns]
  open_info2 = [sample_open_info_df2[col_].to_numpy() for col_ in sample_open_info_df2.columns]
  # side_arr, zone_arr, id_arr, id_idx_arr = open_info1
  side_arr, _, _, _ = open_info1

  # ------------ get paired_res ------------ #
  start_0 = time.time()
  paired_res = en_ex_pairing_v6(res_df, [sample_open_idx1, sample_open_idx2], [open_info1, open_info2], ohlc_list, config_list, np_timeidx, funcs)
  # valid_openi_arr, pair_idx_arr, pair_price_arr, lvrg_arr, fee_arr, tpout_arr = paired_res
  print("en_ex_pairing elapsed time :", time.time() - start_0)  #  0.37 --> 0.3660471439361572 --> 0.21(lesser if)

  # ------------ idep_plot ------------ #
  start_0 = time.time()
  high, low = ohlc_list[1:3]
  res = idep_plot_v15(res_df, len_df, config_list[0], high, low, sample_open_idx1, side_arr, paired_res, inversion=inversion, sample_ratio=1 - test_ratio, signi=signi)
  print("idep_plot elapsed time :", time.time() - start_0)   # 1.40452 (v6) 1.4311 (v5)

  return res


def get_res_v5(res_df, open_info_df, ohlc_list, config_list, np_timeidx, funcs, inversion=False, test_ratio=0.3, plot_is=True, signi=False):
  # ------------ make open_info_list ------------ #
  open_idx = open_info_df.index.to_numpy()
  len_df = len(res_df)
  s_idx = (open_idx < int(len_df * (1 - test_ratio))) == plot_is
  s_open_info_df = open_info_df[s_idx]
  s_open_idx = open_idx[s_idx]

  open_info_list = [s_open_info_df[col_].to_numpy() for col_ in s_open_info_df.columns]
  side_arr, zone_arr, id_arr, id_idx_arr = open_info_list

  # ------------ get paired_res ------------ #
  start_0 = time.time()
  paired_res = en_ex_pairing_v5(res_df, s_open_idx, open_info_list, ohlc_list, config_list, np_timeidx, funcs)
  # valid_openi_arr, pair_idx_arr, pair_price_arr, lvrg_arr, fee_arr, tpout_arr = paired_res
  print("en_ex_pairing elapsed time :", time.time() - start_0)  #  0.37 --> 0.3660471439361572 --> 0.21(lesser if)

  # ------------ idep_plot ------------ #
  start_0 = time.time()
  high, low = ohlc_list[1:3]
  res = idep_plot_v14(res_df, len_df, config_list[0], high, low, s_open_idx, side_arr, paired_res, inversion=inversion, sample_ratio=1 - test_ratio, signi=signi)
  print("idep_plot elapsed time :", time.time() - start_0)   # 1.40452 (v6) 1.4311 (v5)

  return res

def get_open_info_df_v2(ep_loc_v2, res_df, np_timeidx, ID_list, config_list, id_idx_list, open_num=1):
  start_0 = time.time()
  # ------ get mr_res, zone_arr ------ #
  short_mr_res_obj = np.array([ep_loc_v2(res_df, config_, np_timeidx, show_detail=True, ep_loc_side=OrderSide.SELL) for config_ in config_list])
  long_mr_res_obj = np.array([ep_loc_v2(res_df, config_, np_timeidx, show_detail=True, ep_loc_side=OrderSide.BUY) for config_ in config_list])
  short_open_idx_list = [np.where(res_df['short_open{}_{}'.format(open_num, id)].to_numpy() * mr_res)[0] for id, mr_res in zip(ID_list, short_mr_res_obj[:, 0].astype(np.float64))]   # "point * mr_Res"
  long_open_idx_list = [np.where(res_df['long_open{}_{}'.format(open_num, id)].to_numpy() * mr_res)[0] for id, mr_res in zip(ID_list, long_mr_res_obj[:, 0].astype(np.float64))]  # zip 으로 zone (str) 과 묶어서 dtype 변경됨

  # ------ open_info_arr ------ #
  short_side_list = [np.full(len(list_), OrderSide.SELL) for list_ in short_open_idx_list]
  long_side_list = [np.full(len(list_), OrderSide.BUY) for list_ in long_open_idx_list]

  short_zone_list = [zone_res[short_open_idx] for zone_res, short_open_idx in zip(short_mr_res_obj[:, 1], short_open_idx_list)]
  long_zone_list = [zone_res[long_open_idx] for zone_res, long_open_idx in zip(long_mr_res_obj[:, 1], long_open_idx_list)]

  short_id_list = [np.full(len(list_), id) for id, list_ in zip(ID_list, short_open_idx_list)]
  long_id_list = [np.full(len(list_), id) for id, list_ in zip(ID_list, long_open_idx_list)]

  selected_id_idx = np.arange(len(id_idx_list))
  short_id_idx_list = [np.full(len(list_), id) for id, list_ in zip(selected_id_idx, short_open_idx_list)]
  long_id_idx_list = [np.full(len(list_), id) for id, list_ in zip(selected_id_idx, long_open_idx_list)]

  # ------ get open_info_df ------ #
  #   series 만들어서 short / long 끼리 합치고 둘이 합치고, 중복은 우선 순위 정해서 제거
  short_open_df_list = [pd.DataFrame(index=index_, data=np.vstack((data_)).T, columns=['side', 'zone', 'id', 'id_idx']) for index_, data_ in zip(short_open_idx_list, zip(short_side_list, short_zone_list, short_id_list, short_id_idx_list))]
  long_open_df_list = [pd.DataFrame(index=index_, data=np.vstack((data_)).T, columns=['side', 'zone', 'id', 'id_idx']) for index_, data_ in zip(long_open_idx_list, zip(long_side_list, long_zone_list, long_id_list, long_id_idx_list))]

  open_info_df = pd.concat(short_open_df_list + long_open_df_list)
  # ------ sorting + unique ------ #
  open_info_df.sort_index(inplace=True)
  # print(len(open_info_df))
  # print(len(open_info_df))
  # open_info_df.head()
  print("get_open_info_df elapsed time :", time.time() - start_0)
  return open_info_df[~open_info_df.index.duplicated(keep='first')]  # 먼저 순서를 우선으로 지정

def get_open_info_df(ep_loc_v2, res_df, np_timeidx, ID_list, config_list, id_idx_list):
  start_0 = time.time()
  # ------ get mr_res, zone_arr ------ #
  short_mr_res_obj = np.array([ep_loc_v2(res_df, config_, np_timeidx, show_detail=True, ep_loc_side=OrderSide.SELL) for config_ in config_list])
  long_mr_res_obj = np.array([ep_loc_v2(res_df, config_, np_timeidx, show_detail=True, ep_loc_side=OrderSide.BUY) for config_ in config_list])
  short_open_idx_list = [np.where(res_df['short_open_{}'.format(id)].to_numpy() * mr_res)[0] for id, mr_res in zip(ID_list, short_mr_res_obj[:, 0].astype(np.float64))]  # zip 으로 zone (str) 과 묶어서 dtype 변경됨
  long_open_idx_list = [np.where(res_df['long_open_{}'.format(id)].to_numpy() * mr_res)[0] for id, mr_res in zip(ID_list, long_mr_res_obj[:, 0].astype(np.float64))]

  # ------ open_info_arr ------ #
  short_side_list = [np.full(len(list_), OrderSide.SELL) for list_ in short_open_idx_list]
  long_side_list = [np.full(len(list_), OrderSide.BUY) for list_ in long_open_idx_list]

  short_zone_list = [zone_res[short_open_idx] for zone_res, short_open_idx in zip(short_mr_res_obj[:, 1], short_open_idx_list)]
  long_zone_list = [zone_res[long_open_idx] for zone_res, long_open_idx in zip(long_mr_res_obj[:, 1], long_open_idx_list)]

  short_id_list = [np.full(len(list_), id) for id, list_ in zip(ID_list, short_open_idx_list)]
  long_id_list = [np.full(len(list_), id) for id, list_ in zip(ID_list, long_open_idx_list)]

  selected_id_idx = np.arange(len(id_idx_list))
  short_id_idx_list = [np.full(len(list_), id) for id, list_ in zip(selected_id_idx, short_open_idx_list)]
  long_id_idx_list = [np.full(len(list_), id) for id, list_ in zip(selected_id_idx, long_open_idx_list)]

  # ------ get open_info_df ------ #
  #   series 만들어서 short / long 끼리 합치고 둘이 합치고, 중복은 우선 순위 정해서 제거
  short_open_df_list = [pd.DataFrame(index=index_, data=np.vstack((data_)).T, columns=['side', 'zone', 'id', 'id_idx']) for index_, data_ in zip(short_open_idx_list, zip(short_side_list, short_zone_list, short_id_list, short_id_idx_list))]
  long_open_df_list = [pd.DataFrame(index=index_, data=np.vstack((data_)).T, columns=['side', 'zone', 'id', 'id_idx']) for index_, data_ in zip(long_open_idx_list, zip(long_side_list, long_zone_list, long_id_list, long_id_idx_list))]

  open_info_df = pd.concat(short_open_df_list + long_open_df_list)
  # ------ sorting + unique ------ #
  open_info_df.sort_index(inplace=True)
  # print(len(open_info_df))
  # print(len(open_info_df))
  # open_info_df.head()
  print("get_open_info_df elapsed time :", time.time() - start_0)
  return open_info_df[~open_info_df.index.duplicated(keep='first')]  # 먼저 순서를 우선으로 지정


# p2_wave validation, p2_wave high validation, out_1's expiry 사용중
def en_ex_pairing_v9_4(res_df, open_idx_list, open_info_list, ohlc_list, config_list, np_timeidx, funcs, show_detail=False):  # 이미 충분히 줄여놓은 idx 임

    open_info1, open_info2 = open_info_list
    side_arr1, _, _, id_idx_arr1 = open_info1
    side_arr2, _, _, _ = open_info2

    expiry_p1, expiry_p2, lvrg_set = funcs

    net_p1_idx_list, p1_idx_list, p2_idx_list, pair_idx_list, pair_price_list, lvrg_list, fee_list, tpout_list, tr_list = [[] for li in range(9)]
    len_df = len(res_df)

    open, high, low, close = ohlc_list

    open_idx1, open_idx2 = open_idx_list
    len_open_idx1 = len(open_idx1)
    len_open_idx2 = len(open_idx2)
    i, open_i1, open_i2 = 0, -1, -1  # i for total_res_df indexing

    while 1:  # for p1's loop

        # Todo, 
        #   1. (갱신) p1's open_i + 1 과 op_idx 를 꺼내오는 건, eik1 또는 tp 체결의 경우만 해당됨, 
        #   2. out 의 경우 p2's op_idx 기준으로 retry 필요
        #     a. 또한, p2's op_idx > p1's op_idx

        # ============ get p1_info ============ #
        # if eik1 or tp_done or first loop:
        open_i1 += 1  # 확인 끝났으면 조기 이탈(+1), 다음 open_idx 조사 진행
        if open_i1 >= len_open_idx1:
            break

        if show_detail:
            print("open_i1 :", open_i1, side_arr1[open_i1])

        op_idx1 = open_idx1[open_i1]  # open_i1 는 i 와 별개로 운영
        if op_idx1 < i:  # i = 이전 거래 끝난후의 res_df index - "거래 종료후 거래 시작", '<' : 거래 종료시점 진입 가능하다는 의미
            continue

        # ------ set loop index i ------ #
        i = op_idx1  # + 1 --> op_idx1 = op_idx2 가능함 # open_signal 이 close_bar.shift(1) 이라고 가정하고 다음 bar 부터 체결확인한다는 의미
        if i >= len_df:  # res_df 의 last_index 까지 돌아야함
            break

            # ------ dynamic arr info by ID ------ #
        #     1. 해당 id 로 config 재할당해야함
        #       a. use open_i1
        open_side = side_arr1[open_i1]
        id_idx = id_idx_arr1.astype(int)[open_i1]
        config = config_list[id_idx]
        selection_id = config.selection_id
        check_hlm = config.tr_set.check_hlm

        # check_net_hhm = 1 if (config.tr_set.wave_itv1 == config.tr_set.wave_itv2) and (config.tr_set.wave_period1 == config.tr_set.wave_period2) else 0

        side_pos = 'short' if open_side == OrderSide.SELL else 'long'
        if show_detail:
            print("============ op_idx1 : {} {} ============".format(op_idx1, open_side))

        # if show_detail:
        #   print("check_hlm :", check_hlm)

        # ------ load tr_data ------ #
        tp_arr = res_df['{}_tp_{}'.format(side_pos, selection_id)].to_numpy()
        ep1_arr = res_df['{}_ep1_{}'.format(side_pos, selection_id)].to_numpy()
        ep2_arr = res_df['{}_ep2_{}'.format(side_pos, selection_id)].to_numpy()
        out_arr = res_df['{}_out_{}'.format(side_pos, selection_id)].to_numpy()

        tr_arr = res_df['{}_tr_{}'.format(side_pos, selection_id)].to_numpy()  # just for p1_hhm

        tp_1_ = res_df['{}_tp_1_{}'.format(side_pos, selection_id)].to_numpy()[op_idx1]  # for p2_box location & p1's exipiry
        tp_0_ = res_df['{}_tp_0_{}'.format(side_pos, selection_id)].to_numpy()[op_idx1]
        tp_gap_ = res_df['{}_tp_gap_{}'.format(side_pos, selection_id)].to_numpy()[op_idx1]

        # if not check_net_hhm:  # this phase exist for p1 entry (net hhm sync.) in p2_platform
        exec_j, ep_j, tp_j, out_j, entry_done, en_p, fee = check_entry_v6(res_df, config, config.ep_set.entry_type, op_idx1, tp_1_, tp_gap_, len_df,
                                                                          open_side,
                                                                          [*ohlc_list, ep1_arr], expiry_p2)
        i = exec_j  # = entry_loop 를 돌고 나온 e_j
        if not entry_done:
            if show_detail:
                print("p1's expiry by expiry_p2 function in p1's loop : continue")
            continue
            # else:        
        #   tp_j = op_idx1

        prev_open_i2 = open_i2
        net_p1_idx_list.append(op_idx1)
        # if check_hlm in [0, 1]:
        #   i = op_idx1  # allow op_idx2 = op_idx1
        allow_exit = 1
        # ============ entry loop ============ #
        while 1:  # for p2's loop (allow retry)

            # ============ get p2_info ============ #
            if check_hlm in [1, 2]:
                open_i2 += 1  # 확인 끝났으면 조기 이탈(+1), 다음 open_idx 조사 진행
                if open_i2 >= len_open_idx2:  # open_i2 소진
                    break

                if show_detail:
                    print("open_i2 :", open_i2, side_arr2[open_i2])

                # ------ check side sync. ------ #
                if open_side != side_arr2[open_i2]:
                    continue

                # ------ assert, op_idx2 >= exec_j ------ #
                op_idx2 = open_idx2[open_i2]  # open_i2 는 i 와 별개로 운영
                if check_hlm == 1 and allow_exit:
                    if op_idx2 < op_idx1:
                        continue
                else:
                    if op_idx2 < i:  # p1 execution 이후의 i 를 허용 (old, 이곳 i = op_idx1 + 1 or p2's exec_j or exit_loop's i + 1)
                        continue

                if check_hlm == 2:
                    i = op_idx2 + 1  # open_signal 이 close_bar.shift(1) 이라고 가정하고 다음 bar 부터 체결확인한다는 의미
                    if i >= len_df:  # res_df 의 last_index 까지 돌아야함
                        break

                if show_detail:
                    print("op_idx1, op_idx2 :", op_idx1, op_idx2, side_arr2[open_i2])

            else:
                op_idx2 = op_idx1

            tp_ = tp_arr[op_idx1]
            ep2_ = ep2_arr[op_idx2]
            out_ = out_arr[op_idx2]

            out_1_ = res_df['{}_out_1_{}'.format(side_pos, selection_id)].to_numpy()[op_idx2]
            out_0_ = res_df['{}_out_0_{}'.format(side_pos, selection_id)].to_numpy()[op_idx2]
            out_gap_ = res_df['{}_out_gap_{}'.format(side_pos, selection_id)].to_numpy()[op_idx2]

            # ------ const. for p2_wave ------ #
            wave_itv1 = config.tr_set.wave_itv1
            wave_period1 = config.tr_set.wave_period1
            wave_itv2 = config.tr_set.wave_itv2
            wave_period2 = config.tr_set.wave_period2

            if check_hlm in [1, 2]:
                # ------ check p1's expiry - Todo, priority ------ # - p2_box 생성 이전의 hl_survey
                # 1. op_idx1 ~ op_idx2 까지의 hl_check 
                # if check_hlm:  # p1_hlm, p2_hlm --> Todo, 이거를 왜 p1_hlm 에도 적용했는지 잘 모르겠음
                if op_idx1 < op_idx2:
                    expire, touch_idx = expiry_p1(res_df, config, op_idx1, op_idx2, tp_1_, tp_0_, tp_gap_, ohlc_list[1:3], open_side)
                    if expire:  # p1's expiry
                        if show_detail:
                            print("expiry_p1, touch_idx = {} : break".format(touch_idx))
                        i = touch_idx  # + 1  --> 이거 아닌것 같음 # op_idx1 과 op_idx2 사이의 op_idx1' 을 살리기 위함, 즉 바로 다음 op_idx1 로 회귀 (건너뛰지 않고)
                        open_i2 = prev_open_i2
                        break  # change op_idx1

                if check_hlm == 2:
                    # ------ p2 point_validation - vectorization unavailable ------ # p1_loop 로 return 되는 정확한 i 를 반환하기 위해서 expiry_p1 에 순서 양보  # Todo, 새로운 tp, ep, out 에 대한 처리 필요 (p1_hlm 사용시)
                    if open_side == OrderSide.SELL:
                        # --- p2_wave validation --- #
                        wave_co_post_idx = res_df['wave_co_post_idx_fill_{}{}'.format(wave_itv2, wave_period2)].to_numpy()[op_idx2]
                        if not (op_idx1 < wave_co_post_idx):
                            if show_detail:
                                print("p2_wave validation : continue")
                            continue  # change op_idx2

                        # --- p2_wave high validation --- #
                        # wave_high_fill1_ = res_df['wave_high_fill_{}{}'.format(wave_itv1, wave_period1)].to_numpy()[op_idx1]
                        # wave_high_fill2_ = res_df['wave_high_fill_{}{}'.format(wave_itv2, wave_period2)].to_numpy()[op_idx2]
                        # if not (wave_high_fill1_ >= wave_high_fill2_):
                        #   if show_detail:
                        #     print("p2_wave high validation : continue")
                        #   continue  # change op_idx2

                        if not (tp_ < ep2_):  # tr_set validation & reject hl_out open_exec.
                            break  # change op_idx1
                        elif not (ep2_ < out_ and close[op_idx2] < out_):
                            if show_detail:
                                print("point validation : continue")
                            continue  # change op_idx2
                    else:
                        # --- p2_wave validation --- #
                        wave_cu_post_idx = res_df['wave_cu_post_idx_fill_{}{}'.format(wave_itv2, wave_period2)].to_numpy()[op_idx2]
                        if not (op_idx1 < wave_cu_post_idx):
                            if show_detail:
                                print("p2_wave validation : continue")
                            continue  # change op_idx2

                        # --- p2_wave low validation --- #
                        # wave_low_fill1_ = res_df['wave_low_fill_{}{}'.format(wave_itv1, wave_period1)].to_numpy()[op_idx1]
                        # wave_low_fill2_ = res_df['wave_low_fill_{}{}'.format(wave_itv2, wave_period2)].to_numpy()[op_idx2]
                        # if not (wave_low_fill1_ <= wave_low_fill2_):
                        #   if show_detail:
                        #     print("p2_wave low validation : continue")
                        #   continue  # change op_idx2

                        if not (tp_ > ep2_):
                            break
                        elif not (ep2_ > out_ and close[op_idx2] > out_):
                            if show_detail:
                                print("point validation : continue")
                            continue

                    # ------ p2_box location ------ #
                    if open_side == OrderSide.SELL:
                        if not ((tp_1_ + tp_gap_ * config.tr_set.p2_box_k1 <= out_1_) and (
                                out_0_ <= tp_0_ - tp_gap_ * config.tr_set.p2_box_k2)):  # tp1, tp0 에 닿으면 expiry
                            # if not ((tp_1_ + tp_gap_ * config.tr_set.p2_box_k1 >= out_1_) and (out_0_ <= tp_0_ - tp_gap_ * config.tr_set.p2_box_k2)):  # tp1, tp0 에 닿으면 expiry
                            if show_detail:
                                print("p2_box rejection : continue")
                            continue
                        else:
                            # ------ p1p2_low ------ #
                            if not high[op_idx1:op_idx2 + 1].max() < tp_0_ - tp_gap_ * config.tr_set.p1p2_low:
                                if show_detail:
                                    print("p1p2_low rejection : continue")
                                continue
                    else:
                        if not ((tp_1_ - tp_gap_ * config.tr_set.p2_box_k1 >= out_1_) and (out_0_ >= tp_0_ + tp_gap_ * config.tr_set.p2_box_k2)):
                            # if not ((tp_1_ - tp_gap_ * config.tr_set.p2_box_k1 <= out_1_) and (out_0_ >= tp_0_ + tp_gap_ * config.tr_set.p2_box_k2)):
                            if show_detail:
                                print("p2_box rejection : continue")
                            continue
                        else:
                            # ------ p1p2_low ------ #
                            if not low[op_idx1:op_idx2 + 1].min() > tp_0_ + tp_gap_ * config.tr_set.p1p2_low:
                                if show_detail:
                                    print("p1p2_low rejection : continue")
                                continue

                    # ------ check p2's expiry ------ # - 현재, op_idx2 기준의 ep2_arr 을 사용 중임.
                    # exec_j, ep_j, _, out_j, entry_done, en_p, fee = check_entry_v6(res_df, config, config.ep_set.point2.entry_type, op_idx2, tp_1_, tp_gap_, len_df, open_side,
                    #                                                                         [*ohlc_list, ep2_arr], expiry_p2)   # Todo, tp_1 & tp_gap 사용이 맞을 것으로 봄                                                                                      
                    exec_j, ep_j, _, out_j, entry_done, en_p, fee = check_entry_v6(res_df, config, config.ep_set.point2.entry_type, op_idx2, out_1_,
                                                                                   out_gap_, len_df, open_side,
                                                                                   [*ohlc_list, ep2_arr],
                                                                                   expiry_p2)  # Todo, tp_1 & tp_gap 사용이 맞을 것으로 봄
                    i = exec_j  # = entry_loop 를 돌고 나온 e_j
                    if not entry_done:  # p2's expiry
                        if show_detail:
                            print("expiry_p2, i = {} : continue".format(i))
                        continue  # change op_idx2            

                    # ------ devectorized tr_calc ------ # - en_p 에 대해 하는게 맞을 것으로봄
                    if open_side == OrderSide.SELL:
                        tr_ = abs((en_p / tp_ - config.trader_set.limit_fee - 1) / (en_p / out_ - config.trader_set.market_fee - 1))
                    else:
                        tr_ = abs((tp_ / en_p - config.trader_set.limit_fee - 1) / (out_ / en_p - config.trader_set.market_fee - 1))

                        # ------ tr_threshold ------ #
                    if config.loc_set.point2.short_tr_thresh != "None":
                        if open_side == OrderSide.SELL:
                            if tr_ < config.loc_set.point2.short_tr_thresh:
                                if show_detail:
                                    print("tr_threshold : continue")
                                continue
                        else:
                            if tr_ < config.loc_set.point2.long_tr_thresh:
                                if show_detail:
                                    print("tr_threshold : continue")
                                continue

            if not allow_exit:  # p1_hlm 의 경우, 한번 out 되면 price 가 wave_range 에 닿기전까지 retrade 를 허용하지 않는다. (expiry_p1 을 이용해 op_idx1 을 변경할 것)  
                if show_detail:
                    print("allow_exit = {} : continue".format(allow_exit))
                continue

            if check_hlm in [0, 1]:
                tr_ = tr_arr[op_idx1]

            # ------ leverage ------ #
            # out = out_arr[out_j]  # lvrg_set use out on out_j (out_j shoud be based on p2)
            leverage = lvrg_set(res_df, config, open_side, en_p, out_, fee)  # res_df 변수 사용됨 - 주석 처리 된 상태일뿐
            if leverage is None:
                if show_detail:
                    print("leverage is None : continue")
                if check_hlm:
                    continue  # change op_idx2
                else:
                    break  # change op_idx1

            exit_done, cross_on = 0, 0
            # ------ check tpout_onexec ------ #
            # if not config.ep_set.static_ep and config.ep_set.entry_type == "LIMIT" and config.ep_set.tpout_onexec:
            if config.ep_set.entry_type == "LIMIT":
                if config.tp_set.tp_onexec:  # dynamic 은 tp_onexec 사용하는 의미가 없음
                    tp_j = exec_j
                if config.out_set.out_onexec:  # dynamic 은 out_onexec 사용하는 의미가 없음
                    out_j = exec_j

            # ============ exit loop ============ #
            while 1:
                if not config.tp_set.static_tp:  # 앞으로 왠만하면 static 만 사용할 예정
                    tp_j = i
                if not config.out_set.static_out:
                    out_j = i

                # ------------ out ------------ #  # out 우선 (보수적 검증)
                # ------ signal_out ------ #
                if not exit_done:
                    exit_done, cross_on, ex_p, fee = check_signal_out_v3(res_df, config, open_i2, i, len_df, fee, open_side, cross_on, exit_done,
                                                                         [*ohlc_list, np_timeidx])
                # ------ hl_out ------ #
                if config.out_set.hl_out:
                    if not exit_done:  # and i != len_df - 1:
                        exit_done, ex_p, fee = check_hl_out_v2(config, i, out_j, len_df, fee, open_side, exit_done, [*ohlc_list, out_arr])

                # ------------ tp ------------ #
                if not config.tp_set.non_tp and i != exec_j:
                    if not exit_done:
                        exit_done, ex_p, fee = check_limit_tp_exec(res_df, config, open_i2, i, tp_j, len_df, fee, open_side, exit_done,
                                                                   [*ohlc_list, [tp_arr]])  # 여기서는 j -> i 로 변경해야함
                        # if config.tp_set.tp_type in ['LIMIT']:  # 'BOTH' -> 앞으로는, LIMIT 밖에 없을거라 주석처리함
                        # if not exit_done and config.tp_set.tp_type in ['MARKET', 'BOTH']:

                if exit_done:  # 이 phase 는 exit_phase 뒤에도 있어야할 것 - entry_done var. 사용은 안하겠지만
                    # ------ append dynamic vars. ------ #
                    p1_idx_list.append(op_idx1)  # side, zone, start_ver arr 모두 openi_list 로 접근하기 위해 open_i 를 담음
                    p2_idx_list.append(op_idx2)
                    pair_idx_list.append([exec_j, i])  # entry & exit (체결 기준임)
                    pair_price_list.append([en_p, ex_p])
                    lvrg_list.append(leverage)
                    fee_list.append(fee)
                    tpout_list.append([tp_arr[tp_j], out_arr[out_j]])  # for tpout_line plot_check
                    tr_list.append(tr_)  # Todo, tr vectorize 불가함, 직접 구해주어야할 건데.. (오래걸리지 않을까 --> tr_set 데이터만 모아서 vecto 계산이 나을 것)

                    # open_i += 1  # 다음 open_idx 조사 진행
                    break

                # 1. 아래있으면, 체결 기준부터 tp, out 허용 -> tp 가 entry_idx 에 체결되는게 다소 염려되기는 함, 일단 진행 (그런 case 가 많지 않았으므로)
                # 2. 위에있으면, entry 다음 tick 부터 exit 허용
                i += 1
                if i >= len_df:  # res_df 의 last_index 까지 돌아야함
                    break

            if i >= len_df:  # res_df 의 last_index 까지 돌아야함
                break

            if exit_done == 1:  # tp_done 은 check_hlm 여부와 무관하게 op_idx1 을 변경함
                if show_detail:
                    print("exit_done = {}, i = {} : break".format(exit_done, i))
                break  # change op_idx1
            else:  # exit_done -> -1 or 0 (0 means end of df) 
                if check_hlm in [1, 2]:
                    if check_hlm == 1:  # exit only once in p1_hlm mode
                        allow_exit = 0
                    if show_detail:
                        print("exit_done = {}, i = {} : continue".format(exit_done, i))
                    continue  # change op_idx2
                else:
                    if show_detail:
                        print("exit_done = {}, i = {} : break".format(exit_done, i))
                    break  # change op_idx1    

        # if op_idx1 >= 16355:
        #   break

        if i >= len_df:  # or open_i >= len_open_idx:  # res_df 의 last_index 까지 돌아야함
            break
        else:
            continue

    return np.array(net_p1_idx_list), np.array(p1_idx_list), np.array(p2_idx_list), np.array(pair_idx_list), np.array(pair_price_list), np.array(
        lvrg_list), np.array(
        fee_list), np.array(tpout_list), np.array(tr_list)

def en_ex_pairing_v9_3(res_df, open_idx_list, open_info_list, ohlc_list, config_list, np_timeidx, funcs, show_detail=False):  # 이미 충분히 줄여놓은 idx 임

    open_info1, open_info2 = open_info_list
    side_arr1, _, _, id_idx_arr1 = open_info1
    side_arr2, _, _, _ = open_info2

    expiry_p1, expiry_p2, lvrg_set = funcs

    net_p1_idx_list, p1_idx_list, p2_idx_list, pair_idx_list, pair_price_list, lvrg_list, fee_list, tpout_list, tr_list = [[] for li in range(9)]
    len_df = len(res_df)

    open, high, low, close = ohlc_list

    open_idx1, open_idx2 = open_idx_list
    len_open_idx1 = len(open_idx1)
    len_open_idx2 = len(open_idx2)
    i, open_i1, open_i2 = 0, -1, -1  # i for total_res_df indexing

    while 1:  # for p1's loop

        # Todo,
        #   1. (갱신) p1's open_i + 1 과 op_idx 를 꺼내오는 건, eik1 또는 tp 체결의 경우만 해당됨,
        #   2. out 의 경우 p2's op_idx 기준으로 retry 필요
        #     a. 또한, p2's op_idx > p1's op_idx

        # ============ get p1_info ============ #
        # if eik1 or tp_done or first loop:
        open_i1 += 1  # 확인 끝났으면 조기 이탈(+1), 다음 open_idx 조사 진행
        if open_i1 >= len_open_idx1:
            break

        if show_detail:
            print("open_i1 :", open_i1, side_arr1[open_i1])

        op_idx1 = open_idx1[open_i1]  # open_i1 는 i 와 별개로 운영
        if op_idx1 < i:  # i = 이전 거래 끝난후의 res_df index - "거래 종료후 거래 시작", '<' : 거래 종료시점 진입 가능하다는 의미
            continue

        # ------ set loop index i ------ #
        i = op_idx1  # + 1 --> op_idx1 = op_idx2 가능함 # open_signal 이 close_bar.shift(1) 이라고 가정하고 다음 bar 부터 체결확인한다는 의미
        if i >= len_df:  # res_df 의 last_index 까지 돌아야함
            break

            # ------ dynamic arr info by ID ------ #
        #     1. 해당 id 로 config 재할당해야함
        #       a. use open_i1
        open_side = side_arr1[open_i1]
        id_idx = id_idx_arr1.astype(int)[open_i1]
        config = config_list[id_idx]
        selection_id = config.selection_id
        check_hlm = config.tr_set.check_hlm

        # check_net_hhm = 1 if (config.tr_set.wave_itv1 == config.tr_set.wave_itv2) and (config.tr_set.wave_period1 == config.tr_set.wave_period2) else 0

        side_pos = 'short' if open_side == OrderSide.SELL else 'long'
        if show_detail:
            print("============ op_idx1 : {} {} ============".format(op_idx1, open_side))

        # if show_detail:
        #   print("check_hlm :", check_hlm)

        # ------ load tr_data ------ #
        tp_arr = res_df['{}_tp_{}'.format(side_pos, selection_id)].to_numpy()
        ep1_arr = res_df['{}_ep1_{}'.format(side_pos, selection_id)].to_numpy()
        ep2_arr = res_df['{}_ep2_{}'.format(side_pos, selection_id)].to_numpy()
        out_arr = res_df['{}_out_{}'.format(side_pos, selection_id)].to_numpy()

        tr_arr = res_df['{}_tr_{}'.format(side_pos, selection_id)].to_numpy()  # just for p1_hhm

        tp_1_ = res_df['{}_tp_1_{}'.format(side_pos, selection_id)].to_numpy()[op_idx1]  # for p2_box location & p1's exipiry
        tp_0_ = res_df['{}_tp_0_{}'.format(side_pos, selection_id)].to_numpy()[op_idx1]
        tp_gap_ = res_df['{}_tp_gap_{}'.format(side_pos, selection_id)].to_numpy()[op_idx1]

        # if not check_net_hhm:  # this phase exist for p1 entry (net hhm sync.) in p2_platform
        exec_j, ep_j, tp_j, out_j, entry_done, en_p, fee = check_entry_v6(res_df, config, config.ep_set.entry_type, op_idx1, tp_1_, tp_gap_, len_df,
                                                                          open_side,
                                                                          [*ohlc_list, ep1_arr], expiry_p2)
        i = exec_j  # = entry_loop 를 돌고 나온 e_j
        if not entry_done:
            if show_detail:
                print("p1's expiry by expiry_p2 function in p1's loop : continue")
            continue
            # else:
        #   tp_j = op_idx1

        prev_open_i2 = open_i2
        net_p1_idx_list.append(op_idx1)
        # if check_hlm in [0, 1]:
        #   i = op_idx1  # allow op_idx2 = op_idx1
        allow_exit = 1
        # ============ entry loop ============ #
        while 1:  # for p2's loop (allow retry)

            # ============ get p2_info ============ #
            if check_hlm in [1, 2]:
                open_i2 += 1  # 확인 끝났으면 조기 이탈(+1), 다음 open_idx 조사 진행
                if open_i2 >= len_open_idx2:  # open_i2 소진
                    break

                if show_detail:
                    print("open_i2 :", open_i2, side_arr2[open_i2])

                # ------ check side sync. ------ #
                if open_side != side_arr2[open_i2]:
                    continue

                # ------ assert, op_idx2 >= exec_j ------ #
                op_idx2 = open_idx2[open_i2]  # open_i2 는 i 와 별개로 운영
                if check_hlm == 1 and allow_exit:
                    if op_idx2 < op_idx1:
                        continue
                else:
                    if op_idx2 < i:  # p1 execution 이후의 i 를 허용 (old, 이곳 i = op_idx1 + 1 or p2's exec_j or exit_loop's i + 1)
                        continue

                if check_hlm == 2:
                    i = op_idx2 + 1  # open_signal 이 close_bar.shift(1) 이라고 가정하고 다음 bar 부터 체결확인한다는 의미
                    if i >= len_df:  # res_df 의 last_index 까지 돌아야함
                        break

                if show_detail:
                    print("op_idx1, op_idx2 :", op_idx1, op_idx2, side_arr2[open_i2])

            else:
                op_idx2 = op_idx1

            tp_ = tp_arr[op_idx1]
            ep2_ = ep2_arr[op_idx2]
            out_ = out_arr[op_idx2]

            out_1_ = res_df['{}_out_1_{}'.format(side_pos, selection_id)].to_numpy()[op_idx2]
            out_0_ = res_df['{}_out_0_{}'.format(side_pos, selection_id)].to_numpy()[op_idx2]
            out_gap_ = res_df['{}_out_gap_{}'.format(side_pos, selection_id)].to_numpy()[op_idx2]

            if check_hlm in [1, 2]:
                # ------ check p1's expiry - Todo, priority ------ # - p2_box 생성 이전의 hl_survey
                # 1. op_idx1 ~ op_idx2 까지의 hl_check
                # if check_hlm:  # p1_hlm, p2_hlm --> Todo, 이거를 왜 p1_hlm 에도 적용했는지 잘 모르겠음
                if op_idx1 < op_idx2:
                    expire, touch_idx = expiry_p1(res_df, config, op_idx1, op_idx2, tp_1_, tp_0_, tp_gap_, ohlc_list[1:3], open_side)
                    if expire:  # p1's expiry
                        if show_detail:
                            print("expiry_p1, touch_idx = {} : break".format(touch_idx))
                        i = touch_idx  # + 1  --> 이거 아닌것 같음 # op_idx1 과 op_idx2 사이의 op_idx1' 을 살리기 위함, 즉 바로 다음 op_idx1 로 회귀 (건너뛰지 않고)
                        open_i2 = prev_open_i2
                        break  # change op_idx1

                if check_hlm == 2:
                    # ------ p2 point_validation - vectorization unavailable ------ # p1_loop 로 return 되는 정확한 i 를 반환하기 위해서 expiry_p1 에 순서 양보  # Todo, 새로운 tp, ep, out 에 대한 처리 필요 (p1_hlm 사용시)
                    if open_side == OrderSide.SELL:
                        if not (tp_ < ep2_):  # tr_set validation & reject hl_out open_exec.
                            break  # change op_idx1
                        elif not (ep2_ < out_ and close[op_idx2] < out_):
                            if show_detail:
                                print("point validation : continue")
                            continue  # change op_idx2
                    else:
                        if not (tp_ > ep2_):
                            break
                        elif not (ep2_ > out_ and close[op_idx2] > out_):
                            if show_detail:
                                print("point validation : continue")
                            continue

                    # ------ p2_box location ------ #
                    if open_side == OrderSide.SELL:
                        if not ((tp_1_ + tp_gap_ * config.tr_set.p2_box_k1 <= out_1_) and (
                                out_0_ <= tp_0_ - tp_gap_ * config.tr_set.p2_box_k2)):  # tp1, tp0 에 닿으면 expiry
                            # if not ((tp_1_ + tp_gap_ * config.tr_set.p2_box_k1 >= out_1_) and (out_0_ <= tp_0_ - tp_gap_ * config.tr_set.p2_box_k2)):  # tp1, tp0 에 닿으면 expiry
                            if show_detail:
                                print("p2_box rejection : continue")
                            continue
                        else:
                            # ------ p1p2_low ------ #
                            if not high[op_idx1:op_idx2 + 1].max() < tp_0_ - tp_gap_ * config.tr_set.p1p2_low:
                                if show_detail:
                                    print("p1p2_low rejection : continue")
                                continue
                    else:
                        if not ((tp_1_ - tp_gap_ * config.tr_set.p2_box_k1 >= out_1_) and (out_0_ >= tp_0_ + tp_gap_ * config.tr_set.p2_box_k2)):
                            # if not ((tp_1_ - tp_gap_ * config.tr_set.p2_box_k1 <= out_1_) and (out_0_ >= tp_0_ + tp_gap_ * config.tr_set.p2_box_k2)):
                            if show_detail:
                                print("p2_box rejection : continue")
                            continue
                        else:
                            # ------ p1p2_low ------ #
                            if not low[op_idx1:op_idx2 + 1].min() > tp_0_ + tp_gap_ * config.tr_set.p1p2_low:
                                if show_detail:
                                    print("p1p2_low rejection : continue")
                                continue

                    # ------ check p2's expiry ------ # - 현재, op_idx2 기준의 ep2_arr 을 사용 중임.
                    exec_j, ep_j, _, out_j, entry_done, en_p, fee = check_entry_v6(res_df, config, config.ep_set.point2.entry_type, op_idx2, tp_1_,
                                                                                   tp_gap_, len_df, open_side,
                                                                                   [*ohlc_list, ep2_arr],
                                                                                   expiry_p2)  # Todo, tp_1 & tp_gap 사용이 맞을 것으로 봄
                    i = exec_j  # = entry_loop 를 돌고 나온 e_j
                    if not entry_done:  # p2's expiry
                        if show_detail:
                            print("expiry_p2, i = {} : continue".format(i))
                        continue  # change op_idx2

                    # ------ devectorized tr_calc ------ # - en_p 에 대해 하는게 맞을 것으로봄
                    if open_side == OrderSide.SELL:
                        tr_ = abs((en_p / tp_ - config.trader_set.limit_fee - 1) / (en_p / out_ - config.trader_set.market_fee - 1))
                    else:
                        tr_ = abs((tp_ / en_p - config.trader_set.limit_fee - 1) / (out_ / en_p - config.trader_set.market_fee - 1))

                        # ------ tr_threshold ------ #
                    if config.loc_set.point2.short_tr_thresh != "None":
                        if open_side == OrderSide.SELL:
                            if tr_ < config.loc_set.point2.short_tr_thresh:
                                if show_detail:
                                    print("tr_threshold : continue")
                                continue
                        else:
                            if tr_ < config.loc_set.point2.long_tr_thresh:
                                if show_detail:
                                    print("tr_threshold : continue")
                                continue

            if not allow_exit:  # p1_hlm 의 경우, 한번 out 되면 price 가 wave_range 에 닿기전까지 retrade 를 허용하지 않는다. (expiry_p1 을 이용해 op_idx1 을 변경할 것)
                if show_detail:
                    print("allow_exit = {} : continue".format(allow_exit))
                continue

            if check_hlm in [0, 1]:
                tr_ = tr_arr[op_idx1]

            # ------ leverage ------ #
            # out = out_arr[out_j]  # lvrg_set use out on out_j (out_j shoud be based on p2)
            leverage = lvrg_set(res_df, config, open_side, en_p, out_, fee)  # res_df 변수 사용됨 - 주석 처리 된 상태일뿐
            if leverage is None:
                if show_detail:
                    print("leverage is None : continue")
                if check_hlm:
                    continue  # change op_idx2
                else:
                    break  # change op_idx1

            exit_done, cross_on = 0, 0
            # ------ check tpout_onexec ------ #
            # if not config.ep_set.static_ep and config.ep_set.entry_type == "LIMIT" and config.ep_set.tpout_onexec:
            if config.ep_set.entry_type == "LIMIT":
                if config.tp_set.tp_onexec:  # dynamic 은 tp_onexec 사용하는 의미가 없음
                    tp_j = exec_j
                if config.out_set.out_onexec:  # dynamic 은 out_onexec 사용하는 의미가 없음
                    out_j = exec_j

            # ============ exit loop ============ #
            while 1:
                if not config.tp_set.static_tp:  # 앞으로 왠만하면 static 만 사용할 예정
                    tp_j = i
                if not config.out_set.static_out:
                    out_j = i

                # ------------ out ------------ #  # out 우선 (보수적 검증)
                # ------ signal_out ------ #
                if not exit_done:
                    exit_done, cross_on, ex_p, fee = check_signal_out_v3(res_df, config, open_i2, i, len_df, fee, open_side, cross_on, exit_done,
                                                                         [*ohlc_list, np_timeidx])
                # ------ hl_out ------ #
                if config.out_set.hl_out:
                    if not exit_done:  # and i != len_df - 1:
                        exit_done, ex_p, fee = check_hl_out_v2(config, i, out_j, len_df, fee, open_side, exit_done, [*ohlc_list, out_arr])

                # ------------ tp ------------ #
                if not config.tp_set.non_tp and i != exec_j:
                    if not exit_done:
                        exit_done, ex_p, fee = check_limit_tp_exec(res_df, config, open_i2, i, tp_j, len_df, fee, open_side, exit_done,
                                                                   [*ohlc_list, [tp_arr]])  # 여기서는 j -> i 로 변경해야함
                        # if config.tp_set.tp_type in ['LIMIT']:  # 'BOTH' -> 앞으로는, LIMIT 밖에 없을거라 주석처리함
                        # if not exit_done and config.tp_set.tp_type in ['MARKET', 'BOTH']:

                if exit_done:  # 이 phase 는 exit_phase 뒤에도 있어야할 것 - entry_done var. 사용은 안하겠지만
                    # ------ append dynamic vars. ------ #
                    p1_idx_list.append(op_idx1)  # side, zone, start_ver arr 모두 openi_list 로 접근하기 위해 open_i 를 담음
                    p2_idx_list.append(op_idx2)
                    pair_idx_list.append([exec_j, i])  # entry & exit (체결 기준임)
                    pair_price_list.append([en_p, ex_p])
                    lvrg_list.append(leverage)
                    fee_list.append(fee)
                    tpout_list.append([tp_arr[tp_j], out_arr[out_j]])  # for tpout_line plot_check
                    tr_list.append(tr_)  # Todo, tr vectorize 불가함, 직접 구해주어야할 건데.. (오래걸리지 않을까 --> tr_set 데이터만 모아서 vecto 계산이 나을 것)

                    # open_i += 1  # 다음 open_idx 조사 진행
                    break

                # 1. 아래있으면, 체결 기준부터 tp, out 허용 -> tp 가 entry_idx 에 체결되는게 다소 염려되기는 함, 일단 진행 (그런 case 가 많지 않았으므로)
                # 2. 위에있으면, entry 다음 tick 부터 exit 허용
                i += 1
                if i >= len_df:  # res_df 의 last_index 까지 돌아야함
                    break

            if i >= len_df:  # res_df 의 last_index 까지 돌아야함
                break

            if exit_done == 1:  # tp_done 은 check_hlm 여부와 무관하게 op_idx1 을 변경함
                if show_detail:
                    print("exit_done = {}, i = {} : break".format(exit_done, i))
                break  # change op_idx1
            else:  # exit_done -> -1 or 0 (0 means end of df)
                if check_hlm in [1, 2]:
                    if check_hlm == 1:  # exit only once in p1_hlm mode
                        allow_exit = 0
                    if show_detail:
                        print("exit_done = {}, i = {} : continue".format(exit_done, i))
                    continue  # change op_idx2
                else:
                    if show_detail:
                        print("exit_done = {}, i = {} : break".format(exit_done, i))
                    break  # change op_idx1

        # if op_idx1 >= 16355:
        #   break

        if i >= len_df:  # or open_i >= len_open_idx:  # res_df 의 last_index 까지 돌아야함
            break
        else:
            continue

    return np.array(net_p1_idx_list), np.array(p1_idx_list), np.array(p2_idx_list), np.array(pair_idx_list), np.array(pair_price_list), np.array(
        lvrg_list), np.array(
        fee_list), np.array(tpout_list), np.array(tr_list)

def en_ex_pairing_v9_2(res_df, open_idx_list, open_info_list, ohlc_list, config_list, np_timeidx, funcs, show_detail=False):  # 이미 충분히 줄여놓은 idx 임

    open_info1, open_info2 = open_info_list
    side_arr1, _, _, id_idx_arr1 = open_info1
    side_arr2, _, _, _ = open_info2

    expiry_p1, expiry_p2, lvrg_set = funcs

    net_p1_idx_list, p1_idx_list, p2_idx_list, pair_idx_list, pair_price_list, lvrg_list, fee_list, tpout_list, tr_list = [[] for li in range(9)]
    len_df = len(res_df)

    open, high, low, close = ohlc_list

    open_idx1, open_idx2 = open_idx_list
    len_open_idx1 = len(open_idx1)
    len_open_idx2 = len(open_idx2)
    i, open_i1, open_i2 = 0, -1, -1  # i for total_res_df indexing

    while 1:  # for p1's loop

        # Todo,
        #   1. (갱신) p1's open_i + 1 과 op_idx 를 꺼내오는 건, eik1 또는 tp 체결의 경우만 해당됨,
        #   2. out 의 경우 p2's op_idx 기준으로 retry 필요
        #     a. 또한, p2's op_idx > p1's op_idx

        # ============ get p1_info ============ #
        # if eik1 or tp_done or first loop:
        open_i1 += 1  # 확인 끝났으면 조기 이탈(+1), 다음 open_idx 조사 진행
        if open_i1 >= len_open_idx1:
            break

        if show_detail:
            print("open_i1 :", open_i1, side_arr1[open_i1])

        op_idx1 = open_idx1[open_i1]  # open_i1 는 i 와 별개로 운영
        if op_idx1 < i:  # i = 이전 거래 끝난후의 res_df index - "거래 종료후 거래 시작", '<' : 거래 종료시점 진입 가능하다는 의미
            continue

        # ------ set loop index i ------ #
        i = op_idx1  # + 1 --> op_idx1 = op_idx2 가능함 # open_signal 이 close_bar.shift(1) 이라고 가정하고 다음 bar 부터 체결확인한다는 의미
        if i >= len_df:  # res_df 의 last_index 까지 돌아야함
            break

            # ------ dynamic arr info by ID ------ #
        #     1. 해당 id 로 config 재할당해야함
        #       a. use open_i1
        open_side = side_arr1[open_i1]
        id_idx = id_idx_arr1.astype(int)[open_i1]
        config = config_list[id_idx]
        selection_id = config.selection_id
        check_hlm = config.tr_set.check_hlm

        # check_net_hhm = 1 if (config.tr_set.wave_itv1 == config.tr_set.wave_itv2) and (config.tr_set.wave_period1 == config.tr_set.wave_period2) else 0

        side_pos = 'short' if open_side == OrderSide.SELL else 'long'
        if show_detail:
            print("============ op_idx1 : {} {} ============".format(op_idx1, open_side))

        # if show_detail:
        #   print("check_hlm :", check_hlm)

        # ------ load tr_data ------ #
        tp_arr = res_df['{}_tp_{}'.format(side_pos, selection_id)].to_numpy()
        ep1_arr = res_df['{}_ep1_{}'.format(side_pos, selection_id)].to_numpy()
        ep2_arr = res_df['{}_ep2_{}'.format(side_pos, selection_id)].to_numpy()
        out_arr = res_df['{}_out_{}'.format(side_pos, selection_id)].to_numpy()

        tr_arr = res_df['{}_tr_{}'.format(side_pos, selection_id)].to_numpy()  # just for p1_hhm

        tp_1_ = res_df['{}_tp_1_{}'.format(side_pos, selection_id)].to_numpy()[op_idx1]  # for p2_box location & p1's exipiry
        tp_0_ = res_df['{}_tp_0_{}'.format(side_pos, selection_id)].to_numpy()[op_idx1]
        tp_gap_ = res_df['{}_tp_gap_{}'.format(side_pos, selection_id)].to_numpy()[op_idx1]

        # if not check_net_hhm:  # this phase exist for p1 entry (net hhm sync.) in p2_platform
        exec_j, ep_j, tp_j, out_j, entry_done, en_p, fee = check_entry_v6(res_df, config, config.ep_set.entry_type, op_idx1, tp_1_, tp_gap_, len_df,
                                                                          open_side,
                                                                          [*ohlc_list, ep1_arr], expiry_p2)
        i = exec_j  # = entry_loop 를 돌고 나온 e_j
        if not entry_done:
            if show_detail:
                print("p1's expiry by expiry_p2 function in p1's loop : continue")
            continue
            # else:
        #   tp_j = op_idx1

        prev_open_i2 = open_i2
        net_p1_idx_list.append(op_idx1)
        # if check_hlm in [0, 1]:
        #   i = op_idx1  # allow op_idx2 = op_idx1
        # allow_exit = 1
        # ============ entry loop ============ #
        while 1:  # for p2's loop (allow retry)

            # ============ get p2_info ============ #
            if check_hlm == 2:
                open_i2 += 1  # 확인 끝났으면 조기 이탈(+1), 다음 open_idx 조사 진행
                if open_i2 >= len_open_idx2:  # open_i2 소진
                    break

                if show_detail:
                    print("open_i2 :", open_i2, side_arr2[open_i2])

                # ------ check side sync. ------ #
                if open_side != side_arr2[open_i2]:
                    continue

                # ------ assert, op_idx2 >= exec_j ------ #
                op_idx2 = open_idx2[open_i2]  # open_i2 는 i 와 별개로 운영
                if op_idx2 < i:  # p1 execution 이후의 i 를 허용 (old, 이곳 i = op_idx1 + 1 or p2's exec_j or exit_loop's i + 1)
                    continue

                if show_detail:
                    print("op_idx1, op_idx2 :", op_idx1, op_idx2, side_arr2[open_i2])

                i = op_idx2 + 1  # open_signal 이 close_bar.shift(1) 이라고 가정하고 다음 bar 부터 체결확인한다는 의미
                if i >= len_df:  # res_df 의 last_index 까지 돌아야함
                    break

            else:
                op_idx2 = op_idx1

            tp_ = tp_arr[op_idx1]
            ep2_ = ep2_arr[op_idx2]
            out_ = out_arr[op_idx2]

            out_1_ = res_df['{}_out_1_{}'.format(side_pos, selection_id)].to_numpy()[op_idx2]
            out_0_ = res_df['{}_out_0_{}'.format(side_pos, selection_id)].to_numpy()[op_idx2]
            out_gap_ = res_df['{}_out_gap_{}'.format(side_pos, selection_id)].to_numpy()[op_idx2]

            if check_hlm == 2:
                # ------ check p1's expiry - Todo, priority ------ # - p2_box 생성 이전의 hl_survey
                # 1. op_idx1 ~ op_idx2 까지의 hl_check
                # if check_hlm:  # p1_hlm, p2_hlm --> Todo, 이거를 왜 p1_hlm 에도 적용했는지 잘 모르겠음
                if op_idx1 < op_idx2:
                    expire, touch_idx = expiry_p1(res_df, config, op_idx1, op_idx2, tp_1_, tp_0_, tp_gap_, ohlc_list[1:3], open_side)
                    if expire:  # p1's expiry
                        if show_detail:
                            print("expiry_p1, touch_idx = {} : break".format(touch_idx))
                        i = touch_idx  # + 1  --> 이거 아닌것 같음 # op_idx1 과 op_idx2 사이의 op_idx1' 을 살리기 위함, 즉 바로 다음 op_idx1 로 회귀 (건너뛰지 않고)
                        open_i2 = prev_open_i2
                        break  # change op_idx1

                # ------ p2 point_validation - vectorization unavailable ------ # p1_loop 로 return 되는 정확한 i 를 반환하기 위해서 expiry_p1 에 순서 양보  # Todo, 새로운 tp, ep, out 에 대한 처리 필요 (p1_hlm 사용시)
                if open_side == OrderSide.SELL:
                    if not (tp_ < ep2_):  # tr_set validation & reject hl_out open_exec.
                        break  # change op_idx1
                    elif not (ep2_ < out_ and close[op_idx2] < out_):
                        if show_detail:
                            print("point validation : continue")
                        continue  # change op_idx2
                else:
                    if not (tp_ > ep2_):
                        break
                    elif not (ep2_ > out_ and close[op_idx2] > out_):
                        if show_detail:
                            print("point validation : continue")
                        continue

                # ------ p2_box location ------ #
                if open_side == OrderSide.SELL:
                    # if not ((tp_1_ + tp_gap_ * config.tr_set.p2_box_k1 <= out_1_) and (out_0_ <= tp_0_ - tp_gap_ * config.tr_set.p2_box_k2)):  # tp1, tp0 에 닿으면 expiry
                    if not ((tp_1_ + tp_gap_ * config.tr_set.p2_box_k1 >= out_1_) and (
                            out_0_ <= tp_0_ - tp_gap_ * config.tr_set.p2_box_k2)):  # tp1, tp0 에 닿으면 expiry
                        if show_detail:
                            print("p2_box rejection : continue")
                        continue
                    else:
                        # ------ p1p2_low ------ #
                        if not high[op_idx1:op_idx2 + 1].max() < tp_0_ - tp_gap_ * config.tr_set.p1p2_low:
                            if show_detail:
                                print("p1p2_low rejection : continue")
                            continue
                else:
                    # if not ((tp_1_ - tp_gap_ * config.tr_set.p2_box_k1 >= out_1_) and (out_0_ >= tp_0_ + tp_gap_ * config.tr_set.p2_box_k2)):
                    if not ((tp_1_ - tp_gap_ * config.tr_set.p2_box_k1 <= out_1_) and (out_0_ >= tp_0_ + tp_gap_ * config.tr_set.p2_box_k2)):
                        if show_detail:
                            print("p2_box rejection : continue")
                        continue
                    else:
                        # ------ p1p2_low ------ #
                        if not low[op_idx1:op_idx2 + 1].min() > tp_0_ + tp_gap_ * config.tr_set.p1p2_low:
                            if show_detail:
                                print("p1p2_low rejection : continue")
                            continue

                # ------ check p2's expiry ------ #
                exec_j, ep_j, _, out_j, entry_done, en_p, fee = check_entry_v6(res_df, config, config.ep_set.point2.entry_type, op_idx2, tp_1_,
                                                                               tp_gap_, len_df, open_side,
                                                                               [*ohlc_list, ep2_arr], expiry_p2)  # Todo, tp_1 & tp_gap 사용이 맞을 것으로 봄
                i = exec_j  # = entry_loop 를 돌고 나온 e_j
                if not entry_done:  # p2's expiry
                    if show_detail:
                        print("expiry_p2, i = {} : continue".format(i))
                    continue  # change op_idx2

                # ------ devectorized tr_calc ------ #   # en_p 에 대해 하는게 맞을 것으로봄
                if open_side == OrderSide.SELL:
                    tr_ = abs((en_p / tp_ - config.trader_set.limit_fee - 1) / (en_p / out_ - config.trader_set.market_fee - 1))
                else:
                    tr_ = abs((tp_ / en_p - config.trader_set.limit_fee - 1) / (out_ / en_p - config.trader_set.market_fee - 1))

                # ------ tr_threshold ------ #
                if config.loc_set.point2.short_tr_thresh != "None":
                    if open_side == OrderSide.SELL:
                        if tr_ < config.loc_set.point2.short_tr_thresh:
                            if show_detail:
                                print("tr_threshold : continue")
                            continue
                    else:
                        if tr_ < config.loc_set.point2.long_tr_thresh:
                            if show_detail:
                                print("tr_threshold : continue")
                            continue

            else:
                tr_ = tr_arr[op_idx1]

            # ------ leverage ------ #
            # out = out_arr[out_j]  # lvrg_set use out on out_j (out_j shoud be based on p2)
            leverage = lvrg_set(res_df, config, open_side, en_p, out_, fee)  # res_df 변수 사용됨 - 주석 처리 된 상태일뿐
            if leverage is None:
                if show_detail:
                    print("leverage is None : continue")
                if check_hlm:
                    continue  # change op_idx2
                else:
                    break  # change op_idx1

            exit_done, cross_on = 0, 0
            # ------ check tpout_onexec ------ #
            # if not config.ep_set.static_ep and config.ep_set.entry_type == "LIMIT" and config.ep_set.tpout_onexec:
            if config.ep_set.entry_type == "LIMIT":
                if config.tp_set.tp_onexec:  # dynamic 은 tp_onexec 사용하는 의미가 없음
                    tp_j = exec_j
                if config.out_set.out_onexec:  # dynamic 은 out_onexec 사용하는 의미가 없음
                    out_j = exec_j

            # ============ exit loop ============ # --> p1_hlm 의 경우, 1번만 실행
            # if not allow_exit:
            #   continue

            while 1:
                if not config.tp_set.static_tp:  # 앞으로 왠만하면 static 만 사용할 예정
                    tp_j = i
                if not config.out_set.static_out:
                    out_j = i

                # ------------ out ------------ #  # out 우선 (보수적 검증)
                # ------ signal_out ------ #
                if not exit_done:
                    exit_done, cross_on, ex_p, fee = check_signal_out_v3(res_df, config, open_i2, i, len_df, fee, open_side, cross_on, exit_done,
                                                                         [*ohlc_list, np_timeidx])
                # ------ hl_out ------ #
                if config.out_set.hl_out != "None":
                    if not exit_done:  # and i != len_df - 1:
                        exit_done, ex_p, fee = check_hl_out_v2(config, i, out_j, len_df, fee, open_side, exit_done, [*ohlc_list, out_arr])

                # ------------ tp ------------ #
                if not config.tp_set.non_tp and i != exec_j:
                    if not exit_done:
                        exit_done, ex_p, fee = check_limit_tp_exec(res_df, config, open_i2, i, tp_j, len_df, fee, open_side, exit_done,
                                                                   [*ohlc_list, [tp_arr]])  # 여기서는 j -> i 로 변경해야함
                        # if config.tp_set.tp_type in ['LIMIT']:  # 'BOTH' -> 앞으로는, LIMIT 밖에 없을거라 주석처리함
                        # if not exit_done and config.tp_set.tp_type in ['MARKET', 'BOTH']:

                if exit_done:  # 이 phase 는 exit_phase 뒤에도 있어야할 것 - entry_done var. 사용은 안하겠지만
                    # ------ append dynamic vars. ------ #
                    p1_idx_list.append(op_idx1)  # side, zone, start_ver arr 모두 openi_list 로 접근하기 위해 open_i 를 담음
                    p2_idx_list.append(op_idx2)
                    pair_idx_list.append([exec_j, i])  # entry & exit (체결 기준임)
                    pair_price_list.append([en_p, ex_p])
                    lvrg_list.append(leverage)
                    fee_list.append(fee)
                    tpout_list.append([tp_arr[tp_j], out_arr[out_j]])  # for tpout_line plot_check
                    tr_list.append(tr_)  # Todo, tr vectorize 불가함, 직접 구해주어야할 건데.. (오래걸리지 않을까 --> tr_set 데이터만 모아서 vecto 계산이 나을 것)

                    # open_i += 1  # 다음 open_idx 조사 진행
                    break

                # 1. 아래있으면, 체결 기준부터 tp, out 허용 -> tp 가 entry_idx 에 체결되는게 다소 염려되기는 함, 일단 진행 (그런 case 가 많지 않았으므로)
                # 2. 위에있으면, entry 다음 tick 부터 exit 허용
                i += 1
                if i >= len_df:  # res_df 의 last_index 까지 돌아야함
                    break

            if i >= len_df:  # res_df 의 last_index 까지 돌아야함
                break

            if exit_done == 1:  # tp_done
                if show_detail:
                    print("exit_done = {}, i = {} : break".format(exit_done, i))
                break  # change op_idx1
            else:  # exit_done -> -1 or 0 (0 means end of df)
                if check_hlm == 2:
                    # if check_hlm == 1:   # exit only once in p1_hlm mode
                    #   allow_exit = 0
                    if show_detail:
                        print("exit_done = {}, i = {} : continue".format(exit_done, i))
                    continue  # change op_idx2
                else:
                    if show_detail:
                        print("exit_done = {}, i = {} : break".format(exit_done, i))
                    break  # change op_idx1

        # if op_idx1 >= 16355:
        #   break

        if i >= len_df:  # or open_i >= len_open_idx:  # res_df 의 last_index 까지 돌아야함
            break
        else:
            continue

    return np.array(net_p1_idx_list), np.array(p1_idx_list), np.array(p2_idx_list), np.array(pair_idx_list), np.array(pair_price_list), np.array(
        lvrg_list), np.array(
        fee_list), np.array(tpout_list), np.array(tr_list)

def en_ex_pairing_v9_1(res_df, open_idx_list, open_info_list, ohlc_list, config_list, np_timeidx, funcs, show_detail=False):  # 이미 충분히 줄여놓은 idx 임

    open_info1, open_info2 = open_info_list
    side_arr1, _, _, id_idx_arr1 = open_info1
    side_arr2, _, _, _ = open_info2

    expiry_p1, expiry_p2, lvrg_set = funcs

    net_p1_idx_list, p1_idx_list, p2_idx_list, pair_idx_list, pair_price_list, lvrg_list, fee_list, tpout_list, tr_list = [[] for li in range(9)]
    len_df = len(res_df)

    open, high, low, close = ohlc_list

    open_idx1, open_idx2 = open_idx_list
    len_open_idx1 = len(open_idx1)
    len_open_idx2 = len(open_idx2)
    i, open_i1, open_i2 = 0, -1, -1  # i for total_res_df indexing

    while 1:  # for p1's loop

        # Todo,
        #   1. (갱신) p1's open_i + 1 과 op_idx 를 꺼내오는 건, eik1 또는 tp 체결의 경우만 해당됨,
        #   2. out 의 경우 p2's op_idx 기준으로 retry 필요
        #     a. 또한, p2's op_idx > p1's op_idx

        # ============ get p1_info ============ #
        # if eik1 or tp_done or first loop:
        open_i1 += 1  # 확인 끝났으면 조기 이탈(+1), 다음 open_idx 조사 진행
        if open_i1 >= len_open_idx1:
            break

        if show_detail:
            print("open_i1 :", open_i1, side_arr1[open_i1])

        op_idx1 = open_idx1[open_i1]  # open_i1 는 i 와 별개로 운영
        if op_idx1 < i:  # i = 이전 거래 끝난후의 res_df index - "거래 종료후 거래 시작", '<' : 거래 종료시점 진입 가능하다는 의미
            continue

        # ------ set loop index i ------ #
        i = op_idx1  # + 1 --> op_idx1 = op_idx2 가능함 # open_signal 이 close_bar.shift(1) 이라고 가정하고 다음 bar 부터 체결확인한다는 의미
        if i >= len_df:  # res_df 의 last_index 까지 돌아야함
            break

            # ------ dynamic arr info by ID ------ #
        #     1. 해당 id 로 config 재할당해야함
        #       a. use open_i1
        open_side = side_arr1[open_i1]
        id_idx = id_idx_arr1.astype(int)[open_i1]
        config = config_list[id_idx]
        selection_id = config.selection_id
        check_hlm = config.tr_set.check_hlm

        # check_net_hhm = 1 if (config.tr_set.wave_itv1 == config.tr_set.wave_itv2) and (config.tr_set.wave_period1 == config.tr_set.wave_period2) else 0

        side_pos = 'short' if open_side == OrderSide.SELL else 'long'
        if show_detail:
            print("============ op_idx1 : {} {} ============".format(op_idx1, open_side))

        # if show_detail:
        #   print("check_hlm :", check_hlm)

        # ------ load tr_data ------ #
        tp_arr = res_df['{}_tp_{}'.format(side_pos, selection_id)].to_numpy()
        ep1_arr = res_df['{}_ep1_{}'.format(side_pos, selection_id)].to_numpy()
        ep2_arr = res_df['{}_ep2_{}'.format(side_pos, selection_id)].to_numpy()
        out_arr = res_df['{}_out_{}'.format(side_pos, selection_id)].to_numpy()

        tr_arr = res_df['{}_tr_{}'.format(side_pos, selection_id)].to_numpy()  # just for p1_hhm

        tp_1_ = res_df['{}_tp_1_{}'.format(side_pos, selection_id)].to_numpy()[op_idx1]  # for p2_box location & p1's exipiry
        tp_0_ = res_df['{}_tp_0_{}'.format(side_pos, selection_id)].to_numpy()[op_idx1]
        tp_gap_ = res_df['{}_tp_gap_{}'.format(side_pos, selection_id)].to_numpy()[op_idx1]

        # if not check_net_hhm:  # this phase exist for p1 entry (net hhm sync.) in p2_platform
        exec_j, ep_j, tp_j, out_j, entry_done, en_p, fee = check_entry_v5(res_df, config, op_idx1, tp_1_, tp_gap_, len_df, open_side,
                                                                          [*ohlc_list, ep1_arr], expiry_p2)
        i = exec_j  # = entry_loop 를 돌고 나온 e_j
        if not entry_done:
            if show_detail:
                print("p1's expiry by expiry_p2 function in p1's loop : continue")
            continue
            # else:
        #   tp_j = op_idx1

        prev_open_i2 = open_i2
        net_p1_idx_list.append(op_idx1)
        # if check_hlm in [0, 1]:
        #   i = op_idx1  # allow op_idx2 = op_idx1
        allow_exit = 1
        # ============ entry loop ============ #
        while 1:  # for p2's loop (allow retry)

            # ============ get p2_info ============ #
            if check_hlm == 2:
                open_i2 += 1  # 확인 끝났으면 조기 이탈(+1), 다음 open_idx 조사 진행
                if open_i2 >= len_open_idx2:  # open_i2 소진
                    break

                if show_detail:
                    print("open_i2 :", open_i2, side_arr2[open_i2])

                # ------ check side sync. ------ #
                if open_side != side_arr2[open_i2]:
                    continue

                # ------ assert, op_idx2 >= exec_j ------ #
                op_idx2 = open_idx2[open_i2]  # open_i2 는 i 와 별개로 운영
                if op_idx2 < i:  # p1 execution 이후의 i 를 허용 (old, 이곳 i = op_idx1 + 1 or p2's exec_j or exit_loop's i + 1)
                    continue

                if show_detail:
                    print("op_idx1, op_idx2 :", op_idx1, op_idx2, side_arr2[open_i2])

                i = op_idx2 + 1  # open_signal 이 close_bar.shift(1) 이라고 가정하고 다음 bar 부터 체결확인한다는 의미
                if i >= len_df:  # res_df 의 last_index 까지 돌아야함
                    break

            else:
                op_idx2 = op_idx1

            tp_ = tp_arr[op_idx1]
            ep2_ = ep2_arr[op_idx2]
            out_ = out_arr[op_idx2]

            out_1_ = res_df['{}_out_1_{}'.format(side_pos, selection_id)].to_numpy()[op_idx2]
            out_0_ = res_df['{}_out_0_{}'.format(side_pos, selection_id)].to_numpy()[op_idx2]
            out_gap_ = res_df['{}_out_gap_{}'.format(side_pos, selection_id)].to_numpy()[op_idx2]

            if check_hlm == 2:
                # ------ check p1's expiry - Todo, priority ------ # - p2_box 생성 이전의 hl_survey
                # 1. op_idx1 ~ op_idx2 까지의 hl_check
                # if check_hlm:  # p1_hlm, p2_hlm --> Todo, 이거를 왜 p1_hlm 에도 적용했는지 잘 모르겠음
                if op_idx1 < op_idx2:
                    expire, touch_idx = expiry_p1(res_df, config, op_idx1, op_idx2, tp_1_, tp_0_, tp_gap_, ohlc_list[1:3], open_side)
                    if expire:  # p1's expiry
                        if show_detail:
                            print("expiry_p1, touch_idx = {} : break".format(touch_idx))
                        i = touch_idx  # + 1  --> 이거 아닌것 같음 # op_idx1 과 op_idx2 사이의 op_idx1' 을 살리기 위함, 즉 바로 다음 op_idx1 로 회귀 (건너뛰지 않고)
                        open_i2 = prev_open_i2
                        break  # change op_idx1

                # ------ p2 point_validation - vectorization unavailable ------ # p1_loop 로 return 되는 정확한 i 를 반환하기 위해서 expiry_p1 에 순서 양보  # Todo, 새로운 tp, ep, out 에 대한 처리 필요 (p1_hlm 사용시)
                if open_side == OrderSide.SELL:
                    if not (tp_ < ep2_):  # tr_set validation & reject hl_out open_exec.
                        break  # change op_idx1
                    elif not (ep2_ < out_ and close[op_idx2] < out_):
                        if show_detail:
                            print("point validation : continue")
                        continue  # change op_idx2
                else:
                    if not (tp_ > ep2_):
                        break
                    elif not (ep2_ > out_ and close[op_idx2] > out_):
                        if show_detail:
                            print("point validation : continue")
                        continue

                # ------ p2_box location ------ #
                if open_side == OrderSide.SELL:
                    if not ((tp_1_ + tp_gap_ * config.tr_set.p2_box_k1 <= out_1_) and (
                            out_0_ <= tp_0_ - tp_gap_ * config.tr_set.p2_box_k2)):  # tp1, tp0 에 닿으면 expiry
                        if show_detail:
                            print("p2_box rejection : continue")
                        continue
                    else:
                        # ------ p1p2_low ------ #
                        if not high[op_idx1:op_idx2 + 1].max() < tp_0_ - tp_gap_ * config.tr_set.p1p2_low:
                            if show_detail:
                                print("p1p2_low rejection : continue")
                            continue
                else:
                    if not ((tp_1_ - tp_gap_ * config.tr_set.p2_box_k1 >= out_1_) and (out_0_ >= tp_0_ + tp_gap_ * config.tr_set.p2_box_k2)):
                        if show_detail:
                            print("p2_box rejection : continue")
                        continue
                    else:
                        # ------ p1p2_low ------ #
                        if not low[op_idx1:op_idx2 + 1].min() > tp_0_ + tp_gap_ * config.tr_set.p1p2_low:
                            if show_detail:
                                print("p1p2_low rejection : continue")
                            continue

                # ------ check p2's expiry ------ #
                exec_j, ep_j, _, out_j, entry_done, en_p, fee = check_entry_v5(res_df, config, op_idx2, tp_1_, tp_gap_, len_df, open_side,
                                                                               [*ohlc_list, ep2_arr], expiry_p2)  # tp_j 는 op_idx1 사용
                i = exec_j  # = entry_loop 를 돌고 나온 e_j
                if not entry_done:  # p2's expiry
                    if show_detail:
                        print("expiry_p2, i = {} : continue".format(i))
                    continue  # change op_idx2

                # ------ devectorized tr_calc ------ #   # en_p 에 대해 하는게 맞을 것으로봄
                if open_side == OrderSide.SELL:
                    tr_ = abs((en_p / tp_ - config.trader_set.limit_fee - 1) / (en_p / out_ - config.trader_set.market_fee - 1))
                else:
                    tr_ = abs((tp_ / en_p - config.trader_set.limit_fee - 1) / (out_ / en_p - config.trader_set.market_fee - 1))

            else:
                tr_ = tr_arr[op_idx1]

            # ------ tr_threshold ------ #
            if config.loc_set.point1.short_tr_thresh != "None":
                if open_side == OrderSide.SELL:
                    if tr_ < config.loc_set.point1.short_tr_thresh:
                        print("tr_threshold : continue")
                        continue
                else:
                    if tr_ < config.loc_set.point1.long_tr_thresh:
                        print("tr_threshold : continue")
                        continue

            # ------ leverage ------ #
            # out = out_arr[out_j]  # lvrg_set use out on out_j (out_j shoud be based on p2)
            leverage = lvrg_set(res_df, config, open_side, en_p, out_, fee)  # res_df 변수 사용됨 - 주석 처리 된 상태일뿐
            if leverage is None:
                if show_detail:
                    print("leverage is None : continue")
                if check_hlm:
                    continue  # change op_idx2
                else:
                    break  # change op_idx1

            exit_done, cross_on = 0, 0
            # ------ check tpout_onexec ------ #
            # if not config.ep_set.static_ep and config.ep_set.entry_type == "LIMIT" and config.ep_set.tpout_onexec:
            if config.ep_set.entry_type == "LIMIT":
                if config.tp_set.tp_onexec:  # dynamic 은 tp_onexec 사용하는 의미가 없음
                    tp_j = exec_j
                if config.out_set.out_onexec:  # dynamic 은 out_onexec 사용하는 의미가 없음
                    out_j = exec_j

            # ============ exit loop ============ # --> p1_hlm 의 경우, 1번만 실행
            if not allow_exit:
                continue

            while 1:
                if not config.tp_set.static_tp:  # 앞으로 왠만하면 static 만 사용할 예정
                    tp_j = i
                if not config.out_set.static_out:
                    out_j = i

                # ------------ out ------------ #  # out 우선 (보수적 검증)
                # ------ signal_out ------ #
                if not exit_done:
                    exit_done, cross_on, ex_p, fee = check_signal_out_v3(res_df, config, open_i2, i, len_df, fee, open_side, cross_on, exit_done,
                                                                         [*ohlc_list, np_timeidx])
                # ------ hl_out ------ #
                if config.out_set.hl_out != "None":
                    if not exit_done:  # and i != len_df - 1:
                        exit_done, ex_p, fee = check_hl_out_v2(config, i, out_j, len_df, fee, open_side, exit_done, [*ohlc_list, out_arr])

                # ------------ tp ------------ #
                if not config.tp_set.non_tp and i != exec_j:
                    if not exit_done:
                        exit_done, ex_p, fee = check_limit_tp_exec(res_df, config, open_i2, i, tp_j, len_df, fee, open_side, exit_done,
                                                                   [*ohlc_list, [tp_arr]])  # 여기서는 j -> i 로 변경해야함
                        # if config.tp_set.tp_type in ['LIMIT']:  # 'BOTH' -> 앞으로는, LIMIT 밖에 없을거라 주석처리함
                        # if not exit_done and config.tp_set.tp_type in ['MARKET', 'BOTH']:

                if exit_done:  # 이 phase 는 exit_phase 뒤에도 있어야할 것 - entry_done var. 사용은 안하겠지만
                    # ------ append dynamic vars. ------ #
                    p1_idx_list.append(op_idx1)  # side, zone, start_ver arr 모두 openi_list 로 접근하기 위해 open_i 를 담음
                    p2_idx_list.append(op_idx2)
                    pair_idx_list.append([exec_j, i])  # entry & exit (체결 기준임)
                    pair_price_list.append([en_p, ex_p])
                    lvrg_list.append(leverage)
                    fee_list.append(fee)
                    tpout_list.append([tp_arr[tp_j], out_arr[out_j]])  # for tpout_line plot_check
                    tr_list.append(tr_)  # Todo, tr vectorize 불가함, 직접 구해주어야할 건데.. (오래걸리지 않을까 --> tr_set 데이터만 모아서 vecto 계산이 나을 것)

                    # open_i += 1  # 다음 open_idx 조사 진행
                    break

                # 1. 아래있으면, 체결 기준부터 tp, out 허용 -> tp 가 entry_idx 에 체결되는게 다소 염려되기는 함, 일단 진행 (그런 case 가 많지 않았으므로)
                # 2. 위에있으면, entry 다음 tick 부터 exit 허용
                i += 1
                if i >= len_df:  # res_df 의 last_index 까지 돌아야함
                    break

            if i >= len_df:  # res_df 의 last_index 까지 돌아야함
                break

            if exit_done == 1:  # tp_done
                if show_detail:
                    print("exit_done = {}, i = {} : break".format(exit_done, i))
                break  # change op_idx1
            else:
                if check_hlm:
                    if check_hlm == 1:  # exit only once in p1_hlm mode
                        allow_exit = 0
                    if show_detail:
                        print("exit_done = {}, i = {} : continue".format(exit_done, i))
                    continue  # change op_idx2
                else:
                    if show_detail:
                        print("exit_done = {}, i = {} : break".format(exit_done, i))
                    break  # change op_idx1

        # if op_idx1 >= 16355:
        #   break

        if i >= len_df:  # or open_i >= len_open_idx:  # res_df 의 last_index 까지 돌아야함
            break
        else:
            continue

    return np.array(net_p1_idx_list), np.array(p1_idx_list), np.array(p2_idx_list), np.array(pair_idx_list), np.array(pair_price_list), np.array(
        lvrg_list), np.array(
        fee_list), np.array(tpout_list), np.array(tr_list)


def en_ex_pairing_v9(res_df, open_idx_list, open_info_list, ohlc_list, config_list, np_timeidx, funcs, show_detail=False):  # 이미 충분히 줄여놓은 idx 임

    open_info1, open_info2 = open_info_list
    side_arr1, _, _, id_idx_arr1 = open_info1
    side_arr2, _, _, _ = open_info2

    expiry_p1, expiry_p2, lvrg_set = funcs

    net_p1_idx_list, p1_idx_list, p2_idx_list, pair_idx_list, pair_price_list, lvrg_list, fee_list, tpout_list, tr_list = [[] for li in range(9)]
    len_df = len(res_df)

    open, high, low, close = ohlc_list

    open_idx1, open_idx2 = open_idx_list
    len_open_idx1 = len(open_idx1)
    len_open_idx2 = len(open_idx2)
    i, open_i1, open_i2 = 0, -1, -1  # i for total_res_df indexing

    while 1:  # for p1's loop

        # Todo,
        #   1. (갱신) p1's open_i + 1 과 op_idx 를 꺼내오는 건, eik1 또는 tp 체결의 경우만 해당됨,
        #   2. out 의 경우 p2's op_idx 기준으로 retry 필요
        #     a. 또한, p2's op_idx > p1's op_idx

        # ============ get p1_info ============ #
        # if eik1 or tp_done or first loop:
        open_i1 += 1  # 확인 끝났으면 조기 이탈(+1), 다음 open_idx 조사 진행
        if open_i1 >= len_open_idx1:
            break

        if show_detail:
            print("open_i1 :", open_i1, side_arr1[open_i1])

        op_idx1 = open_idx1[open_i1]  # open_i1 는 i 와 별개로 운영
        if op_idx1 < i:  # i = 이전 거래 끝난후의 res_df index - "거래 종료후 거래 시작", '<' : 거래 종료시점 진입 가능하다는 의미
            continue

        # ------ set loop index i ------ #
        i = op_idx1  # + 1 --> op_idx1 = op_idx2 가능함 # open_signal 이 close_bar.shift(1) 이라고 가정하고 다음 bar 부터 체결확인한다는 의미
        if i >= len_df:  # res_df 의 last_index 까지 돌아야함
            break

        # ------ dynamic arr info by ID ------ #
        #     1. 해당 id 로 config 재할당해야함
        #       a. use open_i1
        open_side = side_arr1[open_i1]
        id_idx = id_idx_arr1.astype(int)[open_i1]
        config = config_list[id_idx]
        selection_id = config.selection_id
        check_hlm = config.tr_set.check_hlm

        # check_net_hhm = 1 if (config.tr_set.wave_itv1 == config.tr_set.wave_itv2) and (config.tr_set.wave_period1 == config.tr_set.wave_period2) else 0

        side_pos = 'short' if open_side == OrderSide.SELL else 'long'
        if show_detail:
            print("============ op_idx1 : {} {} ============".format(op_idx1, open_side))

        # if show_detail:
        #   print("check_hlm :", check_hlm)

        # ------ load tr_data ------ #
        tp_arr = res_df['{}_tp_{}'.format(side_pos, selection_id)].to_numpy()
        ep1_arr = res_df['{}_ep1_{}'.format(side_pos, selection_id)].to_numpy()
        ep2_arr = res_df['{}_ep2_{}'.format(side_pos, selection_id)].to_numpy()
        out_arr = res_df['{}_out_{}'.format(side_pos, selection_id)].to_numpy()

        tr_arr = res_df['{}_tr_{}'.format(side_pos, selection_id)].to_numpy()  # just for p1_hhm

        tp_1_ = res_df['{}_tp_1_{}'.format(side_pos, selection_id)].to_numpy()[op_idx1]  # for p2_box location & p1's exipiry
        tp_0_ = res_df['{}_tp_0_{}'.format(side_pos, selection_id)].to_numpy()[op_idx1]
        tp_gap_ = res_df['{}_tp_gap_{}'.format(side_pos, selection_id)].to_numpy()[op_idx1]

        # if not check_net_hhm:  # this phase exist for p1 entry (net hhm sync.) in p2_platform
        exec_j, ep_j, tp_j, out_j, entry_done, en_p, fee = check_entry_v5(res_df, config, op_idx1, tp_1_, tp_gap_, len_df, open_side,
                                                                          [*ohlc_list, ep1_arr], expiry_p2)
        i = exec_j  # = entry_loop 를 돌고 나온 e_j
        if not entry_done:
            if show_detail:
                print("p1's expiry by expiry_p2 function in p1's loop : continue")
            continue
            # else:
        #   tp_j = op_idx1

        prev_open_i2 = open_i2
        net_p1_idx_list.append(op_idx1)
        if check_hlm in [0, 1]:
            i = op_idx1  # allow op_idx2 = op_idx1
        allow_exit = 1
        # ============ entry loop ============ #
        while 1:  # for p2's loop (allow retry)

            # ============ get p2_info ============ #
            open_i2 += 1  # 확인 끝났으면 조기 이탈(+1), 다음 open_idx 조사 진행
            if open_i2 >= len_open_idx2:  # open_i2 소진
                break

            if show_detail:
                print("open_i2 :", open_i2, side_arr2[open_i2])

            # ------ check side sync. ------ #
            if open_side != side_arr2[open_i2]:
                continue

            op_idx2 = open_idx2[open_i2]  # open_i2 는 i 와 별개로 운영
            if op_idx2 < i:  # p1 execution 이후의 i 를 허용 (old, 이곳 i = op_idx1 + 1 or p2's exec_j or exit_loop's i + 1)
                continue

            if show_detail:
                print("op_idx1, op_idx2 :", op_idx1, op_idx2, side_arr2[open_i2])

            i = op_idx2 + 1  # open_signal 이 close_bar.shift(1) 이라고 가정하고 다음 bar 부터 체결확인한다는 의미
            if i >= len_df:  # res_df 의 last_index 까지 돌아야함
                break

            tp_ = tp_arr[op_idx1]
            ep2_ = ep2_arr[op_idx2]
            out_ = out_arr[op_idx2]

            out_1_ = res_df['{}_out_1_{}'.format(side_pos, selection_id)].to_numpy()[op_idx2]
            out_0_ = res_df['{}_out_0_{}'.format(side_pos, selection_id)].to_numpy()[op_idx2]
            out_gap_ = res_df['{}_out_gap_{}'.format(side_pos, selection_id)].to_numpy()[op_idx2]

            # ------ check p1's expiry - Todo, priority ------ # - p2_box 생성 이전의 hl_survey
            # 1. op_idx1 ~ op_idx2 까지의 hl_check
            if check_hlm:  # p1_hlm, p2_hlm
                if op_idx1 < op_idx2:
                    expire, touch_idx = expiry_p1(res_df, config, op_idx1, op_idx2, tp_1_, tp_0_, tp_gap_, ohlc_list[1:3], open_side)
                    if expire:  # p1's expiry
                        if show_detail:
                            print("expiry_p1, touch_idx = {} : break".format(touch_idx))
                        i = touch_idx  # + 1  --> 이거 아닌것 같음 # op_idx1 과 op_idx2 사이의 op_idx1' 을 살리기 위함, 즉 바로 다음 op_idx1 로 회귀 (건너뛰지 않고)
                        open_i2 = prev_open_i2
                        break  # change op_idx1

            # ------ point validation ------ # p1_loop 로 return 되는 정확한 i 를 반환하기 위해서 expiry_p1 에 순서 양보  # Todo, 새로운 tp, ep, out 에 대한 처리 필요 (p1_hlm 사용시)
            if open_side == OrderSide.SELL:
                if not (tp_ < ep2_):  # tr_set validation & reject hl_out open_exec.
                    break  # change op_idx1
                elif not (ep2_ < out_ and close[op_idx2] < out_):
                    continue  # change op_idx2
            else:
                if not (tp_ > ep2_):
                    break
                elif not (ep2_ > out_ and close[op_idx2] > out_):
                    continue

            if check_hlm == 2:
                # ------ p2_box location ------ #
                if open_side == OrderSide.SELL:
                    if not ((tp_1_ + tp_gap_ * config.tr_set.p2_box_k1 <= out_1_) and (
                            out_0_ <= tp_0_ - tp_gap_ * config.tr_set.p2_box_k2)):  # tp1, tp0 에 닿으면 expiry
                        if show_detail:
                            print("p2_box rejection : continue")
                        continue
                    else:
                        # ------ p1p2_low ------ #
                        if not high[op_idx1:op_idx2 + 1].max() < tp_0_ - tp_gap_ * config.tr_set.p1p2_low:
                            if show_detail:
                                print("p1p2_low rejection : continue")
                            continue
                else:
                    if not ((tp_1_ - tp_gap_ * config.tr_set.p2_box_k1 >= out_1_) and (out_0_ >= tp_0_ + tp_gap_ * config.tr_set.p2_box_k2)):
                        if show_detail:
                            print("p2_box rejection : continue")
                        continue
                    else:
                        # ------ p1p2_low ------ #
                        if not low[op_idx1:op_idx2 + 1].min() > tp_0_ + tp_gap_ * config.tr_set.p1p2_low:
                            if show_detail:
                                print("p1p2_low rejection : continue")
                            continue

                # ------ check p2's expiry ------ #
                exec_j, ep_j, _, out_j, entry_done, en_p, fee = check_entry_v5(res_df, config, op_idx2, out_1_, out_gap_, len_df, open_side,
                                                                               [*ohlc_list, ep2_arr], expiry_p2)  # tp_j 는 op_idx1 사용
                i = exec_j  # = entry_loop 를 돌고 나온 e_j
                if not entry_done:  # p2's expiry
                    if show_detail:
                        print("expiry_p2, i = {} : continue".format(i))
                    continue  # change op_idx2

            # ------ tr_threshold ------ #   # en_p 에 대해 하는게 맞을 것으로봄
            if check_hlm:
                if open_side == OrderSide.SELL:
                    tr_ = abs((en_p / tp_ - config.trader_set.limit_fee - 1) / (en_p / out_ - config.trader_set.market_fee - 1))
                    if config.loc_set.point1.short_tr_thresh != "None":
                        if tr_ < config.loc_set.point1.short_tr_thresh:
                            continue
                else:
                    tr_ = abs((tp_ / en_p - config.trader_set.limit_fee - 1) / (out_ / en_p - config.trader_set.market_fee - 1))
                    if config.loc_set.point1.short_tr_thresh != "None":  # thresh 여부는 short 기준 공통
                        if tr_ < config.loc_set.point1.long_tr_thresh:
                            continue
            else:
                tr_ = tr_arr[op_idx1]

            # ------ leverage ------ #
            # out = out_arr[out_j]  # lvrg_set use out on out_j (out_j shoud be based on p2)
            leverage = lvrg_set(res_df, config, open_side, en_p, out_, fee)  # res_df 변수 사용됨 - 주석 처리 된 상태일뿐
            if leverage is None:
                if show_detail:
                    print("leverage is None : continue")
                if check_hlm:
                    continue  # change op_idx2
                else:
                    break  # change op_idx1

            exit_done, cross_on = 0, 0
            # ------ check tpout_onexec ------ #
            # if not config.ep_set.static_ep and config.ep_set.entry_type == "LIMIT" and config.ep_set.tpout_onexec:
            if config.ep_set.entry_type == "LIMIT":
                if config.tp_set.tp_onexec:  # dynamic 은 tp_onexec 사용하는 의미가 없음
                    tp_j = exec_j
                if config.out_set.out_onexec:  # dynamic 은 out_onexec 사용하는 의미가 없음
                    out_j = exec_j

            # ============ exit loop ============ # --> p1_hlm 의 경우, 1번만 실행
            if not allow_exit:
                continue

            while 1:
                if not config.tp_set.static_tp:  # 앞으로 왠만하면 static 만 사용할 예정
                    tp_j = i
                if not config.out_set.static_out:
                    out_j = i

                # ------------ out ------------ #  # out 우선 (보수적 검증)
                # ------ signal_out ------ #
                if not exit_done:
                    exit_done, cross_on, ex_p, fee = check_signal_out_v3(res_df, config, open_i2, i, len_df, fee, open_side, cross_on, exit_done,
                                                                         [*ohlc_list, np_timeidx])
                # ------ hl_out ------ #
                if config.out_set.hl_out != "None":
                    if not exit_done:  # and i != len_df - 1:
                        exit_done, ex_p, fee = check_hl_out_v2(config, i, out_j, len_df, fee, open_side, exit_done, [*ohlc_list, out_arr])

                # ------------ tp ------------ #
                if not config.tp_set.non_tp and i != exec_j:
                    if not exit_done:
                        exit_done, ex_p, fee = check_limit_tp_exec(res_df, config, open_i2, i, tp_j, len_df, fee, open_side, exit_done,
                                                                   [*ohlc_list, [tp_arr]])  # 여기서는 j -> i 로 변경해야함
                        # if config.tp_set.tp_type in ['LIMIT']:  # 'BOTH' -> 앞으로는, LIMIT 밖에 없을거라 주석처리함
                        # if not exit_done and config.tp_set.tp_type in ['MARKET', 'BOTH']:

                if exit_done:  # 이 phase 는 exit_phase 뒤에도 있어야할 것 - entry_done var. 사용은 안하겠지만
                    # ------ append dynamic vars. ------ #
                    p1_idx_list.append(op_idx1)  # side, zone, start_ver arr 모두 openi_list 로 접근하기 위해 open_i 를 담음
                    p2_idx_list.append(op_idx2)
                    pair_idx_list.append([exec_j, i])  # entry & exit (체결 기준임)
                    pair_price_list.append([en_p, ex_p])
                    lvrg_list.append(leverage)
                    fee_list.append(fee)
                    tpout_list.append([tp_arr[tp_j], out_arr[out_j]])  # for tpout_line plot_check
                    tr_list.append(tr_)  # Todo, tr vectorize 불가함, 직접 구해주어야할 건데.. (오래걸리지 않을까 --> tr_set 데이터만 모아서 vecto 계산이 나을 것)

                    # open_i += 1  # 다음 open_idx 조사 진행
                    break

                # 1. 아래있으면, 체결 기준부터 tp, out 허용 -> tp 가 entry_idx 에 체결되는게 다소 염려되기는 함, 일단 진행 (그런 case 가 많지 않았으므로)
                # 2. 위에있으면, entry 다음 tick 부터 exit 허용
                i += 1
                if i >= len_df:  # res_df 의 last_index 까지 돌아야함
                    break

            if i >= len_df:  # res_df 의 last_index 까지 돌아야함
                break

            if exit_done == 1:  # tp_done
                if show_detail:
                    print("exit_done = {}, i = {} : break".format(exit_done, i))
                break  # change op_idx1
            else:
                if check_hlm:
                    if check_hlm == 1:  # exit only once in p1_hlm mode
                        allow_exit = 0
                    if show_detail:
                        print("exit_done = {}, i = {} : continue".format(exit_done, i))
                    continue  # change op_idx2
                else:
                    if show_detail:
                        print("exit_done = {}, i = {} : break".format(exit_done, i))
                    break  # change op_idx1

        if i >= len_df:  # or open_i >= len_open_idx:  # res_df 의 last_index 까지 돌아야함
            break
        else:
            continue

    return np.array(net_p1_idx_list), np.array(p1_idx_list), np.array(p2_idx_list), np.array(pair_idx_list), np.array(pair_price_list), np.array(
        lvrg_list), np.array(
        fee_list), np.array(tpout_list), np.array(tr_list)

def en_ex_pairing_v8(res_df, open_idx_list, open_info_list, ohlc_list, config_list, np_timeidx, funcs, check_hlm=0,
                     show_detail=False):  # 이미 충분히 줄여놓은 idx 임

    open_info1, open_info2 = open_info_list
    side_arr1, _, _, id_idx_arr1 = open_info1
    side_arr2, _, _, _ = open_info2

    expiry_p1, expiry_p2, lvrg_set = funcs

    net_p1_idx_list, p1_idx_list, p2_idx_list, pair_idx_list, pair_price_list, lvrg_list, fee_list, tpout_list, tr_list = [[] for li in range(9)]
    len_df = len(res_df)

    close = ohlc_list[3]

    open_idx1, open_idx2 = open_idx_list
    len_open_idx1 = len(open_idx1)
    len_open_idx2 = len(open_idx2)
    i, open_i1, open_i2 = 0, -1, -1  # i for total_res_df indexing

    while 1:  # for p1's loop

        # Todo,
        #   1. (갱신) p1's open_i + 1 과 op_idx 를 꺼내오는 건, eik1 또는 tp 체결의 경우만 해당됨,
        #   2. out 의 경우 p2's op_idx 기준으로 retry 필요
        #     a. 또한, p2's op_idx > p1's op_idx

        # ============ get p1_info ============ #
        # if eik1 or tp_done or first loop:
        open_i1 += 1  # 확인 끝났으면 조기 이탈(+1), 다음 open_idx 조사 진행
        if open_i1 >= len_open_idx1:
            break

        op_idx1 = open_idx1[open_i1]  # open_i1 는 i 와 별개로 운영
        if op_idx1 < i:  # i = 이전 거래 끝난후의 res_df index - "거래 종료후 거래 시작", '<' : 거래 종료시점 진입 가능하다는 의미
            continue

        # ------ set loop index i ------ #
        i = op_idx1  # + 1 --> op_idx1 = op_idx2 가능함 # open_signal 이 close_bar.shift(1) 이라고 가정하고 다음 bar 부터 체결확인한다는 의미
        if i >= len_df:  # res_df 의 last_index 까지 돌아야함
            break

            # ------ dynamic arr info by ID ------ #
        #     1. 해당 id 로 config 재할당해야함
        #       a. use open_i1
        open_side = side_arr1[open_i1]
        id_idx = id_idx_arr1.astype(int)[open_i1]
        config = config_list[id_idx]
        selection_id = config.selection_id
        # check_net_hhm = 1 if (config.tr_set.wave_itv1 == config.tr_set.wave_itv2) and (config.tr_set.wave_period1 == config.tr_set.wave_period2) else 0

        side_pos = 'short' if open_side == OrderSide.SELL else 'long'
        if show_detail:
            print("============ op_idx1 : {} {} ======".format(op_idx1, open_side))

        # ------ load tr_data ------ #
        tp_arr = res_df['{}_tp_{}'.format(side_pos, selection_id)].to_numpy()
        ep1_arr = res_df['{}_ep1_{}'.format(side_pos, selection_id)].to_numpy()
        ep2_arr = res_df['{}_ep2_{}'.format(side_pos, selection_id)].to_numpy()
        out_arr = res_df['{}_out_{}'.format(side_pos, selection_id)].to_numpy()

        tp_1_ = res_df['{}_tp_1_{}'.format(side_pos, selection_id)].to_numpy()[op_idx1]  # for p2_box location & p1's exipiry
        tp_0_ = res_df['{}_tp_0_{}'.format(side_pos, selection_id)].to_numpy()[op_idx1]
        tp_gap_ = res_df['{}_tp_gap_{}'.format(side_pos, selection_id)].to_numpy()[op_idx1]

        # if not check_net_hhm:  # this phase exist for p1 entry (net hhm sync.) in p2_platform
        exec_j, ep_j, tp_j, out_j, entry_done, en_p, fee = check_entry_v5(res_df, config, op_idx1, tp_1_, tp_gap_, len_df, open_side,
                                                                          [*ohlc_list, ep1_arr], expiry_p2)
        i = exec_j  # = entry_loop 를 돌고 나온 e_j
        if not entry_done:
            if show_detail:
                print("p1's expiry on p1_loop continue")
            continue
            # else:
        #   tp_j = op_idx1

        prev_open_i2 = open_i2
        net_p1_idx_list.append(op_idx1)
        if check_hlm in [0, 1]:
            i = op_idx1  # allow op_idx2 = op_idx1
        allow_exit = 1
        # ============ entry loop ============ #
        while 1:  # for p2's loop (allow retry)

            # ============ get p2_info ============ #
            open_i2 += 1  # 확인 끝났으면 조기 이탈(+1), 다음 open_idx 조사 진행
            if open_i2 >= len_open_idx2:  # open_i2 소진
                break

            if show_detail:
                print("open_i2 :", open_i2, side_arr2[open_i2])

            # ------ check side sync. ------ #
            if open_side != side_arr2[open_i2]:
                continue

            op_idx2 = open_idx2[open_i2]  # open_i2 는 i 와 별개로 운영
            if op_idx2 < i:  # 이곳 i = op_idx1 + 1 or p2's exec_j or exit_loop's i + 1
                continue

            if show_detail:
                print("op_idx1, op_idx2 :", op_idx1, op_idx2, side_arr2[open_i2])

            i = op_idx2 + 1  # open_signal 이 close_bar.shift(1) 이라고 가정하고 다음 bar 부터 체결확인한다는 의미
            if i >= len_df:  # res_df 의 last_index 까지 돌아야함
                break

            tp_ = tp_arr[op_idx1]
            ep2_ = ep2_arr[op_idx2]
            out_ = out_arr[op_idx2]

            out_1_ = res_df['{}_out_1_{}'.format(side_pos, selection_id)].to_numpy()[op_idx2]
            out_0_ = res_df['{}_out_0_{}'.format(side_pos, selection_id)].to_numpy()[op_idx2]
            out_gap_ = res_df['{}_out_gap_{}'.format(side_pos, selection_id)].to_numpy()[op_idx2]

            # ------ check p1's expiry - Todo, priority ------ # - p2_box 생성 이전의 hl_survey
            # 1. op_idx1 ~ op_idx2 까지의 hl_check
            # if not check_net_hhm:
            if check_hlm:  # p1_hlm, p2_hlm
                if op_idx1 < op_idx2:
                    expire, touch_idx = expiry_p1(res_df, config, op_idx1, op_idx2, tp_1_, tp_0_, tp_gap_, ohlc_list[1:3], open_side)
                    if expire:  # p1's expiry
                        if show_detail:
                            print("expiry_p1 break {}".format(touch_idx))
                        i = touch_idx  # + 1  --> 이거 아닌것 같음 # op_idx1 과 op_idx2 사이의 op_idx1' 을 살리기 위함, 즉 바로 다음 op_idx1 로 회귀 (건너뛰지 않고)
                        open_i2 = prev_open_i2
                        break  # change op_idx1

            # ------ point validation ------ # p1_loop 로 return 되는 정확한 i 를 반환하기 위해서 expiry_p1 에 순서 양보  # Todo, 새로운 tp, ep, out 에 대한 처리 필요 (p1_hlm 사용시)
            if open_side == OrderSide.SELL:
                if not (tp_ < ep2_):  # tr_set validation & reject hl_out open_exec.
                    break  # change op_idx1
                elif not (ep2_ < out_ and close[op_idx2] < out_):
                    continue  # change op_idx2
            else:
                if not (tp_ > ep2_):
                    break
                elif not (ep2_ > out_ and close[op_idx2] > out_):
                    continue

            # ------ tr_threshold ------ #
            if open_side == OrderSide.SELL:
                tr_ = abs((ep2_ / tp_ - config.trader_set.limit_fee - 1) / (ep2_ / out_ - config.trader_set.market_fee - 1))
            else:
                tr_ = abs((tp_ / ep2_ - config.trader_set.limit_fee - 1) / (out_ / ep2_ - config.trader_set.market_fee - 1))

            if check_hlm == 2:
                # ------ p2_box location ------ #
                if open_side == OrderSide.SELL:
                    if not ((tp_1_ + tp_gap_ * config.tr_set.p2_box_k1 <= out_1_) and (
                            out_0_ <= tp_0_ - tp_gap_ * config.tr_set.p2_box_k2)):  # tp1, tp0 에 닿으면 expiry
                        if show_detail:
                            print("p2_box continue")
                        continue
                else:
                    if not ((tp_1_ - tp_gap_ * config.tr_set.p2_box_k1 >= out_1_) and (out_0_ >= tp_0_ + tp_gap_ * config.tr_set.p2_box_k2)):
                        if show_detail:
                            print("p2_box continue")
                        continue

                # ------ check p2's expiry ------ #
                exec_j, ep_j, _, out_j, entry_done, en_p, fee = check_entry_v5(res_df, config, op_idx2, out_1_, out_gap_, len_df, open_side,
                                                                               [*ohlc_list, ep2_arr], expiry_p2)  # tp_j 는 op_idx1 사용
                i = exec_j  # = entry_loop 를 돌고 나온 e_j
                if not entry_done:  # p2's expiry
                    if show_detail:
                        print("expiry_p2 continue {}".format(i))
                    continue  # change op_idx2

            # ------ leverage ------ #
            # out = out_arr[out_j]  # lvrg_set use out on out_j (out_j shoud be based on p2)
            leverage = lvrg_set(res_df, config, open_side, en_p, out_, fee)  # res_df 변수 사용됨 - 주석 처리 된 상태일뿐
            if leverage is None:
                if show_detail:
                    print("leverage is None continue")
                if check_hlm:
                    continue  # change op_idx2
                else:
                    break  # change op_idx1

            exit_done, cross_on = 0, 0
            # ------ check tpout_onexec ------ #
            # if not config.ep_set.static_ep and config.ep_set.entry_type == "LIMIT" and config.ep_set.tpout_onexec:
            if config.ep_set.entry_type == "LIMIT":
                if config.tp_set.tp_onexec:  # dynamic 은 tp_onexec 사용하는 의미가 없음
                    tp_j = exec_j
                if config.out_set.out_onexec:  # dynamic 은 out_onexec 사용하는 의미가 없음
                    out_j = exec_j

            # ============ exit loop ============ # --> p1_hlm 의 경우, 1번만 실행
            if not allow_exit:
                continue

            while 1:
                if not config.tp_set.static_tp:  # 앞으로 왠만하면 static 만 사용할 예정
                    tp_j = i
                if not config.out_set.static_out:
                    out_j = i

                # ------------ out ------------ #  # out 우선 (보수적 검증)
                # ------ signal_out ------ #
                if not exit_done:
                    exit_done, cross_on, ex_p, fee = check_signal_out_v3(res_df, config, open_i2, i, len_df, fee, open_side, cross_on, exit_done,
                                                                         [*ohlc_list, np_timeidx])
                # ------ hl_out ------ #
                if config.out_set.hl_out != "None":
                    if not exit_done:  # and i != len_df - 1:
                        exit_done, ex_p, fee = check_hl_out_v2(config, i, out_j, len_df, fee, open_side, exit_done, [*ohlc_list, out_arr])

                # ------------ tp ------------ #
                if not config.tp_set.non_tp and i != exec_j:
                    if not exit_done:
                        exit_done, ex_p, fee = check_limit_tp_exec(res_df, config, open_i2, i, tp_j, len_df, fee, open_side, exit_done,
                                                                   [*ohlc_list, [tp_arr]])  # 여기서는 j -> i 로 변경해야함
                        # if config.tp_set.tp_type in ['LIMIT']:  # 'BOTH' -> 앞으로는, LIMIT 밖에 없을거라 주석처리함
                        # if not exit_done and config.tp_set.tp_type in ['MARKET', 'BOTH']:

                if exit_done:  # 이 phase 는 exit_phase 뒤에도 있어야할 것 - entry_done var. 사용은 안하겠지만
                    # ------ append dynamic vars. ------ #
                    p1_idx_list.append(op_idx1)  # side, zone, start_ver arr 모두 openi_list 로 접근하기 위해 open_i 를 담음
                    p2_idx_list.append(op_idx2)
                    pair_idx_list.append([exec_j, i])  # entry & exit (체결 기준임)
                    pair_price_list.append([en_p, ex_p])
                    lvrg_list.append(leverage)
                    fee_list.append(fee)
                    tpout_list.append([tp_arr[tp_j], out_arr[out_j]])  # for tpout_line plot_check
                    tr_list.append(tr_)  # Todo, tr vectorize 불가함, 직접 구해주어야할 건데.. (오래걸리지 않을까 --> tr_set 데이터만 모아서 vecto 계산이 나을 것)

                    # open_i += 1  # 다음 open_idx 조사 진행
                    break

                # 1. 아래있으면, 체결 기준부터 tp, out 허용 -> tp 가 entry_idx 에 체결되는게 다소 염려되기는 함, 일단 진행 (그런 case 가 많지 않았으므로)
                # 2. 위에있으면, entry 다음 tick 부터 exit 허용
                i += 1
                if i >= len_df:  # res_df 의 last_index 까지 돌아야함
                    break

            if i >= len_df:  # res_df 의 last_index 까지 돌아야함
                break

            if exit_done == 1:  # tp_done
                if show_detail:
                    print("exit_done = {} break {}".format(exit_done, i))
                break  # change op_idx1
            else:
                if check_hlm:
                    if check_hlm == 1:  # exit only once in p1_hlm mode
                        allow_exit = 0
                    if show_detail:
                        print("exit_done = {} continue {}".format(exit_done, i))
                    continue  # change op_idx2
                else:
                    if show_detail:
                        print("exit_done = {} break {}".format(exit_done, i))
                    break  # change op_idx1

        if i >= len_df:  # or open_i >= len_open_idx:  # res_df 의 last_index 까지 돌아야함
            break
        else:
            continue

    return np.array(net_p1_idx_list), np.array(p1_idx_list), np.array(p2_idx_list), np.array(pair_idx_list), np.array(pair_price_list), np.array(
        lvrg_list), np.array(
        fee_list), np.array(tpout_list), np.array(tr_list)

def en_ex_pairing_v7(res_df, open_idx_list, open_info_list, ohlc_list, config_list, np_timeidx, funcs, check_net_hhm=False,
                     show_detail=False):  # 이미 충분히 줄여놓은 idx 임

    open_info1, open_info2 = open_info_list
    side_arr1, _, _, id_idx_arr1 = open_info1
    side_arr2, _, _, _ = open_info2

    expiry_p1, expiry_p2, lvrg_set = funcs

    net_p1_idx_list, p1_idx_list, p2_idx_list, pair_idx_list, pair_price_list, lvrg_list, fee_list, tpout_list, tr_list = [[] for li in range(9)]
    len_df = len(res_df)

    close = ohlc_list[3]

    open_idx1, open_idx2 = open_idx_list
    len_open_idx1 = len(open_idx1)
    len_open_idx2 = len(open_idx2)
    i, open_i1, open_i2 = 0, -1, -1  # i for total_res_df indexing

    while 1:  # for p1's loop

        # Todo,
        #   1. (갱신) p1's open_i + 1 과 op_idx 를 꺼내오는 건, eik1 또는 tp 체결의 경우만 해당됨,
        #   2. out 의 경우 p2's op_idx 기준으로 retry 필요
        #     a. 또한, p2's op_idx > p1's op_idx

        # ============ get p1_info ============ #
        # if eik1 or tp_done or first loop:
        open_i1 += 1  # 확인 끝났으면 조기 이탈(+1), 다음 open_idx 조사 진행
        if open_i1 >= len_open_idx1:
            break

        op_idx1 = open_idx1[open_i1]  # open_i1 는 i 와 별개로 운영
        if op_idx1 < i:  # i = 이전 거래 끝난후의 res_df index - "거래 종료후 거래 시작", '<' : 거래 종료시점 진입 가능하다는 의미
            continue

        # ------ set loop index i ------ #
        i = op_idx1  # + 1 --> op_idx1 = op_idx2 가능함 # open_signal 이 close_bar.shift(1) 이라고 가정하고 다음 bar 부터 체결확인한다는 의미
        if i >= len_df:  # res_df 의 last_index 까지 돌아야함
            break

            # ------ dynamic arr info by ID ------ #
        #     1. 해당 id 로 config 재할당해야함
        #       a. use open_i1
        open_side = side_arr1[open_i1]
        id_idx = id_idx_arr1.astype(int)[open_i1]
        config = config_list[id_idx]
        selection_id = config.selection_id
        # check_net_hhm = 1 if (config.tr_set.wave_itv1 == config.tr_set.wave_itv2) and (config.tr_set.wave_period1 == config.tr_set.wave_period2) else 0

        side_pos = 'short' if open_side == OrderSide.SELL else 'long'
        if show_detail:
            print("============ op_idx1 : {} {} ======".format(op_idx1, open_side))

        # ------ load tr_data ------ #
        tp_arr = res_df['{}_tp_{}'.format(side_pos, selection_id)].to_numpy()
        ep1_arr = res_df['{}_ep1_{}'.format(side_pos, selection_id)].to_numpy()
        ep2_arr = res_df['{}_ep2_{}'.format(side_pos, selection_id)].to_numpy()
        out_arr = res_df['{}_out_{}'.format(side_pos, selection_id)].to_numpy()

        tp_1_ = res_df['{}_tp_1_{}'.format(side_pos, selection_id)].to_numpy()[op_idx1]  # for p2_box location & p1's exipiry
        tp_0_ = res_df['{}_tp_0_{}'.format(side_pos, selection_id)].to_numpy()[op_idx1]
        tp_gap_ = res_df['{}_tp_gap_{}'.format(side_pos, selection_id)].to_numpy()[op_idx1]

        # if not check_net_hhm:  # this phase exist for p1 entry (net hhm sync.) in p2_platform
        exec_j, ep_j, tp_j, out_j, entry_done, en_p, fee = check_entry_v5(res_df, config, op_idx1, tp_1_, tp_gap_, len_df, open_side,
                                                                          [*ohlc_list, ep1_arr], expiry_p2)
        i = exec_j  # = entry_loop 를 돌고 나온 e_j
        if not entry_done:
            if show_detail:
                print("p1's expiry on p1_loop continue")
            continue
            # else:
        #   tp_j = op_idx1

        prev_open_i2 = open_i2
        net_p1_idx_list.append(op_idx1)
        if check_net_hhm:
            i = op_idx1  # allow op_idx2 = op_idx1
        # ============ entry loop ============ #
        while 1:  # for p2's loop (allow retry)

            # ============ get p2_info ============ #
            open_i2 += 1  # 확인 끝났으면 조기 이탈(+1), 다음 open_idx 조사 진행
            if open_i2 >= len_open_idx2:  # open_i2 소진
                break

            # ------ check side sync. ------ #
            if open_side != side_arr2[open_i2]:
                continue

            op_idx2 = open_idx2[open_i2]  # open_i2 는 i 와 별개로 운영
            if op_idx2 < i:  # 이곳 i = op_idx1 + 1 or p2's exec_j or exit_loop's i + 1
                continue
            # if not check_net_hhm:
            #   if op_idx2 < i:   # 이곳 i = op_idx1 + 1 or p2's exec_j or exit_loop's i + 1
            #     continue
            # else:
            #   if op_idx2 < op_idx1:
            #     continue

            i = op_idx2 + 1  # open_signal 이 close_bar.shift(1) 이라고 가정하고 다음 bar 부터 체결확인한다는 의미
            if i >= len_df:  # res_df 의 last_index 까지 돌아야함
                break

            tp_ = tp_arr[op_idx1]
            ep2_ = ep2_arr[op_idx2]
            out_ = out_arr[op_idx2]

            # ------ point validation ------ #
            if open_side == OrderSide.SELL:
                if not (tp_ < ep2_ < out_) or not (close[op_idx2] < out_):  # tr_set validation & reject hl_out open_exec.
                    break
            else:
                if not (tp_ > ep2_ > out_) or not (close[op_idx2] > out_):
                    break

            out_1_ = res_df['{}_out_1_{}'.format(side_pos, selection_id)].to_numpy()[op_idx2]
            out_0_ = res_df['{}_out_0_{}'.format(side_pos, selection_id)].to_numpy()[op_idx2]
            out_gap_ = res_df['{}_out_gap_{}'.format(side_pos, selection_id)].to_numpy()[op_idx2]

            # ------ check p1's expiry - Todo, priority ------ # - p2_box 생성 이전의 hl_survey
            # 1. op_idx1 ~ op_idx2 까지의 hl_check
            if not check_net_hhm:
                if op_idx1 < op_idx2:
                    if show_detail:
                        print("op_idx1, op_idx2 :", op_idx1, op_idx2, side_arr2[open_i2])
                    expire, touch_idx = expiry_p1(res_df, config, op_idx1, op_idx2, tp_1_, tp_0_, tp_gap_, ohlc_list[1:3], open_side)
                    if expire:  # p1's expiry
                        if show_detail:
                            print("expiry_p1 break {}".format(touch_idx))
                        i = touch_idx + 1  # op_idx1 과 op_idx2 사이의 op_idx1' 을 살리기 위함, 즉 바로 다음 op_idx1 로 회귀 (건너뛰지 않고)
                        open_i2 = prev_open_i2
                        break  # continue, to p1's loop

                # ------ p2_box location ------ #
                if open_side == OrderSide.SELL:
                    if not ((tp_1_ + tp_gap_ * config.tr_set.p2_box_k1 <= out_1_) and (
                            out_0_ <= tp_0_ - tp_gap_ * config.tr_set.p2_box_k2)):  # tp1, tp0 에 닿으면 expiry
                        if show_detail:
                            print("p2_box continue")
                        continue
                else:
                    if not ((tp_1_ - tp_gap_ * config.tr_set.p2_box_k1 >= out_1_) and (out_0_ >= tp_0_ + tp_gap_ * config.tr_set.p2_box_k2)):
                        if show_detail:
                            print("p2_box continue")
                        continue

            # ------ tr_threshold ------ #
            if open_side == OrderSide.SELL:
                tr_ = abs((ep2_ / tp_ - config.trader_set.limit_fee - 1) / (ep2_ / out_ - config.trader_set.market_fee - 1))
            else:
                tr_ = abs((tp_ / ep2_ - config.trader_set.limit_fee - 1) / (out_ / ep2_ - config.trader_set.market_fee - 1))

            # ------ check p2's expiry ------ #
            if not check_net_hhm:
                exec_j, ep_j, _, out_j, entry_done, en_p, fee = check_entry_v5(res_df, config, op_idx2, out_1_, out_gap_, len_df, open_side,
                                                                               [*ohlc_list, ep2_arr], expiry_p2)  # tp_j 는 op_idx1 사용
                i = exec_j  # = entry_loop 를 돌고 나온 e_j
                if not entry_done:  # p2's expiry
                    if show_detail:
                        print("expiry_p2 continue {}".format(i))
                    continue
                    # if not check_net_hhm:
                    #   continue
                    # else:
                    #   break   # use when check net hhm1

            # ------ leverage ------ #
            # out = out_arr[out_j]  # lvrg_set use out on out_j (out_j shoud be based on p2)
            leverage = lvrg_set(res_df, config, open_side, en_p, out_, fee)  # res_df 변수 사용됨 - 주석 처리 된 상태일뿐
            if leverage is None:
                if show_detail:
                    print("leverage is None continue")
                if not check_net_hhm:
                    continue
                else:
                    break  # use when check net hhm1

            exit_done, cross_on = 0, 0
            # ------ check tpout_onexec ------ #
            # if not config.ep_set.static_ep and config.ep_set.entry_type == "LIMIT" and config.ep_set.tpout_onexec:
            if config.ep_set.entry_type == "LIMIT":
                if config.tp_set.tp_onexec:  # dynamic 은 tp_onexec 사용하는 의미가 없음
                    tp_j = exec_j
                if config.out_set.out_onexec:  # dynamic 은 out_onexec 사용하는 의미가 없음
                    out_j = exec_j

            # ============ exit loop ============ #
            while 1:
                if not config.tp_set.static_tp:  # 앞으로 왠만하면 static 만 사용할 예정
                    tp_j = i
                if not config.out_set.static_out:
                    out_j = i

                # ------------ out ------------ #  # out 우선 (보수적 검증)
                # ------ signal_out ------ #
                if not exit_done:
                    exit_done, cross_on, ex_p, fee = check_signal_out_v3(res_df, config, open_i2, i, len_df, fee, open_side, cross_on, exit_done,
                                                                         [*ohlc_list, np_timeidx])
                # ------ hl_out ------ #
                if config.out_set.hl_out != "None":
                    if not exit_done:  # and i != len_df - 1:
                        exit_done, ex_p, fee = check_hl_out_v2(config, i, out_j, len_df, fee, open_side, exit_done, [*ohlc_list, out_arr])

                # ------------ tp ------------ #
                if not config.tp_set.non_tp and i != exec_j:
                    if not exit_done:
                        exit_done, ex_p, fee = check_limit_tp_exec(res_df, config, open_i2, i, tp_j, len_df, fee, open_side, exit_done,
                                                                   [*ohlc_list, [tp_arr]])  # 여기서는 j -> i 로 변경해야함
                        # if config.tp_set.tp_type in ['LIMIT']:  # 'BOTH' -> 앞으로는, LIMIT 밖에 없을거라 주석처리함
                        # if not exit_done and config.tp_set.tp_type in ['MARKET', 'BOTH']:

                if exit_done:  # 이 phase 는 exit_phase 뒤에도 있어야할 것 - entry_done var. 사용은 안하겠지만
                    # ------ append dynamic vars. ------ #
                    p1_idx_list.append(op_idx1)  # side, zone, start_ver arr 모두 openi_list 로 접근하기 위해 open_i 를 담음
                    p2_idx_list.append(op_idx2)
                    pair_idx_list.append([exec_j, i])  # entry & exit (체결 기준임)
                    pair_price_list.append([en_p, ex_p])
                    lvrg_list.append(leverage)
                    fee_list.append(fee)
                    tpout_list.append([tp_arr[tp_j], out_arr[out_j]])  # for tpout_line plot_check
                    tr_list.append(tr_)  # Todo, tr vectorize 불가함, 직접 구해주어야할 건데.. (오래걸리지 않을까 --> tr_set 데이터만 모아서 vecto 계산이 나을 것)

                    # open_i += 1  # 다음 open_idx 조사 진행
                    break

                # 1. 아래있으면, 체결 기준부터 tp, out 허용 -> tp 가 entry_idx 에 체결되는게 다소 염려되기는 함, 일단 진행 (그런 case 가 많지 않았으므로)
                # 2. 위에있으면, entry 다음 tick 부터 exit 허용
                i += 1
                if i >= len_df:  # res_df 의 last_index 까지 돌아야함
                    break

            if i >= len_df:  # res_df 의 last_index 까지 돌아야함
                break

            if exit_done == 1:  # tp_done
                if show_detail:
                    print("exit_done = {} break {}".format(exit_done, i))
                break
            else:
                if show_detail:
                    print("exit_done = {} continue {}".format(exit_done, i))
                if not check_net_hhm:
                    continue
                else:
                    break  # use when check net hhm1

        if i >= len_df:  # or open_i >= len_open_idx:  # res_df 의 last_index 까지 돌아야함
            break
        else:
            continue

    return np.array(net_p1_idx_list), np.array(p1_idx_list), np.array(p2_idx_list), np.array(pair_idx_list), np.array(pair_price_list), np.array(
        lvrg_list), np.array(
        fee_list), np.array(tpout_list), np.array(tr_list)

def en_ex_pairing_v6(res_df, open_idx_list, open_info_list, ohlc_list, config_list, np_timeidx, funcs):  # 이미 충분히 줄여놓은 idx 임

    open_info1, open_info2 = open_info_list
    side_arr1, _, _, id_idx_arr1 = open_info1
    side_arr2, _, _, _ = open_info2

    expiry_p1, expiry_p2, lvrg_set = funcs

    p1_openi_list, p2_idx_list, pair_idx_list, pair_price_list, lvrg_list, fee_list, tpout_list, tr_list = [[] for li in range(8)]
    len_df = len(res_df)

    open_idx1, open_idx2 = open_idx_list
    len_open_idx1 = len(open_idx1)
    len_open_idx2 = len(open_idx2)
    i, open_i1, open_i2 = 0, -1, -1  # i for total_res_df indexing

    while 1:  # for p1's loop

        # Todo,
        #   1. (갱신) p1's open_i + 1 과 op_idx 를 꺼내오는 건, eik1 또는 tp 체결의 경우만 해당됨,
        #   2. out 의 경우 p2's op_idx 기준으로 retry 필요
        #     a. 또한, p2's op_idx > p1's op_idx

        # ============ get p1_info ============ #
        # if eik1 or tp_done or first loop:
        open_i1 += 1  # 확인 끝났으면 조기 이탈(+1), 다음 open_idx 조사 진행
        if open_i1 >= len_open_idx1:
            break

        op_idx1 = open_idx1[open_i1]  # open_i1 는 i 와 별개로 운영
        if op_idx1 < i:  # i = 이전 거래 끝난후의 res_df index - "거래 종료후 거래 시작", '<' : 거래 종료시점 진입 가능하다는 의미
            continue

        # ------ set loop index i ------ #
        i = op_idx1  # + 1 --> op_idx1 = op_idx2 가능함 # open_signal 이 close_bar.shift(1) 이라고 가정하고 다음 bar 부터 체결확인한다는 의미
        if i >= len_df:  # res_df 의 last_index 까지 돌아야함
            break

        # ------ dynamic arr info by ID ------ #
        #     1. 해당 id 로 config 재할당해야함
        #       a. use open_i1
        open_side = side_arr1[open_i1]
        id_idx = id_idx_arr1.astype(int)[open_i1]
        config = config_list[id_idx]
        selection_id = config.selection_id

        side_pos = 'short' if open_side == OrderSide.SELL else 'long'

        # tr_arr = res_df['{}_tr_{}'.format(side_pos, selection_id)].to_numpy()

        # ------ point1 & 2's tp_j ------ #
        # point_idxgap = point_idxgap_arr[op_idx]
        # if np.isnan(point_idxgap):
        #     continue
        # else:
        #     # ------ allow point2 only next to point1 ------ #
        #     open_arr = res_df['{}_open_{}'.format(side_pos, selection_id)].to_numpy()
        #     tp_j = int(op_idx - point_idxgap)
        #     if np.sum(open_arr[tp_j:op_idx]) != 0:
        #         continue

        tp_j = op_idx1
        print("============ op_idx1 : {} {} ======".format(op_idx1, open_side))

        # ============ entry loop ============ #
        while 1:  # for p2's loop (allow retry)

            # ============ get p2_info ============ #
            open_i2 += 1  # 확인 끝났으면 조기 이탈(+1), 다음 open_idx 조사 진행
            if open_i2 >= len_open_idx2:  # open_i2 소진
                break

            # ------ check side sync. ------ #
            if open_side != side_arr2[open_i2]:
                continue

            op_idx2 = open_idx2[open_i2]  # open_i2 는 i 와 별개로 운영
            if op_idx2 < i:  # 이곳 i = op_idx1 + 1 or p2's exec_j or exit_loop's i + 1
                continue

            prev_i = i
            i = op_idx2 + 1  # open_signal 이 close_bar.shift(1) 이라고 가정하고 다음 bar 부터 체결확인한다는 의미
            if i >= len_df:  # res_df 의 last_index 까지 돌아야함
                break

            # ------ load tr_data ------ #
            tp_arr = res_df['{}_tp_{}'.format(side_pos, selection_id)].to_numpy()
            ep_arr = res_df['{}_ep_{}'.format(side_pos, selection_id)].to_numpy()
            out_arr = res_df['{}_out_{}'.format(side_pos, selection_id)].to_numpy()

            tp_ = tp_arr[op_idx1]
            ep_ = ep_arr[op_idx2]
            out_ = out_arr[op_idx2]

            tp_1_ = res_df['{}_tp_1_{}'.format(side_pos, selection_id)].to_numpy()[op_idx1]  # for p2_box location & p1's exipiry
            tp_0_ = res_df['{}_tp_0_{}'.format(side_pos, selection_id)].to_numpy()[op_idx1]
            tp_gap_ = res_df['{}_tp_gap_{}'.format(side_pos, selection_id)].to_numpy()[op_idx1]

            out_1_ = res_df['{}_out_1_{}'.format(side_pos, selection_id)].to_numpy()[op_idx2]
            out_0_ = res_df['{}_out_0_{}'.format(side_pos, selection_id)].to_numpy()[op_idx2]
            out_gap_ = res_df['{}_out_gap_{}'.format(side_pos, selection_id)].to_numpy()[op_idx2]

            # ------ check p1's expiry - Todo, priority ------ # - p2_box 생성 이전의 hl_survey
            # 1. i ~ op_idx2 까지의 hl_check
            if op_idx1 < op_idx2:
                print("op_idx1, op_idx2 :", op_idx1, op_idx2, side_arr2[open_i2])
                if expiry_p1(res_df, config, op_idx1, op_idx2, tp_1_, tp_0_, tp_gap_, ohlc_list[1:3], open_side):  # p1's expiry
                    # if prev_i < op_idx2:
                    # if expiry_p1(res_df, config, op_idx1, prev_i, op_idx2, ohlc_list[1:3], open_side):   # p1's expiry
                    print("expiry_p1 break")
                    break  # continue, to p1's loop

            # ------ p2_box location & tr_threshold ------ #
            if open_side == OrderSide.SELL:
                if not ((tp_1_ + tp_gap_ * config.tr_set.p2_box_k1 < out_1_) and (out_0_ < tp_0_ - tp_gap_ * config.tr_set.p2_box_k2)):
                    print("p2_box continue")
                    continue
                tr_ = abs((ep_ / tp_ - config.trader_set.limit_fee - 1) / (ep_ / out_ - config.trader_set.market_fee - 1))
            else:
                if not ((tp_1_ - tp_gap_ * config.tr_set.p2_box_k1 > out_1_) and (out_0_ > tp_0_ + tp_gap_ * config.tr_set.p2_box_k2)):
                    print("p2_box continue")
                    continue
                tr_ = abs((tp_ / ep_ - config.trader_set.limit_fee - 1) / (out_ / ep_ - config.trader_set.market_fee - 1))

            # ------ check p2's expiry ------ #
            exec_j, ep_j, tp_j, out_j, entry_done, en_p, fee = check_entry_v4(res_df, config, op_idx2, tp_j, out_1_, out_gap_, len_df, open_side,
                                                                              [*ohlc_list, ep_arr], expiry_p2)
            i = exec_j  # = entry_loop 를 돌고 나온 e_j
            if not entry_done:  # p2's expiry
                print("expiry_p2 conintue")
                continue

            # ------ leverage ------ #
            out = out_arr[out_j]  # lvrg_set use out on out_j (out_j shoud be based on p2)
            leverage = lvrg_set(res_df, config, open_side, en_p, out, fee)  # res_df 변수 사용됨 - 주석 처리 된 상태일뿐
            if leverage is None:
                print("leverage is None conintue")
                continue  # to p2's loop

            exit_done, cross_on = 0, 0
            # ------ check tpout_onexec ------ #
            # if not config.ep_set.static_ep and config.ep_set.entry_type == "LIMIT" and config.ep_set.tpout_onexec:
            if config.ep_set.entry_type == "LIMIT":
                if config.tp_set.tp_onexec:  # dynamic 은 tp_onexec 사용하는 의미가 없음
                    tp_j = exec_j
                if config.out_set.out_onexec:  # dynamic 은 out_onexec 사용하는 의미가 없음
                    out_j = exec_j

            # ============ exit loop ============ #
            while 1:
                if not config.tp_set.static_tp:  # 앞으로 왠만하면 static 만 사용할 예정
                    tp_j = i
                if not config.out_set.static_out:
                    out_j = i

                # ------------ out ------------ #  # out 우선 (보수적 검증)
                # ------ signal_out ------ #
                if not exit_done:
                    exit_done, cross_on, ex_p, fee = check_signal_out_v3(res_df, config, open_i2, i, len_df, fee, open_side, cross_on, exit_done,
                                                                         [*ohlc_list, np_timeidx])
                # ------ hl_out ------ #
                if config.out_set.hl_out != "None":
                    if not exit_done:  # and i != len_df - 1:
                        exit_done, ex_p, fee = check_hl_out_v2(config, i, out_j, len_df, fee, open_side, exit_done, [*ohlc_list, out_arr])

                # ------------ tp ------------ #
                if not config.tp_set.non_tp and i != exec_j:
                    if not exit_done:
                        exit_done, ex_p, fee = check_limit_tp_exec(res_df, config, open_i2, i, tp_j, len_df, fee, open_side, exit_done,
                                                                   [*ohlc_list, [tp_arr]])  # 여기서는 j -> i 로 변경해야함
                        # if config.tp_set.tp_type in ['LIMIT']:  # 'BOTH' -> 앞으로는, LIMIT 밖에 없을거라 주석처리함
                        # if not exit_done and config.tp_set.tp_type in ['MARKET', 'BOTH']:

                if exit_done:  # 이 phase 는 exit_phase 뒤에도 있어야할 것 - entry_done var. 사용은 안하겠지만
                    # ------ append dynamic vars. ------ #
                    p1_openi_list.append(open_i1)  # side, zone, start_ver arr 모두 openi_list 로 접근하기 위해 open_i 를 담음
                    p2_idx_list.append(op_idx2)
                    pair_idx_list.append([exec_j, i])  # entry & exit (체결 기준임)
                    pair_price_list.append([en_p, ex_p])
                    lvrg_list.append(leverage)
                    fee_list.append(fee)
                    tpout_list.append([tp_arr[tp_j], out_arr[out_j]])  # for tpout_line plot_check
                    tr_list.append(tr_)  # Todo, tr vectorize 불가함, 직접 구해주어야할 건데.. (오래걸리지 않을까 --> tr_set 데이터만 모아서 vecto 계산이 나을 것)

                    # open_i += 1  # 다음 open_idx 조사 진행
                    break

                # 1. 아래있으면, 체결 기준부터 tp, out 허용 -> tp 가 entry_idx 에 체결되는게 다소 염려되기는 함, 일단 진행 (그런 case 가 많지 않았으므로)
                # 2. 위에있으면, entry 다음 tick 부터 exit 허용
                i += 1
                if i >= len_df:  # res_df 의 last_index 까지 돌아야함
                    break

            if exit_done == 1:  # tp_done
                print("exit_done = {} break".format(exit_done))
                break
            # else:
            #   continue  # to p2's loop
            print("exit_done = {} continue".format(exit_done))

            if i >= len_df:  # res_df 의 last_index 까지 돌아야함
                break

        # if not entry_done:   # expiry
        #     continue # to p1's loop

        if i >= len_df:  # or open_i >= len_open_idx:  # res_df 의 last_index 까지 돌아야함
            break
        else:
            continue

    return np.array(p1_openi_list), np.array(p2_idx_list), np.array(pair_idx_list), np.array(pair_price_list), np.array(lvrg_list), np.array(
        fee_list), np.array(tpout_list), np.array(tr_list)

def en_ex_pairing_v5(res_df, open_idx, open_info_list, ohlc_list, config_list, np_timeidx, funcs):  # 이미 충분히 줄여놓은 idx 임
    side_arr, zone_arr, id_arr, id_idx_arr = open_info_list
    id_idx_arr = id_idx_arr.astype(int)

    ep_out, ep_loc_point2, lvrg_set = funcs

    point1_list, valid_openi_list, pair_idx_list, pair_price_list, lvrg_list, fee_list, tpout_list, tr_list = [[] for li in range(8)]
    len_df = len(res_df)
    len_open_idx = len(open_idx)
    i, open_i = 0, -1  # i for total_res_df indexing

    while 1:
        # ------------ entry phase ------------ #
        open_i += 1  # 확인 끝났으면 조기 이탈(+1), 다음 open_idx 조사 진행
        if open_i >= len_open_idx:
            break

        # ------ ep_loc ------ #
        op_idx = open_idx[open_i]  # open_i 는 i 와 별개로 운영
        if op_idx < i:  # i = 이전 거래 끝난후의 res_df index - "거래 종료후 거래 시작", '<' : 거래 종료시점 진입 가능하다는 의미
            continue

        # ------ dynamic data by ID ------ #
        #     1. 해당 id 로 config 재할당해야함
        id_idx = id_idx_arr[open_i]
        config = config_list[id_idx]
        selection_id = config.selection_id
        open_side = side_arr[open_i]

        side_pos = 'short' if open_side == OrderSide.SELL else 'long'
        tp_arr = res_df['{}_tp_{}'.format(side_pos, selection_id)].to_numpy()  # => eptpout arr_list 만들어서 꺼내 사용하면 될 것
        # point_idxgap_arr = res_df['{}_point_idxgap_{}'.format(side_pos, selection_id)].to_numpy()

        ep_arr = res_df['{}_ep_{}'.format(side_pos, selection_id)].to_numpy()  # Todo - while loop 내에서 to_numpy() 반복하느니, pd_indexing 이 낫지 않을까
        out_arr = res_df['{}_out_{}'.format(side_pos, selection_id)].to_numpy()
        # bias_info_arr = res_df['{}_bias_info_{}'.format(side_pos, selection_id)].to_numpy()  # ex. rolling(entry ~ end)'s high
        # bias_thresh_arr = res_df['{}_bias_thresh_{}'.format(side_pos, selection_id)].to_numpy()  # ex. close + dc_T20 * 0.5
        tr_arr = res_df['{}_tr_{}'.format(side_pos, selection_id)].to_numpy()

        # ------ ei_k & point2 ------ #
        i = op_idx + 1  # open_signal 이 close_bar.shift(1) 이라고 가정하고 다음 bar 부터 체결확인한다는 의미
        if i >= len_df:  # res_df 의 last_index 까지 돌아야함
            break

        # ------ point1 & 2's tp_j ------ #
        # point_idxgap = point_idxgap_arr[op_idx]
        # if np.isnan(point_idxgap):
        #     continue
        # else:
        #     # ------ allow point2 only next to point1 ------ #
        #     open_arr = res_df['{}_open_{}'.format(side_pos, selection_id)].to_numpy()
        #     tp_j = int(op_idx - point_idxgap)
        #     if np.sum(open_arr[tp_j:op_idx]) != 0:
        #         continue
        tp_j = op_idx

        exec_j, ep_j, tp_j, out_j, entry_done, en_p, fee = check_eik_point2_exec_v3(res_df, config, op_idx, tp_j, len_df, open_side,
                                                                                 [*ohlc_list, ep_arr], ep_out, ep_loc_point2)
        i = exec_j

        if not entry_done:
            continue

        # ------ leverage ------ #
        out = out_arr[out_j]  # lvrg_set use out on out_j
        leverage = lvrg_set(res_df, config, open_side, en_p, out, fee)  # res_df 변수 사용됨 - 주석 처리 된 상태일뿐
        if leverage is None:
            continue

        exit_done, cross_on = 0, 0
        # ------ check tpout_onexec ------ #
        # if not config.ep_set.static_ep and config.ep_set.entry_type == "LIMIT" and config.ep_set.tpout_onexec:
        if config.ep_set.entry_type == "LIMIT":
            if config.tp_set.tp_onexec:  # dynamic 은 tp_onexec 사용하는 의미가 없음
                tp_j = exec_j
            if config.out_set.out_onexec:  # dynamic 은 out_onexec 사용하는 의미가 없음
                out_j = exec_j

        while 1:
            # ------------ exit phase ------------ #
            if not config.tp_set.static_tp:  # 앞으로 왠만하면 static 만 사용할 예정
                tp_j = i
            if not config.out_set.static_out:
                out_j = i

            # ------------ tp ------------ #
            if not config.tp_set.non_tp and i != exec_j:
                exit_done, ex_p, fee = check_limit_tp_exec(res_df, config, open_i, i, tp_j, len_df, fee, open_side, exit_done,
                                                           [*ohlc_list, [tp_arr]])  # 여기서는 j -> i 로 변경해야함
                # if config.tp_set.tp_type in ['LIMIT']:  # 'BOTH' -> 앞으로는, LIMIT 밖에 없을거라 주석처리함
                # if not exit_done and config.tp_set.tp_type in ['MARKET', 'BOTH']:

            # ------------ out ------------ #
            # ------ signal_out ------ #
            if not exit_done:
                exit_done, cross_on, ex_p, fee = check_market_out_exec_v2(config, res_df, np_timeidx, open_i, i, len_df, fee, open_side, cross_on, exit_done)
            # ------ hl_out ------ #
            if config.out_set.hl_out != "None":
                if not exit_done:  # and i != len_df - 1:
                    exit_done, ex_p, fee = check_out(config, open_i, i, out_j, len_df, fee, open_side, exit_done, [*ohlc_list, out_arr])

            if exit_done:  # 이 phase 는 exit_phase 뒤에도 있어야할 것 - entry_done var. 사용은 안하겠지만
                # ------ append dynamic vars. ------ #
                point1_list.append(tp_j)
                valid_openi_list.append(open_i)  # side, zone, start_ver arr 모두 openi_list 로 접근하기 위해 open_i 를 담음
                pair_idx_list.append([exec_j, i])  # entry & exit (체결 기준임)
                pair_price_list.append([en_p, ex_p])
                lvrg_list.append(leverage)
                fee_list.append(fee)
                tpout_list.append([tp_arr[tp_j], out_arr[out_j]])  # for tpout_line plot_check
                # bias_list.append([bias_info_arr[exec_j], bias_thresh_arr[exec_j]])  # backtest 에서만 가능한 future_data 사용
                # bias_list.append([bias_info_arr[exec_j], tp_arr[tp_j]])  # bias_info 는 entry_idx 부터 & tp = bias_thresh
                tr_list.append(tr_arr[op_idx])

                # open_i += 1  # 다음 open_idx 조사 진행
                break

            # 1. 아래있으면, 체결 기준부터 tp, out 허용 -> tp 가 entry_idx 에 체결되는게 다소 염려되기는 함, 일단 진행 (그런 case 가 많지 않았으므로)
            # 2. 위에있으면, entry 다음 tick 부터 exit 허용
            i += 1
            if i >= len_df:  # res_df 의 last_index 까지 돌아야함
                break

        if i >= len_df:  # or open_i >= len_open_idx:  # res_df 의 last_index 까지 돌아야함
            break
        else:
            continue

    return np.array(point1_list), np.array(valid_openi_list), np.array(pair_idx_list), np.array(pair_price_list), np.array(lvrg_list), np.array(
        fee_list), np.array(tpout_list), np.array(tr_list)

def check_entry_v6(res_df, config, entry_type, op_idx, wave1, wave_gap, len_df, open_side, np_datas, expiry):
    open, high, low, close, ep_arr = np_datas
    ep_j = op_idx
    tp_j = op_idx
    out_j = op_idx

    # print("ep_arr[op_idx] :", ep_arr[op_idx])

    selection_id = config.selection_id
    # allow_ep_in = 0 if config.ep_set.point2.use_point2 else 1
    entry_done = 0
    ep = None

    if entry_type == "LIMIT":
        fee = config.trader_set.limit_fee

        for e_j in range(op_idx + 1, len_df):
            # ------ index setting for dynamic options ------ #
            if not config.ep_set.static_ep:
                ep_j = e_j  # dynamic_ep 를 위한 ep_index var.
                out_j = e_j  # dynamic_out 를 위한 out_index var. - 조건식이 static_ep 와 같이 있는 이유 모름 => dynamic_lvrg 로 사료됨

            if not config.tp_set.static_tp:
                tp_j = e_j

            # ------ expire_k & expire_tick ------ # - limit 사용하면 default 로 expire_k 가 존재해야함
            if expiry(res_df, config, op_idx, e_j, wave1, wave_gap, [high, low], open_side):  # tp_j,
              break

            # ------ point2 ------ #
            # if not allow_ep_in:
            #     allow_ep_in, out_j = ep_loc_point2(res_df, config, e_j, out_j, side=OrderSide.SELL)
            #     if allow_ep_in:
            #       if config.ep_set.point2.entry_type == "LIMIT":
            #         ep_j = e_j
            #         # print("e_j in point2 :", e_j)
            #         continue

            # ------ check ep_exec ------ #
            # if allow_ep_in:
              # if config.ep_set.point2.use_point2 and config.ep_set.point2.entry_type == 'MARKET':
              #   entry_done = 1
              #   ep = c[e_j]
              #   break
              # else:

            if open_side == OrderSide.SELL:
              if high[e_j] >= ep_arr[ep_j]:
                  entry_done = 1
                  ep = ep_arr[ep_j]
                  if open[e_j] >= ep_arr[ep_j]:  # open comp 는 결국, 수익률에 얹어주는 logic (반보수) -> 사용 보류
                      ep = open[e_j]
                  break
            else:
              if low[e_j] <= ep_arr[ep_j]:
                  entry_done = 1
                  ep = ep_arr[ep_j]
                  if open[e_j] <= ep_arr[ep_j]:
                      ep = open[e_j]
                  break

    else:  # market entry
        e_j = op_idx + 1
        entry_done = 1
        ep = close[op_idx]
        fee = config.trader_set.market_fee

    return e_j, ep_j, tp_j, out_j, entry_done, ep, fee  # 다음 start_i <-- e_j 로 변경
    #   e_j => 다음 phase 의 시작 index <-> ep_j : ep 의 기준 index
    #   ep_j, tp_j, out_j 가 return 되어야함 - exit phase 에서 이어가기 위함

def check_entry_v5(res_df, config, op_idx, wave1, wave_gap, len_df, open_side, np_datas, expiry):
    open, high, low, close, ep_arr = np_datas
    ep_j = op_idx
    tp_j = op_idx
    out_j = op_idx

    # print("ep_arr[op_idx] :", ep_arr[op_idx])

    selection_id = config.selection_id
    # allow_ep_in = 0 if config.ep_set.point2.use_point2 else 1
    entry_done = 0
    ep = None

    if config.ep_set.entry_type == "LIMIT":
        fee = config.trader_set.limit_fee

        for e_j in range(op_idx + 1, len_df):
            # ------ index setting for dynamic options ------ #
            if not config.ep_set.static_ep:
                ep_j = e_j  # dynamic_ep 를 위한 ep_index var.
                out_j = e_j  # dynamic_out 를 위한 out_index var. - 조건식이 static_ep 와 같이 있는 이유 모름 => dynamic_lvrg 로 사료됨

            if not config.tp_set.static_tp:
                tp_j = e_j

            # ------ expire_k & expire_tick ------ # - limit 사용하면 default 로 expire_k 가 존재해야함
            if expiry(res_df, config, op_idx, e_j, wave1, wave_gap, [high, low], open_side):  # tp_j,
              break

            # ------ point2 ------ #
            # if not allow_ep_in:
            #     allow_ep_in, out_j = ep_loc_point2(res_df, config, e_j, out_j, side=OrderSide.SELL)
            #     if allow_ep_in:
            #       if config.ep_set.point2.entry_type == "LIMIT":
            #         ep_j = e_j
            #         # print("e_j in point2 :", e_j)
            #         continue

            # ------ check ep_exec ------ #
            # if allow_ep_in:
              # if config.ep_set.point2.use_point2 and config.ep_set.point2.entry_type == 'MARKET':
              #   entry_done = 1
              #   ep = c[e_j]
              #   break
              # else:

            if open_side == OrderSide.SELL:
              if high[e_j] >= ep_arr[ep_j]:
                  entry_done = 1
                  ep = ep_arr[ep_j]
                  if open[e_j] >= ep_arr[ep_j]:  # open comp 는 결국, 수익률에 얹어주는 logic (반보수) -> 사용 보류
                      ep = open[e_j]
                  break
            else:
              if low[e_j] <= ep_arr[ep_j]:
                  entry_done = 1
                  ep = ep_arr[ep_j]
                  if open[e_j] <= ep_arr[ep_j]:
                      ep = open[e_j]
                  break

    else:  # market entry
        e_j = op_idx + 1
        entry_done = 1
        ep = close[op_idx]
        fee = config.trader_set.market_fee

    return e_j, ep_j, tp_j, out_j, entry_done, ep, fee  # 다음 start_i <-- e_j 로 변경
    #   e_j => 다음 phase 의 시작 index <-> ep_j : ep 의 기준 index
    #   ep_j, tp_j, out_j 가 return 되어야함 - exit phase 에서 이어가기 위함

def check_entry_v4(res_df, config, op_idx, tp_j, len_df, open_side, np_datas, expiry):
    open, high, low, close, ep_arr = np_datas
    ep_j = op_idx
    # tp_j = op_idx
    out_j = op_idx

    selection_id = config.selection_id
    # allow_ep_in = 0 if config.ep_set.point2.use_point2 else 1
    entry_done = 0
    ep = None

    if config.ep_set.entry_type == "LIMIT":
        fee = config.trader_set.limit_fee

        for e_j in range(op_idx + 1, len_df):
            # ------ index setting for dynamic options ------ #
            if not config.ep_set.static_ep:
                ep_j = e_j  # dynamic_ep 를 위한 ep_index var.
                out_j = e_j  # dynamic_out 를 위한 out_index var. - 조건식이 static_ep 와 같이 있는 이유 모름 => dynamic_lvrg 로 사료됨

            if not config.tp_set.static_tp:
                tp_j = e_j

            # ------ expire_k & expire_tick ------ # - limit 사용하면 default 로 expire_k 가 존재해야함
            if expiry(res_df, config, op_idx, e_j, [high, low], open_side):  # tp_j,
              break

            # ------ point2 ------ #
            # if not allow_ep_in:
            #     allow_ep_in, out_j = ep_loc_point2(res_df, config, e_j, out_j, side=OrderSide.SELL)
            #     if allow_ep_in:
            #       if config.ep_set.point2.entry_type == "LIMIT":
            #         ep_j = e_j
            #         # print("e_j in point2 :", e_j)
            #         continue

            # ------ check ep_exec ------ #
            # if allow_ep_in:
              # if config.ep_set.point2.use_point2 and config.ep_set.point2.entry_type == 'MARKET':
              #   entry_done = 1
              #   ep = c[e_j]
              #   break
              # else:

            if open_side == OrderSide.SELL:
              if high[e_j] >= ep_arr[ep_j]:
                  entry_done = 1
                  ep = ep_arr[ep_j]
                  if open[e_j] >= ep_arr[ep_j]:  # open comp 는 결국, 수익률에 얹어주는 logic (반보수) -> 사용 보류
                      ep = open[e_j]
                  break
            else:
              if low[e_j] <= ep_arr[ep_j]:
                  entry_done = 1
                  ep = ep_arr[ep_j]
                  if open[e_j] <= ep_arr[ep_j]:
                      ep = open[e_j]
                  break

    else:  # market entry
        e_j = op_idx + 1
        entry_done = 1
        ep = close[op_idx]
        fee = config.trader_set.market_fee

    return e_j, ep_j, tp_j, out_j, entry_done, ep, fee  # 다음 start_i <-- e_j 로 변경
    #   e_j => 다음 phase 의 시작 index <-> ep_j : ep 의 기준 index
    #   ep_j, tp_j, out_j 가 return 되어야함 - exit phase 에서 이어가기 위함

def check_eik_point2_exec_v3(res_df, config, op_idx, tp_j, len_df, open_side, np_datas, ep_out, ep_loc_point2):
    o, h, l, c, ep_arr = np_datas
    ep_j = op_idx
    # tp_j = op_idx
    out_j = op_idx

    selection_id = config.selection_id
    allow_ep_in = 0 if config.ep_set.point2.use_point2 else 1
    entry_done = 0
    ep = None

    if config.ep_set.entry_type == "LIMIT":
        fee = config.trader_set.limit_fee

        for e_j in range(op_idx + 1, len_df):
            # ------ index setting for dynamic options ------ #
            if not config.ep_set.static_ep:
                ep_j = e_j  # dynamic_ep 를 위한 ep_index var.
                out_j = e_j  # dynamic_out 를 위한 out_index var. - 조건식이 static_ep 와 같이 있는 이유 모름 => dynamic_lvrg 로 사료됨

            if not config.tp_set.static_tp:
                tp_j = e_j

            # ------ ei_k ------ # - limit 사용하면 default 로 ei_k 가 존재해야함
            if ep_out(res_df, config, op_idx, e_j, tp_j, [h, l], open_side):
              break

            # ------ point2 ------ #
            if not allow_ep_in:
                allow_ep_in, out_j = ep_loc_point2(res_df, config, e_j, out_j, side=OrderSide.SELL)
                if allow_ep_in:
                  if config.ep_set.point2.entry_type == "LIMIT":
                    ep_j = e_j
                    # print("e_j in point2 :", e_j)
                    continue

            # ------ check ep_exec ------ #
            if allow_ep_in:
              if config.ep_set.point2.use_point2 and config.ep_set.point2.entry_type == 'MARKET':
                entry_done = 1
                ep = c[e_j]
                break
              else:
                if open_side == OrderSide.SELL:
                    if h[e_j] >= ep_arr[ep_j]:
                        entry_done = 1
                        ep = ep_arr[ep_j]
                        if o[e_j] >= ep_arr[ep_j]:  # open comp 는 결국, 수익률에 얹어주는 logic (반보수) -> 사용 보류
                            ep = o[e_j]
                        break
                else:
                    if l[e_j] <= ep_arr[ep_j]:
                        entry_done = 1
                        ep = ep_arr[ep_j]
                        if o[e_j] <= ep_arr[ep_j]:
                            ep = o[e_j]
                        break

    else:  # market entry
        e_j = op_idx + 1
        entry_done = 1
        ep = c[op_idx]
        fee = config.trader_set.market_fee

    return e_j, ep_j, tp_j, out_j, entry_done, ep, fee  # 다음 start_i <-- e_j 로 변경
    #   e_j => 다음 phase 의 시작 index <-> ep_j : ep 의 기준 index
    #   ep_j, tp_j, out_j 가 return 되어야함 - exit phase 에서 이어가기 위함


def check_limit_tp_exec(res_df, config, open_i, i, tp_j, len_df, fee, open_side, exit_done, np_datas):

    open, high, low, close, tps = np_datas
    tp = None
    selection_id = config.selection_id
    len_tps = len(tps)

    for tp_i, tp_arr in enumerate(tps):

        #     decay adjustment    #
        #     tp_j includes dynamic_j - functionalize  #
        # try:
        #     if config.tr_set.decay_gap != "None":
        #         decay_share = (j - open_i) // config.tp_set.decay_term
        #         decay_remain = (j - open_i) % config.tp_set.decay_term
        #         if j != open_i and decay_remain == 0:
        #             if open_side == OrderSide.SELL:
        #                 tp_arr[tp_j] += res_df['short_tp_gap_{}'.format(selection_id)].iloc[open_i] * config.tr_set.decay_gap * decay_share
        #             else:
        #                 tp_arr[tp_j] -= res_df['long_tp_gap_{}'.format(selection_id)].iloc[open_i] * config.tr_set.decay_gap * decay_share
        # except:
        #     pass

        if open_side == OrderSide.SELL:
            if low[i] <= tp_arr[tp_j]:  # and partial_tp_cnt == tp_i:  # we use static tp now
                # if low[i] <= tp_arr[i] <= h[i]: --> 이건 잘못되었음
                # partial_tp_cnt += 1 --> partial_tp 보류

                # ------ dynamic tp ------ #
                if tp_arr[i] != tp_arr[i - 1] and not config.tp_set.static_tp:
                    # tp limit 이 불가한 경우 - open 이 이미, tp 를 넘은 경우
                    if open[i] < tp_arr[i]:
                        tp = open[i]
                    # tp limit 이 가능한 경우 - open 이 아직, tp 를 넘지 않은 경우
                    else:
                        tp = tp_arr[i]

                # ------ static tp ------ #
                else:
                    #   tp limit 이 불가한 경우 - open 이 이미, tp 를 넘은 경우
                    if open[i] < tp_arr[tp_j]:  # static 해놓고 decay 사용하면 dynamic 이니까
                        if config.tr_set.decay_gap != "None" and decay_remain == 0:
                            tp = open[i]  # tp_j -> open_i 를 가리키기 때문에 decay 는 한번만 진행되는게 맞음
                        else:
                            tp = tp_arr[tp_j]
                    else:
                        tp = tp_arr[tp_j]

                if tp_i == len_tps - 1:
                    exit_done = 1  # partial 을 고려해 exit_done = 1 상태는 tp_i 가 last_index 로 체결된 경우만 해당

        else:
            if high[i] >= tp_arr[tp_j]:
                # ------ dynamic tp ------ #
                if tp_arr[i] != tp_arr[i - 1] and not config.tp_set.static_tp:
                    if open[i] > tp_arr[i]:
                        tp = open[i]
                    else:
                        tp = tp_arr[i]

                # ------ static tp ------ #
                else:
                    if open[i] > tp_arr[tp_j]:
                        if config.tr_set.decay_gap != "None" and decay_remain == 0:
                            tp = open[i]
                        else:
                            tp = tp_arr[tp_j]
                    else:
                        tp = tp_arr[tp_j]

                if tp_i == len_tps - 1:
                    exit_done = 1  # partial 을 고려해 exit_done = 1 상태는 tp_i 가 last_index 로 체결된 경우만 해당

    if exit_done:
        fee += config.trader_set.limit_fee

    return exit_done, tp, fee

def check_signal_out_v3(res_df, config, open_i, i, len_df, fee, open_side, cross_on, exit_done, np_datas):

    _, _, _, close, np_timeidx = np_datas
    ex_p = None
    selection_id = config.selection_id

    # ------ timestamp ------ #
    if config.out_set.tf_exit != "None":
        if np_timeidx[i] % config.out_set.tf_exit == config.out_set.tf_exit - 1 and i != open_i:
            exit_done = -1

    # ------ rsi ------ # -> vectorize 가능함 => 추후 적용
    if config.out_set.rsi_exit:
        rsi_T = res_df['rsi_T'].to_numpy()

        if open_side == OrderSide.SELL:
            if (rsi_T[i - 1] >= 50 - config.loc_set.point.osc_band) & (rsi_T[i] < 50 - config.loc_set.point.osc_band):
                exit_done = -1
        else:
            if (rsi_T[i - 1] <= 50 + config.loc_set.point.osc_band) & (rsi_T[i] > 50 + config.loc_set.point.osc_band):
                exit_done = -1

    # ------ heikin_ashi ------ #
    # if selection_id in ['v3_3']:
    #     if open_side == OrderSide.SELL:
    #       if (ha_o[i] < ha_c[i]):# & (ha_o[i] == ha_l[i]):   # 양봉 출현
    #           exit_done = -1
    #     else:
    #       if (ha_o[i] > ha_c[i]):# & (ha_o[i] == ha_h[i]):  # 음봉 출현
    #           exit_done = -1

    # ------------ early out ------------ #
    # ------ bb ------ # --> cross_on 기능은 ide latency 개선 여부에 해당되지 않음
    if selection_id in ['v5_2']:
        bb_upper_5T = res_df['bb_upper_5T'].to_numpy()
        bb_lower_5T = res_df['bb_lower_5T'].to_numpy()

        if open_side == OrderSide.SELL:
            if close[i] < bb_lower_5T[i] < close[i - 1]:
                cross_on = 1
            if cross_on == 1 and close[i] > bb_upper_5T[i] > close[i - 1]:
                exit_done = -1
        else:
            if close[i] > bb_upper_5T[i] > close[i - 1]:
                cross_on = 1
            if cross_on == 1 and close[i] < bb_lower_5T[i] < close[i - 1]:
                exit_done = -1

    if exit_done:
        ex_p = close[i]
        fee += config.trader_set.market_fee

    return exit_done, cross_on, ex_p, fee

def check_market_out_exec_v2(config, res_df, np_timeidx, open_i, j, len_df, fee, open_side, cross_on, exit_done):
    close = res_df['close'].to_numpy()
    ex_p = None
    selection_id = config.selection_id

    # ------ timestamp ------ #
    if config.out_set.tf_exit != "None":
        if np_timeidx[j] % config.out_set.tf_exit == config.out_set.tf_exit - 1 and j != open_i:
            exit_done = 1

    # ------ rsi ------ # -> vectorize 가능함 => 추후 적용
    if config.out_set.rsi_exit:
        rsi_T = res_df['rsi_T'].to_numpy()

        if open_side == OrderSide.SELL:
            if (rsi_T[j - 1] >= 50 - config.loc_set.point.osc_band) & (rsi_T[j] < 50 - config.loc_set.point.osc_band):
                exit_done = 1
        else:
            if (rsi_T[j - 1] <= 50 + config.loc_set.point.osc_band) & (rsi_T[j] > 50 + config.loc_set.point.osc_band):
                exit_done = 1

    # ------ heikin_ashi ------ #
    # if selection_id in ['v3_3']:
    #     if open_side == OrderSide.SELL:
    #       if (ha_o[j] < ha_c[j]):# & (ha_o[j] == ha_l[j]):   # 양봉 출현
    #           exit_done = 1
    #     else:
    #       if (ha_o[j] > ha_c[j]):# & (ha_o[j] == ha_h[j]):  # 음봉 출현
    #           exit_done = 1

    # ------------ early out ------------ #
    # ------ bb ------ # --> cross_on 기능은 ide latency 개선 여부에 해당되지 않음
    if selection_id in ['v5_2']:
        bb_upper_5T = res_df['bb_upper_5T'].to_numpy()
        bb_lower_5T = res_df['bb_lower_5T'].to_numpy()

        if open_side == OrderSide.SELL:
            if close[j] < bb_lower_5T[j] < close[j - 1]:
                cross_on = 1
            if cross_on == 1 and close[j] > bb_upper_5T[j] > close[j - 1]:
                exit_done = 1
        else:
            if close[j] > bb_upper_5T[j] > close[j - 1]:
                cross_on = 1
            if cross_on == 1 and close[j] < bb_lower_5T[j] < close[j - 1]:
                exit_done = 1

    if exit_done:
        ex_p = close[j]
        fee += config.trader_set.market_fee

    return exit_done, cross_on, ex_p, fee



def check_hl_out_v2(config, i, out_j, len_df, fee, open_side, exit_done, np_datas):

    open, high, low, close, out_arr = np_datas
    ex_p = None

    if config.out_set.hl_out:
        if open_side == OrderSide.SELL:
            if high[i] >= out_arr[out_j]:  # check out only once
                exit_done = -1
        else:
            if low[i] <= out_arr[out_j]:  # check out only once
                exit_done = -1
    else:  # close_out
        if open_side == OrderSide.SELL:
            if close[i] >= out_arr[out_j]:  # check out only once
                exit_done = -1
        else:
            if close[i] <= out_arr[out_j]:  # check out only once
                ex_p = close[i]
                exit_done = -1

    if exit_done:
        if config.out_set.hl_out:
            ex_p = out_arr[out_j]
        else:
            ex_p = close[i]

        if open_side == OrderSide.SELL:
            if open[i] >= out_arr[out_j]:
                ex_p = open[i]
        else:
            if open[i] <= out_arr[out_j]:
                ex_p = open[i]

        fee += config.trader_set.market_fee

    return exit_done, ex_p, fee

def check_out(config, open_i, j, out_j, len_df, fee, open_side, exit_done, np_datas):
    o, h, l, c, out_arr = np_datas
    ex_p = None

    if config.out_set.hl_out:
        if open_side == OrderSide.SELL:
            if h[j] >= out_arr[out_j]:  # check out only once
                exit_done = 1
        else:
            if l[j] <= out_arr[out_j]:  # check out only once
                exit_done = 1
    else:  # close_out
        if open_side == OrderSide.SELL:
            if c[j] >= out_arr[out_j]:  # check out only once
                exit_done = 1
        else:
            if c[j] <= out_arr[out_j]:  # check out only once
                ex_p = c[j]
                exit_done = 1

    if exit_done:
        if config.out_set.hl_out:
            ex_p = out_arr[out_j]
        else:
            ex_p = c[j]

        if open_side == OrderSide.SELL:
            if o[j] >= out_arr[out_j]:
                ex_p = o[j]
        else:
            if o[j] <= out_arr[out_j]:
                ex_p = o[j]

        fee += config.trader_set.market_fee

    return exit_done, ex_p, fee
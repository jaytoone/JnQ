import time
from funcs.public.idep import *
from funcs.public.broker import itv_to_number


class OrderSide:  # 추후 위치 옮길 것 - colab 에 binance_file 종속할 수 없어 이곳에 임시적으로 선언함
    BUY = "BUY"
    SELL = "SELL"
    INVALID = None


def get_res_v2(res_df, open_info_df_list, ohlc_list, config_list, np_timeidx, en_ex_pairing, funcs1, idep_plot, funcs2, inversion=False, test_ratio=0.3, plot_is=True, signi=False, show_detail=False):
    """
    v1 -> v2
    1. en_ex_pairing, idep_plot 에 필요한 funcs 를 분리함, funcs1, funcs2
    """

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

    if config_list[0].tr_set.check_hlm in [0, 1]:  # 여기서 open_info 자동화하더라도, utils info 는 직접 실행해주어야함
        sample_open_idx2 = sample_open_idx1
        open_info2 = open_info1
    else:
        sample_open_idx2 = open_idx2[sample_idx2]
        open_info2 = [sample_open_info_df2[col_].to_numpy() for col_ in sample_open_info_df2.columns]

    # ------------ get paired_res ------------ #
    start_0 = time.time()
    paired_res = en_ex_pairing(res_df, [sample_open_idx1, sample_open_idx2], [open_info1, open_info2], ohlc_list, config_list, np_timeidx, funcs1, show_detail)
    # net_p1_idx_arr, p1_idx_arr, p2_idx_arr, pair_idx_arr, pair_price_arr, lvrg_arr, fee_arr, tpout_arr, tr_arr = paired_res
    # print(pair_price_arr)
    print("en_ex_pairing elapsed time :", time.time() - start_0)  # 0.37 --> 0.3660471439361572 --> 0.21(lesser if)

    # ------------ idep_plot ------------ #
    start_0 = time.time()
    high, low = ohlc_list[1:3]
    res = idep_plot(res_df, len_df, config_list[0], high, low, sample_open_info_df1, paired_res, funcs2, inversion=inversion, sample_ratio=1 - test_ratio, signi=signi)
    print("idep_plot elapsed time :", time.time() - start_0)  # 1.40452 (v6) 1.4311 (v5)

    return res


def get_res_v1(res_df, open_info_df_list, ohlc_list, config_list, np_timeidx, funcs, en_ex_pairing, idep_plot, inversion=False, test_ratio=0.3, plot_is=True, signi=False, show_detail=False):
    """
    wave_bb 를 사용하기 위한 get_res_() version
    """

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

    if config_list[0].tr_set.check_hlm in [0, 1]:  # 여기서 open_info 자동화하더라도, utils info 는 직접 실행해주어야함
        sample_open_idx2 = sample_open_idx1
        open_info2 = open_info1
    else:
        sample_open_idx2 = open_idx2[sample_idx2]
        open_info2 = [sample_open_info_df2[col_].to_numpy() for col_ in sample_open_info_df2.columns]

    # ------------ get paired_res ------------ #
    start_0 = time.time()
    paired_res = en_ex_pairing(res_df, [sample_open_idx1, sample_open_idx2], [open_info1, open_info2], ohlc_list, config_list, np_timeidx, funcs, show_detail)
    # valid_openi_arr, pair_idx_arr, pair_price_arr, lvrg_arr, fee_arr, tpout_arr = paired_res
    print("en_ex_pairing elapsed time :", time.time() - start_0)  # 0.37 --> 0.3660471439361572 --> 0.21(lesser if)

    # ------------ idep_plot ------------ #
    start_0 = time.time()
    high, low = ohlc_list[1:3]
    res = idep_plot(res_df, len_df, config_list[0], high, low, sample_open_info_df1, paired_res, inversion=inversion, sample_ratio=1 - test_ratio, signi=signi)
    print("idep_plot elapsed time :", time.time() - start_0)  # 1.40452 (v6) 1.4311 (v5)

    return res


def get_res_v10(res_df, open_info_df_list, ohlc_list, config_list, np_timeidx, funcs, inversion=False, test_ratio=0.3, plot_is=True, signi=False, show_detail=False):
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
  paired_res = en_ex_pairing_v9_31(res_df, [sample_open_idx1, sample_open_idx2], [open_info1, open_info2], ohlc_list, config_list, np_timeidx, funcs, show_detail)
  # valid_openi_arr, pair_idx_arr, pair_price_arr, lvrg_arr, fee_arr, tpout_arr = paired_res
  print("en_ex_pairing elapsed time :", time.time() - start_0)  #  0.37 --> 0.3660471439361572 --> 0.21(lesser if)

  # ------------ idep_plot ------------ #
  start_0 = time.time()
  high, low = ohlc_list[1:3]
  res = idep_plot_v18(res_df, len_df, config_list[0], high, low, sample_open_info_df1, paired_res, inversion=inversion, sample_ratio=1 - test_ratio, signi=signi)
  print("idep_plot elapsed time :", time.time() - start_0)   # 1.40452 (v6) 1.4311 (v5)

  return res


def get_res_v9_1(res_df, open_info_df_list, ohlc_list, config_list, np_timeidx, funcs, inversion=False, test_ratio=0.3, plot_is=True, signi=False, show_detail=False):
    """
  candle_game 사용시 사용하는 version
  """

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

    if config_list[0].tr_set.check_hlm in [0, 1]:  # 여기서 open_info 자동화하더라도, utils info 는 직접 실행해주어야함
        sample_open_idx2 = sample_open_idx1
        open_info2 = open_info1
    else:
        sample_open_idx2 = open_idx2[sample_idx2]
        open_info2 = [sample_open_info_df2[col_].to_numpy() for col_ in sample_open_info_df2.columns]

    # ------------ get paired_res ------------ #
    start_0 = time.time()
    paired_res = en_ex_pairing_v9_4(res_df, [sample_open_idx1, sample_open_idx2], [open_info1, open_info2], ohlc_list, config_list, np_timeidx, funcs, show_detail)
    # valid_openi_arr, pair_idx_arr, pair_price_arr, lvrg_arr, fee_arr, tpout_arr = paired_res
    print("en_ex_pairing elapsed time :", time.time() - start_0)  # 0.37 --> 0.3660471439361572 --> 0.21(lesser if)

    # ------------ idep_plot ------------ #
    start_0 = time.time()
    high, low = ohlc_list[1:3]
    res = idep_plot_v16_3(res_df, len_df, config_list[0], high, low, sample_open_info_df1, paired_res, inversion=inversion, sample_ratio=1 - test_ratio, signi=signi)
    print("idep_plot elapsed time :", time.time() - start_0)  # 1.40452 (v6) 1.4311 (v5)

    return res


def get_res_v9(res_df, open_info_df_list, ohlc_list, config_list, np_timeidx, funcs, inversion=False, test_ratio=0.3, plot_is=True, signi=False, show_detail=False):
    """
  wave_cci 사용시 가장 최신 version
  """

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

    if config_list[0].tr_set.check_hlm in [0, 1]:  # 여기서 open_info 자동화하더라도, utils info 는 직접 실행해주어야함
        sample_open_idx2 = sample_open_idx1
        open_info2 = open_info1
    else:
        sample_open_idx2 = open_idx2[sample_idx2]
        open_info2 = [sample_open_info_df2[col_].to_numpy() for col_ in sample_open_info_df2.columns]

    # ------------ get paired_res ------------ #
    start_0 = time.time()
    paired_res = en_ex_pairing_v9_4(res_df, [sample_open_idx1, sample_open_idx2], [open_info1, open_info2], ohlc_list, config_list, np_timeidx, funcs, show_detail)
    # valid_openi_arr, pair_idx_arr, pair_price_arr, lvrg_arr, fee_arr, tpout_arr = paired_res
    print("en_ex_pairing elapsed time :", time.time() - start_0)  # 0.37 --> 0.3660471439361572 --> 0.21(lesser if)

    # ------------ idep_plot ------------ #
    start_0 = time.time()
    high, low = ohlc_list[1:3]
    res = idep_plot_v16_2(res_df, len_df, config_list[0], high, low, sample_open_info_df1, paired_res, inversion=inversion, sample_ratio=1 - test_ratio, signi=signi)
    print("idep_plot elapsed time :", time.time() - start_0)  # 1.40452 (v6) 1.4311 (v5)

    return res


def get_open_info_df_v2(ep_loc_v2, res_df, np_timeidx, id_list, config_list, id_idx_list, open_num=1):
    """
    v1 -> v2
        1. <U32 dtype 으로 인한 memory allocate error 에 대응하기 위해 zone, side value 를 integer 기준으로 수정함.
    """
    start_0 = time.time()
    # ------ get mr_res, zone_arr ------ #
    short_mr_res_obj = np.array([ep_loc_v2(res_df, config_, np_timeidx, show_detail=True, ep_loc_side=OrderSide.SELL) for config_ in config_list])
    long_mr_res_obj = np.array([ep_loc_v2(res_df, config_, np_timeidx, show_detail=True, ep_loc_side=OrderSide.BUY) for config_ in config_list])
    short_open_idx_list = [np.where(res_df['short_open{}_{}'.format(open_num, id)].to_numpy() * mr_res)[0] for id, mr_res in zip(id_list, short_mr_res_obj[:, 0])]  # "point * mr_Res"
    long_open_idx_list = [np.where(res_df['long_open{}_{}'.format(open_num, id)].to_numpy() * mr_res)[0] for id, mr_res in zip(id_list, long_mr_res_obj[:, 0])]  # zip 으로 zone (str) 과 묶어서 dtype 변경됨
    print("~ ep_loc_v2 elapsed time :", time.time() - start_0)

    # ------ open_info_arr ------ #
    short_side_list = [np.full(len(list_), -1) for list_ in short_open_idx_list]
    long_side_list = [np.full(len(list_), 1) for list_ in long_open_idx_list]

    short_zone_list = [zone_res[short_open_idx] for zone_res, short_open_idx in zip(short_mr_res_obj[:, 1], short_open_idx_list)]
    long_zone_list = [zone_res[long_open_idx] for zone_res, long_open_idx in zip(long_mr_res_obj[:, 1], long_open_idx_list)]

    short_id_list = [np.full(len(list_), id) for id, list_ in zip(id_list, short_open_idx_list)]
    long_id_list = [np.full(len(list_), id) for id, list_ in zip(id_list, long_open_idx_list)]

    selected_id_idx = np.arange(len(id_idx_list))
    short_id_idx_list = [np.full(len(list_), id) for id, list_ in zip(selected_id_idx, short_open_idx_list)]
    long_id_idx_list = [np.full(len(list_), id) for id, list_ in zip(selected_id_idx, long_open_idx_list)]

    # ------ get open_info_df ------ #
    #   series 만들어서 short / long 끼리 합치고 둘이 합치고, 중복은 우선 순위 정해서 제거
    short_open_df_list = [pd.DataFrame(index=index_, data=np.vstack((data_)).T, columns=['side', 'zone', 'id', 'id_idx']) for index_, data_ in
                          zip(short_open_idx_list, zip(short_side_list, short_zone_list, short_id_list, short_id_idx_list))]
    long_open_df_list = [pd.DataFrame(index=index_, data=np.vstack((data_)).T, columns=['side', 'zone', 'id', 'id_idx']) for index_, data_ in
                         zip(long_open_idx_list, zip(long_side_list, long_zone_list, long_id_list, long_id_idx_list))]

    open_info_df = pd.concat(short_open_df_list + long_open_df_list)
    # ------ sorting + unique ------ #
    open_info_df.sort_index(inplace=True)
    # print(len(open_info_df))
    # print(len(open_info_df))
    # open_info_df.head()
    print("~ get_open_info_df elapsed time :", time.time() - start_0)
    return open_info_df[~open_info_df.index.duplicated(keep='first')]  # 먼저 순서를 우선으로 지정


def get_open_info_df(ep_loc_v2, res_df, np_timeidx, id_list, config_list, id_idx_list):
    start_0 = time.time()
    # ------ get mr_res, zone_arr ------ #
    short_mr_res_obj = np.array([ep_loc_v2(res_df, config_, np_timeidx, show_detail=True, ep_loc_side=OrderSide.SELL) for config_ in config_list])
    long_mr_res_obj = np.array([ep_loc_v2(res_df, config_, np_timeidx, show_detail=True, ep_loc_side=OrderSide.BUY) for config_ in config_list])
    short_open_idx_list = [np.where(res_df['short_open_{}'.format(id)].to_numpy() * mr_res)[0] for id, mr_res in
                           zip(id_list, short_mr_res_obj[:, 0].astype(np.float64))]  # zip 으로 zone (str) 과 묶어서 dtype 변경됨
    long_open_idx_list = [np.where(res_df['long_open_{}'.format(id)].to_numpy() * mr_res)[0] for id, mr_res in zip(id_list, long_mr_res_obj[:, 0].astype(np.float64))]

    # ------ open_info_arr ------ #
    short_side_list = [np.full(len(list_), OrderSide.SELL) for list_ in short_open_idx_list]
    long_side_list = [np.full(len(list_), OrderSide.BUY) for list_ in long_open_idx_list]

    short_zone_list = [zone_res[short_open_idx] for zone_res, short_open_idx in zip(short_mr_res_obj[:, 1], short_open_idx_list)]
    long_zone_list = [zone_res[long_open_idx] for zone_res, long_open_idx in zip(long_mr_res_obj[:, 1], long_open_idx_list)]

    short_id_list = [np.full(len(list_), id) for id, list_ in zip(id_list, short_open_idx_list)]
    long_id_list = [np.full(len(list_), id) for id, list_ in zip(id_list, long_open_idx_list)]

    selected_id_idx = np.arange(len(id_idx_list))
    short_id_idx_list = [np.full(len(list_), id) for id, list_ in zip(selected_id_idx, short_open_idx_list)]
    long_id_idx_list = [np.full(len(list_), id) for id, list_ in zip(selected_id_idx, long_open_idx_list)]

    # ------ get open_info_df ------ #
    #   series 만들어서 short / long 끼리 합치고 둘이 합치고, 중복은 우선 순위 정해서 제거
    short_open_df_list = [pd.DataFrame(index=index_, data=np.vstack((data_)).T, columns=['side', 'zone', 'id', 'id_idx']) for index_, data_ in
                          zip(short_open_idx_list, zip(short_side_list, short_zone_list, short_id_list, short_id_idx_list))]
    long_open_df_list = [pd.DataFrame(index=index_, data=np.vstack((data_)).T, columns=['side', 'zone', 'id', 'id_idx']) for index_, data_ in
                         zip(long_open_idx_list, zip(long_side_list, long_zone_list, long_id_list, long_id_idx_list))]

    open_info_df = pd.concat(short_open_df_list + long_open_df_list)
    # ------ sorting + unique ------ #
    open_info_df.sort_index(inplace=True)
    # print(len(open_info_df))
    # print(len(open_info_df))
    # open_info_df.head()
    print("get_open_info_df elapsed time :", time.time() - start_0)
    return open_info_df[~open_info_df.index.duplicated(keep='first')]  # 먼저 순서를 우선으로 지정


def en_ex_pairing_v10(res_df, open_idx_list, open_info_list, ohlc_list, config_list, np_timeidx, funcs,
                      show_detail=False):  # 이미 충분히 줄여놓은 idx 임

    """
    v9_42 -> v10 : wave_bb 을 위한 en_ex function.
        1. ep2, out 이 op_idx1 기준으로 변경
        2. p2 rejection 주석처리됨
    """

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

        tp_1_ = res_df['{}_tp_1_{}'.format(side_pos, selection_id)].to_numpy()[
            op_idx1]  # for p2_box location & p1's exipiry
        tp_0_ = res_df['{}_tp_0_{}'.format(side_pos, selection_id)].to_numpy()[op_idx1]
        tp_gap_ = res_df['{}_tp_gap_{}'.format(side_pos, selection_id)].to_numpy()[op_idx1]

        tp_ = tp_arr[op_idx1]
        ep2_ = ep2_arr[op_idx1]
        out_ = out_arr[op_idx1]

        out_1_ = res_df['{}_out_1_{}'.format(side_pos, selection_id)].to_numpy()[op_idx1]
        out_0_ = res_df['{}_out_0_{}'.format(side_pos, selection_id)].to_numpy()[op_idx1]
        out_gap_ = res_df['{}_out_gap_{}'.format(side_pos, selection_id)].to_numpy()[op_idx1]

        # if not check_net_hhm:  # this phase exist for p1 entry (net hhm sync.) in p2_platform
        exec_j, ep_j, tp_j, out_j, entry_done, en_p, fee = check_entry_v7(res_df, config, config.ep_set.entry_type,
                                                                          op_idx1, op_idx1, tp_1_, tp_gap_, len_df,
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
        # ============ p2 loop ============ #
        op_idx2 = i
        p2_confirm = 0
        while 1:
            op_idx2 += 1
            if op_idx2 >= len_df:  # res_df 의 last_index 까지 돌아야함
                break

            if open_side == OrderSide.SELL:
                if tp_1_ > close[op_idx2]:
                    p2_confirm = 1
                    break
                if tp_0_ < high[op_idx2]:  # ei_k
                    break
            else:
                if tp_1_ < close[op_idx2]:
                    p2_confirm = 1
                    break
                if tp_0_ > low[op_idx2]:  # ei_k
                    break

        if not p2_confirm:
            continue

        if show_detail:
            print("op_idx1, op_idx2 :", op_idx1, op_idx2)

        # ============ entry loop ============ #
        while 1:  # for p2's loop (allow retry)

            # ============ get p2_info ============ #
            if check_hlm in [1, 2]:
                pass
            #             open_i2 += 1  # 확인 끝났으면 조기 이탈(+1), 다음 open_idx 조사 진행
            #             if open_i2 >= len_open_idx2:  # open_i2 소진
            #                 break

            #             if show_detail:
            #               print("open_i2 :", open_i2, side_arr2[open_i2])

            # ------ check side sync. ------ #
            # if open_side != side_arr2[open_i2]:
            #   continue

            # ------ assert, op_idx2 >= exec_j ------ #
            #             op_idx2 = open_idx2[open_i2]  # open_i2 는 i 와 별개로 운영
            #             if check_hlm == 1 and allow_exit:
            #               if op_idx2 < op_idx1:
            #                 continue
            #             else:
            #               if op_idx2 < i:   # p1 execution 이후의 i 를 허용 (old, 이곳 i = op_idx1 + 1 or p2's exec_j or exit_loop's i + 1)
            #                 continue

            #             if check_hlm == 2:
            #               i = op_idx2 + 1  # open_signal 이 close_bar.shift(1) 이라고 가정하고 다음 bar 부터 체결확인한다는 의미
            #               if i >= len_df:  # res_df 의 last_index 까지 돌아야함
            #                   break

            else:
                op_idx2 = op_idx1

            if check_hlm in [1, 2]:
                # ------ check p1's expiry - Todo, priority ------ # - p2_box 생성 이전의 hl_survey
                # 1. op_idx1 ~ op_idx2 까지의 hl_check
                # if check_hlm:  # p1_hlm, p2_hlm --> Todo, 이거를 왜 p1_hlm 에도 적용했는지 잘 모르겠음
                # if op_idx1 < op_idx2:
                #   expire, touch_idx = expiry_p1(res_df, config, op_idx1, op_idx2, tp_1_, tp_0_, tp_gap_, ohlc_list[1:3], open_side)
                #   if expire:   # p1's expiry
                #       if show_detail:
                #         print("expiry_p1, touch_idx = {} : break".format(touch_idx))
                #       i = touch_idx  #  + 1  --> 이거 아닌것 같음 # op_idx1 과 op_idx2 사이의 op_idx1' 을 살리기 위함, 즉 바로 다음 op_idx1 로 회귀 (건너뛰지 않고)
                #       open_i2 = prev_open_i2
                #       break   # change op_idx1

                if check_hlm == 2:
                    # # ------ p2 point_validation - vectorization unavailable ------ # p1_loop 로 return 되는 정확한 i 를 반환하기 위해서 expiry_p1 에 순서 양보  # Todo, 새로운 tp, ep, out 에 대한 처리 필요 (p1_hlm 사용시)
                    # if open_side == OrderSide.SELL:
                    #   if not (tp_ < ep2_):  # tr_set validation & reject hl_out open_exec.
                    #     break  # change op_idx1
                    #   elif not (ep2_ < out_ and close[op_idx2] < out_):
                    #     if show_detail:
                    #       print("point validation : continue")
                    #     continue  # change op_idx2
                    # else:
                    #   if not (tp_ > ep2_):
                    #     break
                    #   elif not (ep2_ > out_ and close[op_idx2] > out_):
                    #     if show_detail:
                    #       print("point validation : continue")
                    #     continue

                    # ------ p2_box location ------ #
                    # if open_side == OrderSide.SELL:
                    #   if not ((tp_1_ + tp_gap_ * config.tr_set.p2_box_k1 <= out_1_) and (out_0_ <= tp_0_ - tp_gap_ * config.tr_set.p2_box_k2)):  # tp1, tp0 에 닿으면 expiry
                    #   # if not ((tp_1_ + tp_gap_ * config.tr_set.p2_box_k1 >= out_1_) and (out_0_ <= tp_0_ - tp_gap_ * config.tr_set.p2_box_k2)):  # tp1, tp0 에 닿으면 expiry
                    #     if show_detail:
                    #         print("p2_box rejection : continue")
                    #     continue
                    #   else:
                    #     # ------ p1p2_low ------ #
                    #     if not high[op_idx1:op_idx2 + 1].max() < tp_0_ - tp_gap_ * config.tr_set.p1p2_low:
                    #       if show_detail:
                    #         print("p1p2_low rejection : continue")
                    #       continue
                    # else:
                    #   if not ((tp_1_ - tp_gap_ * config.tr_set.p2_box_k1 >= out_1_) and (out_0_ >= tp_0_ + tp_gap_ * config.tr_set.p2_box_k2)):
                    #   # if not ((tp_1_ - tp_gap_ * config.tr_set.p2_box_k1 <= out_1_) and (out_0_ >= tp_0_ + tp_gap_ * config.tr_set.p2_box_k2)):
                    #     if show_detail:
                    #         print("p2_box rejection : continue")
                    #     continue
                    #   else:
                    #     # ------ p1p2_low ------ #
                    #     if not low[op_idx1:op_idx2 + 1].min() > tp_0_ + tp_gap_ * config.tr_set.p1p2_low:
                    #       if show_detail:
                    #         print("p1p2_low rejection : continue")
                    #       continue

                    # ------ check p2's expiry ------ # - 현재, op_idx2 기준의 ep2_arr 을 사용 중임.
                    # Todo, tp_j, out_j 는 op_idx1 를 따르기 때문에 해당 phase 에서 비워둠.
                    # Todo, tp_1 & tp_gap 사용이 맞을 것으로 봄
                    exec_j, ep_j, _, _, entry_done, en_p, fee = check_entry_v7(res_df, config,
                                                                               config.ep_set.point2.entry_type, op_idx1,
                                                                               op_idx2, tp_1_, tp_gap_, len_df,
                                                                               open_side,
                                                                               [*ohlc_list, ep2_arr], expiry_p2)

                    i = exec_j  # = entry_loop 를 돌고 나온 e_j
                    if not entry_done:  # p2's expiry
                        if show_detail:
                            print("expiry_p2, i = {} : break".format(i))
                        # continue  # change op_idx2
                        break  # change op_idx1

                    # ------ devectorized tr_calc ------ # - en_p 에 대해 하는게 맞을 것으로봄
                    if open_side == OrderSide.SELL:
                        tr_ = abs((en_p / tp_ - config.trader_set.limit_fee - 1) / (
                                en_p / out_ - config.trader_set.market_fee - 1))
                    else:
                        tr_ = abs((tp_ / en_p - config.trader_set.limit_fee - 1) / (
                                out_ / en_p - config.trader_set.market_fee - 1))

                        # ------ tr_threshold ------ #
                    if config.loc_set.point2.short_tr_thresh != "None":
                        if open_side == OrderSide.SELL:
                            if tr_ < config.loc_set.point2.short_tr_thresh:
                                if show_detail:
                                    print("tr_threshold : break")
                                break
                        else:
                            if tr_ < config.loc_set.point2.long_tr_thresh:
                                if show_detail:
                                    print("tr_threshold : break")
                                break

            if not allow_exit:  # p1_hlm 의 경우, 한번 out 되면 price 가 wave_range 에 닿기전까지 retrade 를 허용하지 않는다. (expiry_p1 을 이용해 op_idx1 을 변경할 것)
                if show_detail:
                    print("allow_exit = {} : break".format(allow_exit))
                break

            if check_hlm in [0, 1]:
                tr_ = tr_arr[op_idx1]

            # ------ leverage ------ #
            # out = out_arr[out_j]  # lvrg_set use out on out_j (out_j shoud be based on p2)
            leverage = lvrg_set(res_df, config, open_side, en_p, out_, fee)  # res_df 변수 사용됨 - 주석 처리 된 상태일뿐
            if leverage is None:
                if show_detail:
                    print("leverage is None : break")
                if check_hlm:
                    break  # change op_idx2
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
                    exit_done, cross_on, ex_p, fee = check_signal_out_v3(res_df, config, open_i2, i, len_df, fee,
                                                                         open_side, cross_on, exit_done,
                                                                         [*ohlc_list, np_timeidx])
                # ------ hl_out ------ #
                if config.out_set.hl_out:
                    if not exit_done:  # and i != len_df - 1:
                        exit_done, ex_p, fee = check_hl_out_v2(config, i, out_j, len_df, fee, open_side, exit_done,
                                                               [*ohlc_list, out_arr])

                # ------------ tp ------------ #
                if not config.tp_set.non_tp and i != exec_j:
                    if not exit_done:
                        exit_done, ex_p, fee = check_limit_tp_exec(res_df, config, open_i2, i, tp_j, len_df, fee,
                                                                   open_side, exit_done,
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
                if check_hlm == 1:  # exit only once in p1_hlm mode
                    allow_exit = 0

                if show_detail:
                    print("exit_done = {}, i = {} : break".format(exit_done, i))  # break p2 loop on csd_mode
                break  # change op_idx1

        if i >= len_df:  # or open_i >= len_open_idx:  # res_df 의 last_index 까지 돌아야함
            break
        else:
            continue

    return np.array(net_p1_idx_list), np.array(p1_idx_list), np.array(p2_idx_list), np.array(pair_idx_list), np.array(
        pair_price_list), np.array(lvrg_list), np.array(
        fee_list), np.array(tpout_list), np.array(tr_list)


def en_ex_pairing_v9_5(res_df, open_idx_list, open_info_list, ohlc_list, config_list, np_timeidx, funcs, show_detail=False):  # 이미 충분히 줄여놓은 idx 임

    """
    v9_42 -> v9_5
        1. add reatlime ep_loc
    """

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

    temp_lb_period = 700  # 30T x 21 = 630 -> 넉넉하게 700 rows 로 설정.

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

        # ------ realtime ep_loc ------ #
        if not op_idx1 >= temp_lb_period:
            continue

        temp_df = res_df.iloc[op_idx1 - temp_lb_period:op_idx1 + 1]  # ':' 이전에 + 1 하지 않은 이유는 구지 연산량을 늘이기 싫어서.
        temp_htf_df = to_htf(temp_df, '30T', '1h')

        # print(temp_df.close.tail())  # validated !
        # print(temp_htf_df.tail())

        realtime_cci_30T20 = talib.CCI(temp_htf_df.high, temp_htf_df.low, temp_htf_df.close, timeperiod=20)

        # print(realtime_cci_30T20[-2:])  # validated !

        if pd.isnull(realtime_cci_30T20[-2]):
            continue

        if open_side == OrderSide.SELL:
            if not realtime_cci_30T20[-1] < realtime_cci_30T20[-2]:
                if show_detail:
                    print("realtime_cci phase not in condition")
                continue
        else:
            if not realtime_cci_30T20[-1] > realtime_cci_30T20[-2]:
                if show_detail:
                    print("realtime_cci phase not in condition")
                continue

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
        exec_j, ep_j, tp_j, out_j, entry_done, en_p, fee = check_entry_v6_1(res_df, config,
                                                                            config.ep_set.entry_type, op_idx1, len_df,
                                                                            open_side, [*ohlc_list, ep1_arr], expiry_p2)
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
                    exec_j, ep_j, _, out_j, entry_done, en_p, fee = check_entry_v6_1(res_df, config, config.ep_set.point2.entry_type,
                                                                                     op_idx2, len_df, open_side,
                                                                                     [*ohlc_list, ep2_arr], expiry_p2)
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


def en_ex_pairing_v9_45(res_df, open_idx_list, open_info_list, ohlc_list, config_list, np_timeidx, funcs, show_detail=False):  # 이미 충분히 줄여놓은 idx 임

    """
    v9_44 -> v9_45
        1. add config.loc_set.point1.p2_cnt
        2. len_df - 1's index open 은 허용하지 않는다.
    """

    open_info1, open_info2 = open_info_list
    side_arr1, _, _, id_idx_arr1 = open_info1
    side_arr2, _, _, _ = open_info2

    expiry_p1p2, expiry, lvrg_set, check_entry, check_signal_out, check_hl_out, check_limit_tp_exec = funcs

    net_p1_idx_list, p1_idx_list, p2_idx_list, pair_idx_list, pair_price_list, lvrg_list, fee_list, tpout_list, tr_list = [[] for li in range(9)]
    len_df = len(res_df)

    open, high, low, close = ohlc_list

    open_idx1, open_idx2 = open_idx_list
    len_open_idx1 = len(open_idx1)
    len_open_idx2 = len(open_idx2)
    i, open_i1, open_i2 = 0, -1, -1  # i for total_res_df indexing

    while 1:

        # ------------ p1 phase ------------ #

        # Todo,
        #   1. (갱신) p1's open_i + 1 과 op_idx 를 꺼내오는 건, eik1 또는 tp 체결의 경우만 해당됨,
        #   2. out 의 경우 p2's op_idx 기준으로 retry 필요
        #     a. 또한, p2's op_idx > p1's op_idx

        # ------ 1. get p1_info ------ #
        # if eik1 or tp_done or first loop:
        open_i1 += 1  # 확인 끝났으면 조기 이탈(+1), 다음 open_idx 조사 진행
        if open_i1 >= len_open_idx1:
            break

        if show_detail:
            print("open_i1 : {}, side_arr1 : {}".format(open_i1, side_arr1[open_i1]))

        op_idx1 = open_idx1[open_i1]  # open_i1 는 i 와 별개로 운영
        if op_idx1 < i:  # i = 이전 거래 끝난후의 res_df index - "거래 종료후 거래 시작", '<' : 거래 종료시점 진입 가능하다는 의미
            continue

        # ------ 2. set loop index i ------ #
        i = op_idx1  # + 1 --> op_idx1 = op_idx2 가능함 # open_signal 이 close_bar.shift(1) 이라고 가정하고 다음 bar 부터 체결확인한다는 의미
        if i >= len_df - 1:  # res_df 의 last_index open 은 제외 (이후의 수식에서 error 발생하기 때문)
            break

        # ------ 3. get open info ------ #
        #            a. ID 별로 수행하기 위해 selection_id, config 호출함.
        open_side_num = side_arr1[open_i1]
        id_idx = id_idx_arr1.astype(int)[open_i1]  # indexing 을 위해 integer 로 변환.
        config = config_list[id_idx]
        selection_id = config.selection_id
        check_hlm = config.tr_set.check_hlm

        open_side = OrderSide.SELL if open_side_num == -1 else OrderSide.BUY
        side_pos = 'short' if open_side == OrderSide.SELL else 'long'  # utils paper 접근을 위한 long / short string.
        if show_detail:
            print("------------ op_idx1 : {} {} ------------".format(op_idx1, open_side))

        # if show_detail:
        #   print("check_hlm :", check_hlm)

        # ------ 4. load util paper data ------ #
        """ 
        tr_set_idx initialize.
            1. j 를 둔 이유는 본래 dynamic_tp / out 을 가능케 하기 위함이였음.
                a. exec_j : open 체결 index
                b. ep_j : entry_price 기준 index
                c. tp_j : tp_price 기준 index
                d. out_j : out_price 기준 index
        """
        ep_j, tp_j, out_j = op_idx1, op_idx1, op_idx1  # tr_set p1, p2 에 가변적으로 기준할 수 있도록 구성함.
        p1_tr_set_idx = (ep_j, tp_j, out_j)

        tp_arr = res_df['{}_tp_{}'.format(side_pos, selection_id)].to_numpy()
        ep1_arr = res_df['{}_ep1_{}'.format(side_pos, selection_id)].to_numpy()
        ep2_arr = res_df['{}_ep2_{}'.format(side_pos, selection_id)].to_numpy()
        out_arr = res_df['{}_out_{}'.format(side_pos, selection_id)].to_numpy()

        tr_arr = res_df['{}_tr_{}'.format(side_pos, selection_id)].to_numpy()  # just for p1_hhm

        tp_1_ = res_df['{}_tp_1_{}'.format(side_pos, selection_id)].to_numpy()[tp_j]  # for p2_box location & p1's exipiry
        tp_0_ = res_df['{}_tp_0_{}'.format(side_pos, selection_id)].to_numpy()[tp_j]
        tp_gap_ = res_df['{}_tp_gap_{}'.format(side_pos, selection_id)].to_numpy()[tp_j]

        # if not check_net_hhm:  # this phase exist for p1 entry (net hhm sync.) in p2_platform
        exec_j, entry_done, en_p, fee = check_entry(res_df, config, config.ep_set.entry_type, op_idx1, p1_tr_set_idx, len_df, open_side, [*ohlc_list, ep1_arr], expiry)

        i = exec_j  # = entry_loop 를 돌고 나온 e_j
        if not entry_done:
            if show_detail:
                print("p1's expiry : continue")
            continue
            # else:
        #   tp_j = op_idx1

        # 1.  p2 phase
        prev_open_i2 = open_i2
        net_p1_idx_list.append(op_idx1)
        # if check_hlm in [0, 1]:
        #   i = op_idx1  # allow op_idx2 = op_idx1
        allow_exit = 1
        p2_cnt = 0

        while 1:

            if check_hlm in [1, 2]:
                open_i2 += 1  # 확인 끝났으면 조기 이탈(+1), 다음 open_idx 조사 진행
                if open_i2 >= len_open_idx2:  # open_i2 소진
                    if show_detail:
                        print("open_i2 >= len_open_idx2, open_i2 소진 : break")
                    break

                # a. check side sync.
                if side_arr1[open_i1] != side_arr2[open_i2]:
                    if show_detail:
                        print("side check rejection, open_i2 {}, side_arr2 {}".format(open_i2, side_arr2[open_i2]))
                    continue

                # b. assert, op_idx2 >= exec_j
                op_idx2 = open_idx2[open_i2]  # open_i2 는 i 와 별개로 운영
                if check_hlm == 1 and allow_exit:
                    if op_idx2 < op_idx1:
                        if show_detail:
                            print("check_hlm 1's allow_exit rejection, op_idx2 {} < op_idx1 {}".format(op_idx2, op_idx1))
                        continue
                else:
                    if op_idx2 < i:  # p1 execution 이후의 i 를 허용 (old, 이곳 i = op_idx1 + 1 or p2's exec_j or exit_loop's i + 1)
                        if show_detail:
                            print("op_idx2 {} < i {} : continue".format(op_idx2, i))
                        continue

                if check_hlm == 2:
                    i = op_idx2 + 1  # open_signal 이 close_bar.shift(1) 이라고 가정하고 다음 bar 부터 체결확인한다는 의미
                    if i >= len_df - 1:  # res_df 의 last_index open 은 제외 (이후의 수식에서 error 발생하기 때문)
                        break

                if show_detail:
                    print("op_idx1 : {} op_idx2 : {}".format(op_idx1, op_idx2))

                p2_cnt += 1

                # c. p2_cnt
                if config.loc_set.point2.p2_cnt is not None:
                    if p2_cnt <= config.loc_set.point2.p2_cnt:
                        if show_detail:
                            print("p2_cnt : {}".format(p2_cnt))
                        continue

            else:
                op_idx2 = op_idx1

            # ------ 2. load util paper data for p2  ------ #
            ep_j, tp_j, out_j = op_idx1, op_idx1, op_idx1
            p2_tr_set_idx = (ep_j, tp_j, out_j)

            ep2_ = ep2_arr[ep_j]
            tp_ = tp_arr[tp_j]
            out_ = out_arr[out_j]

            out_1_ = res_df['{}_out_1_{}'.format(side_pos, selection_id)].to_numpy()[out_j]
            out_0_ = res_df['{}_out_0_{}'.format(side_pos, selection_id)].to_numpy()[out_j]
            out_gap_ = res_df['{}_out_gap_{}'.format(side_pos, selection_id)].to_numpy()[out_j]

            # ------ const. for p2_wave ------ #
            wave_itv1 = config.tr_set.wave_itv1
            wave_period1 = config.tr_set.wave_period1
            wave_itv2 = config.tr_set.wave_itv2
            wave_period2 = config.tr_set.wave_period2

            if check_hlm in [1, 2]:

                # ------ check p1's expiry ------ # - p2_box 생성 이전의 hl_survey
                # 1. op_idx1 ~ op_idx2 까지의 expiry check (high & low 둘다)
                #     a. if check_hlm:  # p1_hlm, p2_hlm --> Todo, 이거를 왜 p1_hlm 에도 적용했는지 잘 모르겠음
                if op_idx1 < op_idx2:
                    expire, touch_idx = expiry_p1p2(res_df, config, op_idx1, op_idx2, tp_1_, tp_0_, tp_gap_, ohlc_list[1:3], open_side)
                    if expire:  # p1's expiry
                        if show_detail:
                            print("expiry_p1p2, touch_idx = {} : break".format(touch_idx))
                        i = touch_idx  # + 1  --> 이거 아닌것 같음 # op_idx1 과 op_idx2 사이의 op_idx1' 을 살리기 위함, 즉 바로 다음 op_idx1 로 회귀 (건너뛰지 않고)
                        open_i2 = prev_open_i2
                        break  # change op_idx1

                if check_hlm == 2:

                    """
                    p2 point_validation - vectorization unavailable
                        1. p2 로 wave_unit 을 사용할 경우만, p2 wave_validation & wave_box location 사용할 것.
                        2. p1_loop 로 return 되는 정확한 i 를 반환하기 위해서 expiry_p1p2 뒤에 배치함
                        3. Todo - 새로운 tp, ep, out 에 대한 처리 필요 (p1_hlm 사용시)                        
                    """

                    # ------ p2_wave validation : 정확한 뜻을 아직 잘 모르겠음. ------ #
                    #                     if open_side == OrderSide.SELL:
                    #                         wave_co_post_idx = res_df['wave_co_post_idx_fill_{}{}'.format(wave_itv2, wave_period2)].to_numpy()[op_idx2]
                    #                         if not (op_idx1 < wave_co_post_idx):
                    #                             if show_detail:
                    #                                 print("p2_wave validation : continue")
                    #                             continue  # change op_idx2

                    #                         # --- p2_wave high validation --- #
                    #                         # wave_high_fill1_ = res_df['wave_high_fill_{}{}'.format(wave_itv1, wave_period1)].to_numpy()[op_idx1]
                    #                         # wave_high_fill2_ = res_df['wave_high_fill_{}{}'.format(wave_itv2, wave_period2)].to_numpy()[op_idx2]
                    #                         # if not (wave_high_fill1_ >= wave_high_fill2_):
                    #                         #   if show_detail:
                    #                         #     print("p2_wave high validation : continue")
                    #                         #   continue  # change op_idx2

                    #                     else:
                    #                         wave_cu_post_idx = res_df['wave_cu_post_idx_fill_{}{}'.format(wave_itv2, wave_period2)].to_numpy()[op_idx2]
                    #                         if not (op_idx1 < wave_cu_post_idx):
                    #                             if show_detail:
                    #                                 print("p2_wave validation : continue")
                    #                             continue  # change op_idx2

                    #                         # --- p2_wave low validation --- #
                    #                         # wave_low_fill1_ = res_df['wave_low_fill_{}{}'.format(wave_itv1, wave_period1)].to_numpy()[op_idx1]
                    #                         # wave_low_fill2_ = res_df['wave_low_fill_{}{}'.format(wave_itv2, wave_period2)].to_numpy()[op_idx2]
                    #                         # if not (wave_low_fill1_ <= wave_low_fill2_):
                    #                         #   if show_detail:
                    #                         #     print("p2_wave low validation : continue")
                    #                         #   continue  # change op_idx2

                    #                     # ------ p2 wave_box location ------ #
                    # if open_side == OrderSide.SELL:
                    #     if not ((tp_1_ + tp_gap_ * config.tr_set.p2_box_k1 <= out_1_) and (
                    #             out_0_ <= tp_0_ - tp_gap_ * config.tr_set.p2_box_k2)):  # tp1, tp0 에 닿으면 expiry
                    #         # if not ((tp_1_ + tp_gap_ * config.tr_set.p2_box_k1 >= out_1_) and (out_0_ <= tp_0_ - tp_gap_ * config.tr_set.p2_box_k2)):  # tp1, tp0 에 닿으면 expiry
                    #         if show_detail:
                    #             print("p2_box rejection : continue")
                    #         continue
                    # else:
                    #     if not ((tp_1_ - tp_gap_ * config.tr_set.p2_box_k1 >= out_1_) and (out_0_ >= tp_0_ + tp_gap_ * config.tr_set.p2_box_k2)):
                    #         # if not ((tp_1_ - tp_gap_ * config.tr_set.p2_box_k1 <= out_1_) and (out_0_ >= tp_0_ + tp_gap_ * config.tr_set.p2_box_k2)):
                    #         if show_detail:
                    #             print("p2_box rejection : continue")
                    #         continue

                    # ------ tr_set validation & reject hl_out open_exec. ------ #
                    if open_side == OrderSide.SELL:
                        if not (tp_ < ep2_):
                            break  # change op_idx1
                        elif not (ep2_ < out_ and close[op_idx2] < out_):
                            if show_detail:
                                print("p2 tr_set validation : continue")
                            continue  # change op_idx2
                    else:
                        if not (tp_ > ep2_):
                            break
                        elif not (ep2_ > out_ and close[op_idx2] > out_):
                            if show_detail:
                                print("p2 tr_set validation : continue")
                            continue

                    # ------ p1p2_low ------ #
                    if open_side == OrderSide.SELL:
                        if not high[op_idx1:op_idx2 + 1].max() < tp_0_ - tp_gap_ * config.tr_set.p1p2_low:
                            if show_detail:
                                print("p1p2_low rejection : continue")
                            continue
                    else:
                        if not low[op_idx1:op_idx2 + 1].min() > tp_0_ + tp_gap_ * config.tr_set.p1p2_low:
                            if show_detail:
                                print("p1p2_low rejection : continue")
                            continue

                    # ------ check p2's expiry ------ # - 현재, op_idx2 기준의 ep2_arr 을 사용 중임.
                    """
                    Caution : tr_set_idx 상황에 따라 잘 확인할 것
                    """
                    exec_j, entry_done, en_p, fee = check_entry(res_df, config, config.ep_set.point2.entry_type,
                                                                op_idx2, p2_tr_set_idx, len_df, open_side,
                                                                [*ohlc_list, ep2_arr], expiry)
                    i = exec_j  # = entry_loop 를 돌고 나온 e_j
                    if not entry_done:  # p2's expiry
                        if show_detail:
                            print("expiry, i = {} at p2's : continue".format(i))
                        continue  # change op_idx2

                    # ------ devectorized tr_calc ------ #
                    #    1. en_p 에 대해 하는게 맞을 것으로봄
                    #    2. tr_thresh 와 무관하게 있어야할 phase.
                    #    Todo, fee 계산에 오류가 있는 걸로 보임 => limit_fee 를 앞에 더해주어야할 것.
                    if open_side == OrderSide.SELL:
                        tr_ = abs((en_p / tp_ - config.trader_set.limit_fee - 1) / (en_p / out_ - config.trader_set.market_fee - 1))
                    else:
                        tr_ = abs((tp_ / en_p - config.trader_set.limit_fee - 1) / (out_ / en_p - config.trader_set.market_fee - 1))

                    # ------ tr_threshold ------ #
                    if config.loc_set.point2.tr_thresh_short != "None":
                        if open_side == OrderSide.SELL:
                            if tr_ < config.loc_set.point2.tr_thresh_short:
                                if show_detail:
                                    print("tr_threshold : continue")
                                continue
                        else:
                            if tr_ < config.loc_set.point2.tr_thresh_long:
                                if show_detail:
                                    print("tr_threshold : continue")
                                continue

            # 1. allow_exit = "p1_hlm 의 경우, 한번 out 되면 price 가 "wave_range 에 닿기전까지" retrade 를 허용하지 않음" (expiry_p1p2 을 이용해 op_idx1 을 변경할 것)
            #     a. while phase 내부에 if not allow_exit 을 위치한 이유 : "wave_range 에 닿기전까지" 를 구현하기 위해서.
            if not allow_exit:
                if show_detail:
                    print("allow_exit = {} : continue".format(allow_exit))
                continue

            if check_hlm in [0, 1]:
                tr_ = tr_arr[op_idx1]

            # ------ leverage ------ #
            # out = out_arr[out_j]  # lvrg_set use out on out_j (out_j shoud be based on p2)
            leverage, liqd_p = lvrg_set(res_df, config, open_side, en_p, out_, fee)  # res_df 변수 사용됨 - 주석 처리 된 상태일뿐

            if leverage is None:
                if show_detail:
                    print("leverage is None : continue")
                if check_hlm:
                    continue  # change op_idx2
                else:
                    break  # change op_idx1

            # ------------ exit phase ------------ #
            exit_done, cross_on = 0, 0

            # ------ check tpout_onexec ------ #
            if config.ep_set.entry_type == "LIMIT":
                if config.tp_set.tp_onexec:
                    tp_j = exec_j
                if config.out_set.out_onexec:
                    out_j = exec_j

            while 1:
                # dynamic tp / out 을 사용하고 싶은 경우
                if not config.tp_set.static_tp:
                    tp_j = i
                if not config.out_set.static_out:
                    out_j = i

                # ------------ 1. out ------------ #  # out 우선 (보수적 검증)
                # ------ a. signal_out ------ #
                if not exit_done:
                    exit_done, cross_on, ex_p, fee = check_signal_out(res_df, config, open_i2, i, len_df, fee, open_side, cross_on, exit_done, [*ohlc_list, np_timeidx])
                    # ------ b. hl_out ------ #
                # if config.out_set.hl_out: # --> liqd_p 도입으로 hl_out 내부에서 liqd_p 조건부 수행은 필수불가결이다.
                if not exit_done:  # and i != len_df - 1:
                    exit_done, ex_p, fee = check_hl_out(config, i, out_j, len_df, fee, open_side, exit_done, [*ohlc_list, out_arr, liqd_p])

                # ------------ 2. tp ------------ #
                if not config.tp_set.non_tp and i != exec_j:
                    if not exit_done:
                        # 1. partial_tps 를 고려해 [tp_arr, ...] 형태 사용함.
                        # 2. if config.tp_set.tp_type in ['LIMIT']:  # 'BOTH' -> 앞으로는, LIMIT 밖에 없을거라 주석처리함
                        # 3. Todo, open_i2 는 deacy 기능을 위해 도입한 것 (추후 사용시 재확인)
                        exit_done, ex_p, fee = check_limit_tp_exec(res_df, config, open_i2, i, tp_j, len_df, fee, open_side, exit_done, [*ohlc_list, [tp_arr]])

                if exit_done:  # 이 phase 는 exit_phase 뒤에도 있어야할 것 - entry_done var. 사용은 안하겠지만
                    # ------ 3. append dynamic result vars. ------ #
                    p1_idx_list.append(op_idx1)  # side, zone, start_ver arr 모두 openi_list 로 접근하기 위해 open_i 를 담음
                    p2_idx_list.append(op_idx2)
                    pair_idx_list.append([exec_j, i])  # entry & exit (체결 기준임)
                    pair_price_list.append([en_p, ex_p])
                    lvrg_list.append(leverage)
                    fee_list.append(fee)
                    tr_list.append(tr_)  # Todo, tr vectorize 불가함, 직접 구해주어야할 건데.. (오래걸리지 않을까 --> tr_set 데이터만 모아서 vecto 계산이 나을 것)

                    # for tpout_line plot_check & get_pr calc.
                    if exit_done == 2:
                        tpout_list.append([tp_arr[tp_j], liqd_p])
                    else:
                        tpout_list.append([tp_arr[tp_j], out_arr[out_j]])

                    # open_i += 1  # 다음 open_idx 조사 진행
                    break

                # 1. 아래있으면, 체결 기준부터 tp, out 허용 -> tp 가 entry_idx 에 체결되는게 다소 염려되기는 함, 일단 진행 (그런 case 가 많지 않았으므로)
                # 2. 위에있으면, entry 다음 tick 부터 exit 허용
                i += 1
                if i >= len_df:  # res_df 의 last_index 까지 돌아야함
                    break

            if i >= len_df:  # res_df 의 last_index 까지 돌아야함
                break

            """
            exit_done description            
                1. 1 : tp_done
                    a. check_hlm 여부와 무관하게 outer loop 의 op_idx1 을 변경 가능하도록함.
                2. -1 : out done
                2. 2 : liquidation done
                3. 0 : database done                
            """

            if exit_done == 1:
                if show_detail:
                    print("exit_done = {}, i = {} : break".format(exit_done, i))
                break  # change op_idx1
            else:
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

        if i >= len_df:  # or open_i >= len_open_idx:  # res_df 의 last_index 까지 돌아야함
            break
        else:
            continue

    return np.array(net_p1_idx_list), np.array(p1_idx_list), np.array(p2_idx_list), np.array(pair_idx_list), np.array(pair_price_list), np.array(lvrg_list), np.array(fee_list), np.array(
        tpout_list), np.array(tr_list)


def en_ex_pairing_v9_44(res_df, open_idx_list, open_info_list, ohlc_list, config_list, np_timeidx, funcs, show_detail=False):  # 이미 충분히 줄여놓은 idx 임

    """
    v9_43 -> v9_44
        1. 내부 version 관리 함수들을 모두 외부 참조로 구성함
            a. version 에 가변적으로 대응하기 위함임.
        2. liqd_p 기능 도입함.
        3. p2_tr_set_idx 직접 지정하도록 구성함.
        4. integer type 으로 수정된 side_arr 를 수용하기 위해 코드 변경 진행함.
    """

    open_info1, open_info2 = open_info_list
    side_arr1, _, _, id_idx_arr1 = open_info1
    side_arr2, _, _, _ = open_info2

    expiry_p1p2, expiry, lvrg_set, check_entry, check_signal_out, check_hl_out, check_limit_tp_exec = funcs

    net_p1_idx_list, p1_idx_list, p2_idx_list, pair_idx_list, pair_price_list, lvrg_list, fee_list, tpout_list, tr_list = [[] for li in range(9)]
    len_df = len(res_df)

    open, high, low, close = ohlc_list

    open_idx1, open_idx2 = open_idx_list
    len_open_idx1 = len(open_idx1)
    len_open_idx2 = len(open_idx2)
    i, open_i1, open_i2 = 0, -1, -1  # i for total_res_df indexing

    while 1:

        # ------------ p1 phase ------------ #

        # Todo,
        #   1. (갱신) p1's open_i + 1 과 op_idx 를 꺼내오는 건, eik1 또는 tp 체결의 경우만 해당됨,
        #   2. out 의 경우 p2's op_idx 기준으로 retry 필요
        #     a. 또한, p2's op_idx > p1's op_idx

        # ------ 1. get p1_info ------ #
        # if eik1 or tp_done or first loop:
        open_i1 += 1  # 확인 끝났으면 조기 이탈(+1), 다음 open_idx 조사 진행
        if open_i1 >= len_open_idx1:
            break

        if show_detail:
            print("open_i1 : {}, side_arr1 : {}".format(open_i1, side_arr1[open_i1]))

        op_idx1 = open_idx1[open_i1]  # open_i1 는 i 와 별개로 운영
        if op_idx1 < i:  # i = 이전 거래 끝난후의 res_df index - "거래 종료후 거래 시작", '<' : 거래 종료시점 진입 가능하다는 의미
            continue

        # ------ 2. set loop index i ------ #
        i = op_idx1  # + 1 --> op_idx1 = op_idx2 가능함 # open_signal 이 close_bar.shift(1) 이라고 가정하고 다음 bar 부터 체결확인한다는 의미
        if i >= len_df:  # res_df 의 last_index 까지 돌아야함
            break

        # ------ 3. get open info ------ #
        #            a. ID 별로 수행하기 위해 selection_id, config 호출함.
        open_side_num = side_arr1[open_i1]
        id_idx = id_idx_arr1.astype(int)[open_i1]  # indexing 을 위해 integer 로 변환.
        config = config_list[id_idx]
        selection_id = config.selection_id
        check_hlm = config.tr_set.check_hlm

        open_side = OrderSide.SELL if open_side_num == -1 else OrderSide.BUY
        side_pos = 'short' if open_side == OrderSide.SELL else 'long'  # utils paper 접근을 위한 long / short string.
        if show_detail:
            print("------------ op_idx1 : {} {} ------------".format(op_idx1, open_side))

        # if show_detail:
        #   print("check_hlm :", check_hlm)

        # ------ 4. load util paper data ------ #
        """ 
        tr_set_idx initialize.
            1. j 를 둔 이유는 본래 dynamic_tp / out 을 가능케 하기 위함이였음.
                a. exec_j : open 체결 index
                b. ep_j : entry_price 기준 index
                c. tp_j : tp_price 기준 index
                d. out_j : out_price 기준 index
        """
        ep_j, tp_j, out_j = op_idx1, op_idx1, op_idx1  # tr_set p1, p2 에 가변적으로 기준할 수 있도록 구성함.
        p1_tr_set_idx = (ep_j, tp_j, out_j)

        tp_arr = res_df['{}_tp_{}'.format(side_pos, selection_id)].to_numpy()
        ep1_arr = res_df['{}_ep1_{}'.format(side_pos, selection_id)].to_numpy()
        ep2_arr = res_df['{}_ep2_{}'.format(side_pos, selection_id)].to_numpy()
        out_arr = res_df['{}_out_{}'.format(side_pos, selection_id)].to_numpy()

        tr_arr = res_df['{}_tr_{}'.format(side_pos, selection_id)].to_numpy()  # just for p1_hhm

        tp_1_ = res_df['{}_tp_1_{}'.format(side_pos, selection_id)].to_numpy()[tp_j]  # for p2_box location & p1's exipiry
        tp_0_ = res_df['{}_tp_0_{}'.format(side_pos, selection_id)].to_numpy()[tp_j]
        tp_gap_ = res_df['{}_tp_gap_{}'.format(side_pos, selection_id)].to_numpy()[tp_j]

        # if not check_net_hhm:  # this phase exist for p1 entry (net hhm sync.) in p2_platform
        exec_j, entry_done, en_p, fee = check_entry(res_df, config, config.ep_set.entry_type, op_idx1, p1_tr_set_idx, len_df, open_side, [*ohlc_list, ep1_arr], expiry)

        i = exec_j  # = entry_loop 를 돌고 나온 e_j
        if not entry_done:
            if show_detail:
                print("p1's expiry : continue")
            continue
            # else:
        #   tp_j = op_idx1

        prev_open_i2 = open_i2
        net_p1_idx_list.append(op_idx1)
        # if check_hlm in [0, 1]:
        #   i = op_idx1  # allow op_idx2 = op_idx1
        allow_exit = 1

        while 1:
            # ------------ p2 phase ------------ #

            # ------ 1. get p2_info ------ #
            if check_hlm in [1, 2]:
                open_i2 += 1  # 확인 끝났으면 조기 이탈(+1), 다음 open_idx 조사 진행
                if open_i2 >= len_open_idx2:  # open_i2 소진
                    if show_detail:
                        print("open_i2 >= len_open_idx2, open_i2 소진 : break")
                    break

                # ------ check side sync. ------ #
                if side_arr1[open_i1] != side_arr2[open_i2]:
                    if show_detail:
                        print("side check rejection, open_i2 {}, side_arr2 {}".format(open_i2, side_arr2[open_i2]))
                    continue

                # ------ assert, op_idx2 >= exec_j ------ #
                op_idx2 = open_idx2[open_i2]  # open_i2 는 i 와 별개로 운영
                if check_hlm == 1 and allow_exit:
                    if op_idx2 < op_idx1:
                        if show_detail:
                            print("check_hlm 1's allow_exit rejection, op_idx2 {} < op_idx1 {}".format(op_idx2, op_idx1))
                        continue
                else:
                    if op_idx2 < i:  # p1 execution 이후의 i 를 허용 (old, 이곳 i = op_idx1 + 1 or p2's exec_j or exit_loop's i + 1)
                        if show_detail:
                            print("op_idx2 {} < i {} : continue".format(op_idx2, i))
                        continue

                if check_hlm == 2:
                    i = op_idx2 + 1  # open_signal 이 close_bar.shift(1) 이라고 가정하고 다음 bar 부터 체결확인한다는 의미
                    if i >= len_df:  # res_df 의 last_index 까지 돌아야함
                        break

                if show_detail:
                    print("op_idx1 : {} op_idx2 : {}".format(op_idx1, op_idx2))

            else:
                op_idx2 = op_idx1

            # ------ 2. load util paper data for p2  ------ #
            ep_j, tp_j, out_j = op_idx1, op_idx1, op_idx1
            p2_tr_set_idx = (ep_j, tp_j, out_j)

            ep2_ = ep2_arr[ep_j]
            tp_ = tp_arr[tp_j]
            out_ = out_arr[out_j]

            out_1_ = res_df['{}_out_1_{}'.format(side_pos, selection_id)].to_numpy()[out_j]
            out_0_ = res_df['{}_out_0_{}'.format(side_pos, selection_id)].to_numpy()[out_j]
            out_gap_ = res_df['{}_out_gap_{}'.format(side_pos, selection_id)].to_numpy()[out_j]

            # ------ const. for p2_wave ------ #
            wave_itv1 = config.tr_set.wave_itv1
            wave_period1 = config.tr_set.wave_period1
            wave_itv2 = config.tr_set.wave_itv2
            wave_period2 = config.tr_set.wave_period2

            if check_hlm in [1, 2]:

                # ------ check p1's expiry ------ # - p2_box 생성 이전의 hl_survey
                # 1. op_idx1 ~ op_idx2 까지의 expiry check (high & low 둘다)
                #     a. if check_hlm:  # p1_hlm, p2_hlm --> Todo, 이거를 왜 p1_hlm 에도 적용했는지 잘 모르겠음
                if op_idx1 < op_idx2:
                    expire, touch_idx = expiry_p1p2(res_df, config, op_idx1, op_idx2, tp_1_, tp_0_, tp_gap_, ohlc_list[1:3], open_side)
                    if expire:  # p1's expiry
                        if show_detail:
                            print("expiry_p1p2, touch_idx = {} : break".format(touch_idx))
                        i = touch_idx  # + 1  --> 이거 아닌것 같음 # op_idx1 과 op_idx2 사이의 op_idx1' 을 살리기 위함, 즉 바로 다음 op_idx1 로 회귀 (건너뛰지 않고)
                        open_i2 = prev_open_i2
                        break  # change op_idx1

                if check_hlm == 2:

                    """
                    p2 point_validation - vectorization unavailable
                        1. p2 로 wave_unit 을 사용할 경우만, p2 wave_validation & wave_box location 사용할 것.
                        2. p1_loop 로 return 되는 정확한 i 를 반환하기 위해서 expiry_p1p2 뒤에 배치함
                        3. Todo - 새로운 tp, ep, out 에 대한 처리 필요 (p1_hlm 사용시)                        
                    """

                    # ------ p2_wave validation : 정확한 뜻을 아직 잘 모르겠음. ------ #
                    #                     if open_side == OrderSide.SELL:
                    #                         wave_co_post_idx = res_df['wave_co_post_idx_fill_{}{}'.format(wave_itv2, wave_period2)].to_numpy()[op_idx2]
                    #                         if not (op_idx1 < wave_co_post_idx):
                    #                             if show_detail:
                    #                                 print("p2_wave validation : continue")
                    #                             continue  # change op_idx2

                    #                         # --- p2_wave high validation --- #
                    #                         # wave_high_fill1_ = res_df['wave_high_fill_{}{}'.format(wave_itv1, wave_period1)].to_numpy()[op_idx1]
                    #                         # wave_high_fill2_ = res_df['wave_high_fill_{}{}'.format(wave_itv2, wave_period2)].to_numpy()[op_idx2]
                    #                         # if not (wave_high_fill1_ >= wave_high_fill2_):
                    #                         #   if show_detail:
                    #                         #     print("p2_wave high validation : continue")
                    #                         #   continue  # change op_idx2

                    #                     else:
                    #                         wave_cu_post_idx = res_df['wave_cu_post_idx_fill_{}{}'.format(wave_itv2, wave_period2)].to_numpy()[op_idx2]
                    #                         if not (op_idx1 < wave_cu_post_idx):
                    #                             if show_detail:
                    #                                 print("p2_wave validation : continue")
                    #                             continue  # change op_idx2

                    #                         # --- p2_wave low validation --- #
                    #                         # wave_low_fill1_ = res_df['wave_low_fill_{}{}'.format(wave_itv1, wave_period1)].to_numpy()[op_idx1]
                    #                         # wave_low_fill2_ = res_df['wave_low_fill_{}{}'.format(wave_itv2, wave_period2)].to_numpy()[op_idx2]
                    #                         # if not (wave_low_fill1_ <= wave_low_fill2_):
                    #                         #   if show_detail:
                    #                         #     print("p2_wave low validation : continue")
                    #                         #   continue  # change op_idx2

                    #                     # ------ p2 wave_box location ------ #
                    #                     if open_side == OrderSide.SELL:
                    #                         if not ((tp_1_ + tp_gap_ * config.tr_set.p2_box_k1 <= out_1_) and (
                    #                                 out_0_ <= tp_0_ - tp_gap_ * config.tr_set.p2_box_k2)):  # tp1, tp0 에 닿으면 expiry
                    #                             # if not ((tp_1_ + tp_gap_ * config.tr_set.p2_box_k1 >= out_1_) and (out_0_ <= tp_0_ - tp_gap_ * config.tr_set.p2_box_k2)):  # tp1, tp0 에 닿으면 expiry
                    #                             if show_detail:
                    #                                 print("p2_box rejection : continue")
                    #                             continue
                    #                     else:
                    #                         if not ((tp_1_ - tp_gap_ * config.tr_set.p2_box_k1 >= out_1_) and (out_0_ >= tp_0_ + tp_gap_ * config.tr_set.p2_box_k2)):
                    #                             # if not ((tp_1_ - tp_gap_ * config.tr_set.p2_box_k1 <= out_1_) and (out_0_ >= tp_0_ + tp_gap_ * config.tr_set.p2_box_k2)):
                    #                             if show_detail:
                    #                                 print("p2_box rejection : continue")
                    #                             continue

                    # ------ tr_set validation & reject hl_out open_exec. ------ #
                    if open_side == OrderSide.SELL:
                        if not (tp_ < ep2_):
                            break  # change op_idx1
                        elif not (ep2_ < out_ and close[op_idx2] < out_):
                            if show_detail:
                                print("p2 tr_set validation : continue")
                            continue  # change op_idx2
                    else:
                        if not (tp_ > ep2_):
                            break
                        elif not (ep2_ > out_ and close[op_idx2] > out_):
                            if show_detail:
                                print("p2 tr_set validation : continue")
                            continue

                    # ------ p1p2_low ------ #
                    if open_side == OrderSide.SELL:
                        if not high[op_idx1:op_idx2 + 1].max() < tp_0_ - tp_gap_ * config.tr_set.p1p2_low:
                            if show_detail:
                                print("p1p2_low rejection : continue")
                            continue
                    else:
                        if not low[op_idx1:op_idx2 + 1].min() > tp_0_ + tp_gap_ * config.tr_set.p1p2_low:
                            if show_detail:
                                print("p1p2_low rejection : continue")
                            continue

                    # ------ check p2's expiry ------ # - 현재, op_idx2 기준의 ep2_arr 을 사용 중임.
                    """
                    Caution : tr_set_idx 상황에 따라 잘 확인할 것
                    """
                    exec_j, entry_done, en_p, fee = check_entry(res_df, config, config.ep_set.point2.entry_type,
                                                                op_idx2, p2_tr_set_idx, len_df, open_side,
                                                                [*ohlc_list, ep2_arr], expiry)
                    i = exec_j  # = entry_loop 를 돌고 나온 e_j
                    if not entry_done:  # p2's expiry
                        if show_detail:
                            print("expiry, i = {} at p2's : continue".format(i))
                        continue  # change op_idx2

                    # ------ devectorized tr_calc ------ #
                    #    1. en_p 에 대해 하는게 맞을 것으로봄
                    #    2. tr_thresh 와 무관하게 있어야할 phase.
                    #    Todo, fee 계산에 오류가 있는 걸로 보임 => limit_fee 를 앞에 더해주어야할 것.
                    if open_side == OrderSide.SELL:
                        tr_ = abs((en_p / tp_ - config.trader_set.limit_fee - 1) / (en_p / out_ - config.trader_set.market_fee - 1))
                    else:
                        tr_ = abs((tp_ / en_p - config.trader_set.limit_fee - 1) / (out_ / en_p - config.trader_set.market_fee - 1))

                    # ------ tr_threshold ------ #
                    if config.loc_set.point2.tr_thresh_short != "None":
                        if open_side == OrderSide.SELL:
                            if tr_ < config.loc_set.point2.tr_thresh_short:
                                if show_detail:
                                    print("tr_threshold : continue")
                                continue
                        else:
                            if tr_ < config.loc_set.point2.tr_thresh_long:
                                if show_detail:
                                    print("tr_threshold : continue")
                                continue

            # 1. allow_exit = "p1_hlm 의 경우, 한번 out 되면 price 가 "wave_range 에 닿기전까지" retrade 를 허용하지 않음" (expiry_p1p2 을 이용해 op_idx1 을 변경할 것)
            #     a. while phase 내부에 if not allow_exit 을 위치한 이유 : "wave_range 에 닿기전까지" 를 구현하기 위해서.
            if not allow_exit:
                if show_detail:
                    print("allow_exit = {} : continue".format(allow_exit))
                continue

            if check_hlm in [0, 1]:
                tr_ = tr_arr[op_idx1]

            # ------ leverage ------ #
            # out = out_arr[out_j]  # lvrg_set use out on out_j (out_j shoud be based on p2)
            leverage, liqd_p = lvrg_set(res_df, config, open_side, en_p, out_, fee)  # res_df 변수 사용됨 - 주석 처리 된 상태일뿐

            if leverage is None:
                if show_detail:
                    print("leverage is None : continue")
                if check_hlm:
                    continue  # change op_idx2
                else:
                    break  # change op_idx1

            # ------------ exit phase ------------ #
            exit_done, cross_on = 0, 0

            # ------ check tpout_onexec ------ #
            if config.ep_set.entry_type == "LIMIT":
                if config.tp_set.tp_onexec:
                    tp_j = exec_j
                if config.out_set.out_onexec:
                    out_j = exec_j

            while 1:
                # dynamic tp / out 을 사용하고 싶은 경우
                if not config.tp_set.static_tp:
                    tp_j = i
                if not config.out_set.static_out:
                    out_j = i

                # ------------ 1. out ------------ #  # out 우선 (보수적 검증)
                # ------ a. signal_out ------ #
                if not exit_done:
                    exit_done, cross_on, ex_p, fee = check_signal_out(res_df, config, open_i2, i, len_df, fee, open_side, cross_on, exit_done, [*ohlc_list, np_timeidx])
                    # ------ b. hl_out ------ #
                # if config.out_set.hl_out: # --> liqd_p 도입으로 hl_out 내부에서 liqd_p 조건부 수행은 필수불가결이다.
                if not exit_done:  # and i != len_df - 1:
                    exit_done, ex_p, fee = check_hl_out(config, i, out_j, len_df, fee, open_side, exit_done, [*ohlc_list, out_arr, liqd_p])

                # ------------ 2. tp ------------ #
                if not config.tp_set.non_tp and i != exec_j:
                    if not exit_done:
                        # 1. partial_tps 를 고려해 [tp_arr, ...] 형태 사용함.
                        # 2. if config.tp_set.tp_type in ['LIMIT']:  # 'BOTH' -> 앞으로는, LIMIT 밖에 없을거라 주석처리함
                        # 3. Todo, open_i2 는 deacy 기능을 위해 도입한 것 (추후 사용시 재확인)
                        exit_done, ex_p, fee = check_limit_tp_exec(res_df, config, open_i2, i, tp_j, len_df, fee, open_side, exit_done, [*ohlc_list, [tp_arr]])

                if exit_done:  # 이 phase 는 exit_phase 뒤에도 있어야할 것 - entry_done var. 사용은 안하겠지만
                    # ------ 3. append dynamic result vars. ------ #
                    p1_idx_list.append(op_idx1)  # side, zone, start_ver arr 모두 openi_list 로 접근하기 위해 open_i 를 담음
                    p2_idx_list.append(op_idx2)
                    pair_idx_list.append([exec_j, i])  # entry & exit (체결 기준임)
                    pair_price_list.append([en_p, ex_p])
                    lvrg_list.append(leverage)
                    fee_list.append(fee)
                    tr_list.append(tr_)  # Todo, tr vectorize 불가함, 직접 구해주어야할 건데.. (오래걸리지 않을까 --> tr_set 데이터만 모아서 vecto 계산이 나을 것)

                    # for tpout_line plot_check & get_pr calc.
                    if exit_done == 2:
                        tpout_list.append([tp_arr[tp_j], liqd_p])
                    else:
                        tpout_list.append([tp_arr[tp_j], out_arr[out_j]])

                    # open_i += 1  # 다음 open_idx 조사 진행
                    break

                # 1. 아래있으면, 체결 기준부터 tp, out 허용 -> tp 가 entry_idx 에 체결되는게 다소 염려되기는 함, 일단 진행 (그런 case 가 많지 않았으므로)
                # 2. 위에있으면, entry 다음 tick 부터 exit 허용
                i += 1
                if i >= len_df:  # res_df 의 last_index 까지 돌아야함
                    break

            if i >= len_df:  # res_df 의 last_index 까지 돌아야함
                break

            """
            exit_done description            
                1. 1 : tp_done
                    a. check_hlm 여부와 무관하게 outer loop 의 op_idx1 을 변경 가능하도록함.
                2. -1 : out done
                2. 2 : liquidation done
                3. 0 : database done                
            """

            if exit_done == 1:
                if show_detail:
                    print("exit_done = {}, i = {} : break".format(exit_done, i))
                break  # change op_idx1
            else:
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

        if i >= len_df:  # or open_i >= len_open_idx:  # res_df 의 last_index 까지 돌아야함
            break
        else:
            continue

    return np.array(net_p1_idx_list), np.array(p1_idx_list), np.array(p2_idx_list), np.array(pair_idx_list), np.array(pair_price_list), np.array(lvrg_list), np.array(fee_list), np.array(
        tpout_list), np.array(tr_list)


def en_ex_pairing_v9_43(res_df, open_idx_list, open_info_list, ohlc_list, config_list, np_timeidx, funcs, show_detail=False):  # 이미 충분히 줄여놓은 idx 임

    """
    v9_42 -> v9_43
        1. p2 를 도입하면서, p1 기준의 tr_set 을 사용하는 경우 (ep2 & out)
        2. Skip side check temporarily
    """

    open_info1, open_info2 = open_info_list
    side_arr1, _, _, id_idx_arr1 = open_info1
    side_arr2, _, _, _ = open_info2

    expiry_p1p2, expiry, lvrg_set = funcs

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

        # ------ get p1_info ------ #
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

        """
        1. exec_j : open 체결 index
        2. ep_j : entry_price 기준 index
        3. tp_j : tp_price 기준 index
        4. out_j : out_price 기준 index
        """
        # if not check_net_hhm:  # this phase exist for p1 entry (net hhm sync.) in p2_platform
        exec_j, ep_j, tp_j, out_j, entry_done, en_p, fee = check_entry_v6_1(res_df, config,
                                                                            config.ep_set.entry_type, op_idx1, len_df,
                                                                            open_side, [*ohlc_list, ep1_arr], expiry)
        i = exec_j  # = entry_loop 를 돌고 나온 e_j
        if not entry_done:
            if show_detail:
                print("p1's expiry : continue")
            continue
            # else:
        #   tp_j = op_idx1

        prev_open_i2 = open_i2
        net_p1_idx_list.append(op_idx1)
        # if check_hlm in [0, 1]:
        #   i = op_idx1  # allow op_idx2 = op_idx1
        allow_exit = 1

        # ------ entry loop ------ #
        while 1:  # for p2's loop (allow retry)

            # ------ get p2_info ------ #
            if check_hlm in [1, 2]:
                open_i2 += 1  # 확인 끝났으면 조기 이탈(+1), 다음 open_idx 조사 진행
                if open_i2 >= len_open_idx2:  # open_i2 소진
                    if show_detail:
                        print("open_i2 >= len_open_idx2, open_i2 소진 : break")
                    break

                # ------ check side sync. ------ #
                # if open_side != side_arr2[open_i2]:
                #     if show_detail:
                #         print("side check rejection, open_i2 {}, open_side {}".format(open_i2, side_arr2[open_i2]))
                #     continue

                # ------ assert, op_idx2 >= exec_j ------ #
                op_idx2 = open_idx2[open_i2]  # open_i2 는 i 와 별개로 운영
                if check_hlm == 1 and allow_exit:
                    if op_idx2 < op_idx1:
                        if show_detail:
                            print("check_hlm 1's allow rejection, op_idx2 {} < op_idx1 {}".format(op_idx2, op_idx1))
                        continue
                else:
                    if op_idx2 < i:  # p1 execution 이후의 i 를 허용 (old, 이곳 i = op_idx1 + 1 or p2's exec_j or exit_loop's i + 1)
                        if show_detail:
                            print("op_idx2 {} < i {} : continue".format(op_idx2, i))
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
            ep2_ = ep2_arr[op_idx1]
            out_ = out_arr[op_idx1]

            out_1_ = res_df['{}_out_1_{}'.format(side_pos, selection_id)].to_numpy()[op_idx1]
            out_0_ = res_df['{}_out_0_{}'.format(side_pos, selection_id)].to_numpy()[op_idx1]
            out_gap_ = res_df['{}_out_gap_{}'.format(side_pos, selection_id)].to_numpy()[op_idx1]

            out_ = out_0_

            # ------ const. for p2_wave ------ #
            wave_itv1 = config.tr_set.wave_itv1
            wave_period1 = config.tr_set.wave_period1
            wave_itv2 = config.tr_set.wave_itv2
            wave_period2 = config.tr_set.wave_period2

            if check_hlm in [1, 2]:

                # ------ check p1's expiry ------ # - p2_box 생성 이전의 hl_survey
                # 1. op_idx1 ~ op_idx2 까지의 expiry check (high & low 둘다)
                #     a. if check_hlm:  # p1_hlm, p2_hlm --> Todo, 이거를 왜 p1_hlm 에도 적용했는지 잘 모르겠음
                if op_idx1 < op_idx2:
                    expire, touch_idx = expiry_p1p2(res_df, config, op_idx1, op_idx2, tp_1_, tp_0_, tp_gap_, ohlc_list[1:3], open_side)
                    if expire:  # p1's expiry
                        if show_detail:
                            print("expiry_p1p2, touch_idx = {} : break".format(touch_idx))
                        i = touch_idx  # + 1  --> 이거 아닌것 같음 # op_idx1 과 op_idx2 사이의 op_idx1' 을 살리기 위함, 즉 바로 다음 op_idx1 로 회귀 (건너뛰지 않고)
                        open_i2 = prev_open_i2
                        break  # change op_idx1

                if check_hlm == 2:

                    """
                    p2 point_validation - vectorization unavailable
                        1. p2 로 wave_unit 을 사용할 경우만, p2 wave_validation & wave_box location 사용할 것.
                        2. p1_loop 로 return 되는 정확한 i 를 반환하기 위해서 expiry_p1p2 뒤에 배치함
                        3. Todo - 새로운 tp, ep, out 에 대한 처리 필요 (p1_hlm 사용시)                        
                    """

                    # ------ p2_wave validation : 정확한 뜻을 아직 잘 모르겠음. ------ #
                    #                     if open_side == OrderSide.SELL:
                    #                         wave_co_post_idx = res_df['wave_co_post_idx_fill_{}{}'.format(wave_itv2, wave_period2)].to_numpy()[op_idx2]
                    #                         if not (op_idx1 < wave_co_post_idx):
                    #                             if show_detail:
                    #                                 print("p2_wave validation : continue")
                    #                             continue  # change op_idx2

                    #                         # --- p2_wave high validation --- #
                    #                         # wave_high_fill1_ = res_df['wave_high_fill_{}{}'.format(wave_itv1, wave_period1)].to_numpy()[op_idx1]
                    #                         # wave_high_fill2_ = res_df['wave_high_fill_{}{}'.format(wave_itv2, wave_period2)].to_numpy()[op_idx2]
                    #                         # if not (wave_high_fill1_ >= wave_high_fill2_):
                    #                         #   if show_detail:
                    #                         #     print("p2_wave high validation : continue")
                    #                         #   continue  # change op_idx2

                    #                     else:
                    #                         wave_cu_post_idx = res_df['wave_cu_post_idx_fill_{}{}'.format(wave_itv2, wave_period2)].to_numpy()[op_idx2]
                    #                         if not (op_idx1 < wave_cu_post_idx):
                    #                             if show_detail:
                    #                                 print("p2_wave validation : continue")
                    #                             continue  # change op_idx2

                    #                         # --- p2_wave low validation --- #
                    #                         # wave_low_fill1_ = res_df['wave_low_fill_{}{}'.format(wave_itv1, wave_period1)].to_numpy()[op_idx1]
                    #                         # wave_low_fill2_ = res_df['wave_low_fill_{}{}'.format(wave_itv2, wave_period2)].to_numpy()[op_idx2]
                    #                         # if not (wave_low_fill1_ <= wave_low_fill2_):
                    #                         #   if show_detail:
                    #                         #     print("p2_wave low validation : continue")
                    #                         #   continue  # change op_idx2

                    #                     # ------ p2 wave_box location ------ #
                    #                     if open_side == OrderSide.SELL:
                    #                         if not ((tp_1_ + tp_gap_ * config.tr_set.p2_box_k1 <= out_1_) and (
                    #                                 out_0_ <= tp_0_ - tp_gap_ * config.tr_set.p2_box_k2)):  # tp1, tp0 에 닿으면 expiry
                    #                             # if not ((tp_1_ + tp_gap_ * config.tr_set.p2_box_k1 >= out_1_) and (out_0_ <= tp_0_ - tp_gap_ * config.tr_set.p2_box_k2)):  # tp1, tp0 에 닿으면 expiry
                    #                             if show_detail:
                    #                                 print("p2_box rejection : continue")
                    #                             continue
                    #                     else:
                    #                         if not ((tp_1_ - tp_gap_ * config.tr_set.p2_box_k1 >= out_1_) and (out_0_ >= tp_0_ + tp_gap_ * config.tr_set.p2_box_k2)):
                    #                             # if not ((tp_1_ - tp_gap_ * config.tr_set.p2_box_k1 <= out_1_) and (out_0_ >= tp_0_ + tp_gap_ * config.tr_set.p2_box_k2)):
                    #                             if show_detail:
                    #                                 print("p2_box rejection : continue")
                    #                             continue

                    # ------ tr_set validation & reject hl_out open_exec. ------ #
                    if open_side == OrderSide.SELL:
                        if not (tp_ < ep2_):
                            break  # change op_idx1
                        elif not (ep2_ < out_ and close[op_idx2] < out_):
                            if show_detail:
                                print("p2 tr_set validation : continue")
                            continue  # change op_idx2
                    else:
                        if not (tp_ > ep2_):
                            break
                        elif not (ep2_ > out_ and close[op_idx2] > out_):
                            if show_detail:
                                print("p2 tr_set validation : continue")
                            continue

                    # ------ p1p2_low ------ #
                    if open_side == OrderSide.SELL:
                        if not high[op_idx1:op_idx2 + 1].max() < tp_0_ - tp_gap_ * config.tr_set.p1p2_low:
                            if show_detail:
                                print("p1p2_low rejection : continue")
                            continue
                    else:
                        if not low[op_idx1:op_idx2 + 1].min() > tp_0_ + tp_gap_ * config.tr_set.p1p2_low:
                            if show_detail:
                                print("p1p2_low rejection : continue")
                            continue

                    # ------ check p2's expiry ------ # - 현재, op_idx2 기준의 ep2_arr 을 사용 중임.
                    """
                    CAUTION : check_entry() 함수에서 op_idx 기준으로 ep, tp, out_j 가 init 되기 때문에, tr_set 의 기준에 따라 재선언 여부를 결정해야함.
                        1. v9_43 에서는 p1 기준 tr_set 을 사용하기 때문에, 재선언을 생략함.
                    """
                    exec_j, _, _, _, entry_done, en_p, fee = check_entry_v6_1(res_df, config, config.ep_set.point2.entry_type,
                                                                              op_idx2, len_df, open_side,
                                                                              [*ohlc_list, ep2_arr], expiry)
                    i = exec_j  # = entry_loop 를 돌고 나온 e_j
                    if not entry_done:  # p2's expiry
                        if show_detail:
                            print("expiry, i = {} at p2's : continue".format(i))
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

            if not allow_exit:  # p1_hlm 의 경우, 한번 out 되면 price 가 wave_range 에 닿기전까지 retrade 를 허용하지 않는다. (expiry_p1p2 을 이용해 op_idx1 을 변경할 것)
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

            # 1 : tp_done, check_hlm 여부와 무관하게 op_idx1 을 변경함
            if exit_done == 1:
                if show_detail:
                    print("exit_done = {}, i = {} : break".format(exit_done, i))
                break  # change op_idx1
            # -1 : out done / 0 : database done
            else:
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


def en_ex_pairing_v9_42(res_df, open_idx_list, open_info_list, ohlc_list, config_list, np_timeidx, funcs, show_detail=False):  # 이미 충분히 줄여놓은 idx 임

    """
    v9_4 -> v9_42
        1. check_entry_v6_1 도입 : expiry_tp 에 최적화된 function input 사용 
    """

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
        exec_j, ep_j, tp_j, out_j, entry_done, en_p, fee = check_entry_v6_1(res_df, config,
                                                                            config.ep_set.entry_type, op_idx1, len_df,
                                                                            open_side, [*ohlc_list, ep1_arr], expiry_p2)
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
                    exec_j, ep_j, _, out_j, entry_done, en_p, fee = check_entry_v6_1(res_df, config, config.ep_set.point2.entry_type,
                                                                                     op_idx2, len_df, open_side,
                                                                                     [*ohlc_list, ep2_arr], expiry_p2)
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


def en_ex_pairing_v9_41(res_df, open_idx_list, open_info_list, ohlc_list, config_list, np_timeidx, funcs, show_detail=False):  # 이미 충분히 줄여놓은 idx 임

    """
     lvrg_set_v2 adj. on v9_4
    """

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
            leverage = lvrg_set_v2(config.trader_set.initial_asset, config, open_side, tp_, out_, fee, config.lvrg_set.limit_leverage)  # res_df 변수 사용됨 - 주석 처리 된 상태일뿐
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


def en_ex_pairing_v9_6(res_df, open_idx_list, open_info_list, ohlc_list, config_list, np_timeidx, funcs, show_detail=False):  # 이미 충분히 줄여놓은 idx 임

    """
    tp target 이 p2_box 기준인 경우에 대한 en_ex function
    """

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
                if open_i2 >= len_open_idx2 - 1:  # open_i2 소진 / get_wave_bias 의 last_idx 가 len_df - 1 이기 때문에 arr zero_size 를 방지하기 위해 -1.
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

            # tp_1_ = res_df['{}_tp_1_{}'.format(side_pos, selection_id)].to_numpy()[op_idx2]  # p2's tp_box 를 위한 재정의
            # tp_0_ = res_df['{}_tp_0_{}'.format(side_pos, selection_id)].to_numpy()[op_idx2]
            # tp_gap_ = res_df['{}_tp_gap_{}'.format(side_pos, selection_id)].to_numpy()[op_idx2]

            tp_ = tp_arr[op_idx2]
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
                        # wave_co_post_idx = res_df['wave_co_post_idx_fill_{}{}'.format(wave_itv2, wave_period2)].to_numpy()[op_idx2]
                        # if not (op_idx1 < wave_co_post_idx):
                        #   if show_detail:
                        #     print("p2_wave validation : continue")
                        #   continue  # change op_idx2

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
                        # wave_cu_post_idx = res_df['wave_cu_post_idx_fill_{}{}'.format(wave_itv2, wave_period2)].to_numpy()[op_idx2]
                        # if not (op_idx1 < wave_cu_post_idx):
                        #   if show_detail:
                        #     print("p2_wave validation : continue")
                        #   continue  # change op_idx2

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
            else:
                if check_hlm == 2:
                    tp_j = exec_j
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
                    exit_done, cross_on, ex_p, fee = check_signal_out_v3_1(res_df, config, open_i2, i, len_df, fee, open_side, cross_on, exit_done,
                                                                           [*ohlc_list, np_timeidx])
                # ------ hl_out ------ #
                if config.out_set.hl_out:
                    if not exit_done:  # and i != len_df - 1:
                        exit_done, ex_p, fee = check_hl_out_v2(config, i, out_j, len_df, fee, open_side, exit_done, [*ohlc_list, out_arr])

                # ------------ tp ------------ #
                if not config.tp_set.non_tp and i != exec_j:
                    if not exit_done:
                        if show_detail:
                            print("i, exec_j, op_idx1, op_idx2 :", i, exec_j, op_idx1, op_idx2)
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


def en_ex_pairing_v9_4(res_df, open_idx_list, open_info_list, ohlc_list, config_list, np_timeidx, funcs, show_detail=False):  # 이미 충분히 줄여놓은 idx 임

    """
    1. p2_wave high validation 이 추가됨 <-> v9_3, out_1's expiry 사용중
    2. wave_cci 를 위한 latest stable version
    """

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


def check_entry_v7(res_df, config, entry_type, op_idx1, op_idx2, wave1, wave_gap, len_df, open_side, np_datas, expiry):

    """
    1. expiry() 에 wave1, wave_gap 을 없애면서, tp 를 기준한 expiration 사용 <-> 기존은 wave_1 기준이였음.

    v6_1 -> v7
        1. e_j loop 시작 index 가 op_idx2 기준임, v6_1 은 op_idx1 기준.
    """

    open, high, low, close, ep_arr = np_datas
    ep_j = op_idx1
    tp_j = op_idx1
    out_j = op_idx1

    # print("ep_arr[op_idx] :", ep_arr[op_idx])

    selection_id = config.selection_id
    # allow_ep_in = 0 if config.ep_set.point2.use_point2 else 1
    entry_done = 0
    ep = None

    if entry_type == "LIMIT":
        fee = config.trader_set.limit_fee

        for e_j in range(op_idx2 + 1, len_df):
            # ------ index setting for dynamic options ------ #
            if not config.ep_set.static_ep:
                ep_j = e_j  # dynamic_ep 를 위한 ep_index var.
                out_j = e_j  # dynamic_out 를 위한 out_index var. - 조건식이 static_ep 와 같이 있는 이유 모름 => dynamic_lvrg 로 사료됨

            if not config.tp_set.static_tp:
                tp_j = e_j

            # ------ expire_k & expire_tick ------ # - limit 사용하면 default 로 expire_k 가 존재해야함
            # if expiry(res_df, config, op_idx, e_j, wave1, wave_gap, [high, low], open_side):  # tp_j,
            if expiry(res_df, config, op_idx2, e_j, tp_j, [high, low], open_side):  # tp 기준 expiry 사용
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
                    if open[e_j] > ep_arr[ep_j]:  # open comp 는 결국, 수익률에 얹어주는 logic (반보수) -> 사용 보류
                        ep = open[e_j]
                    break
            else:
                if low[e_j] <= ep_arr[ep_j]:
                    entry_done = 1
                    ep = ep_arr[ep_j]
                    if open[e_j] < ep_arr[ep_j]:
                        ep = open[e_j]
                    break

    else:  # market entry
        e_j = op_idx2 + 1
        entry_done = 1
        ep = close[op_idx2]
        fee = config.trader_set.market_fee

    return e_j, ep_j, tp_j, out_j, entry_done, ep, fee  # 다음 start_i <-- e_j 로 변경
    #   e_j => 다음 phase 의 시작 index <-> ep_j : ep 의 기준 index
    #   ep_j, tp_j, out_j 가 return 되어야함 - exit phase 에서 이어가기 위함


def check_entry_v6_3(res_df, config, entry_type, op_idx, tr_set_idx, len_df, open_side, np_datas, expiry):
    """
    v6_2 -> v6_3
        1. 보수적 검증 적용 (체결률 100%)
    """

    open, high, low, close, ep_arr = np_datas
    ep_base_idx, tp_base_idx, out_base_idx = tr_set_idx

    # selection_id = config.selection_id
    # allow_ep_in = 0 if config.ep_set.point2.use_point2 else 1
    entry_done = 0
    ep = None

    if entry_type == "LIMIT":
        fee = config.trader_set.limit_fee

        for e_j in range(op_idx + 1, len_df):
            # ------ index setting for dynamic options ------ #
            if not config.ep_set.static_ep:
                ep_base_idx = e_j  # dynamic_ep 를 위한 ep_index var.
                out_base_idx = e_j  # dynamic_out 를 위한 out_index var. - 조건식이 static_ep 와 같이 있는 이유 모름 => dynamic_lvrg 로 사료됨

            if not config.tp_set.static_tp:
                tp_base_idx = e_j

            # ------ expire_k & expire_tick ------ # - limit 사용하면 default 로 expire_k 가 존재해야함
            if expiry(res_df, config, op_idx, e_j, tp_base_idx, [high, low], open_side):
                break

            # ------ point2 ------ #
            # if not allow_ep_in:
            #     allow_ep_in, out_base_idx = ep_loc_point2(res_df, config, e_j, out_base_idx, side=OrderSide.SELL)
            #     if allow_ep_in:
            #       if config.ep_set.point2.entry_type == "LIMIT":
            #         ep_base_idx = e_j
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
                # if high[e_j] >= ep_arr[ep_base_idx]:
                if high[e_j] > ep_arr[ep_base_idx]:  # 보수적 검증 (체결률 100%)
                    entry_done = 1
                    ep = ep_arr[ep_base_idx]
                    if open[e_j] >= ep_arr[ep_base_idx]:  # open comp 는 결국, 수익률에 얹어주는 logic (반보수) -> 사용 보류
                        ep = open[e_j]
                    break
            else:
                # if low[e_j] <= ep_arr[ep_base_idx]:
                if low[e_j] < ep_arr[ep_base_idx]:
                    entry_done = 1
                    ep = ep_arr[ep_base_idx]
                    if open[e_j] <= ep_arr[ep_base_idx]:
                        ep = open[e_j]
                    break

        try:
            exec_idx = e_j

        except Exception as e:
            exec_idx = None  # 어차피, 외부에서 entry_done = 0 로 빠지면 continue 되기 때문에 의미 없음.
            print("error in check_entry e_j loop : {}".format(e))

    else:  # market entry
        exec_idx = op_idx + 1
        entry_done = 1
        ep = close[op_idx]
        fee = config.trader_set.market_fee

    return exec_idx, entry_done, ep, fee


def check_entry_v6_2(res_df, config, entry_type, op_idx, tr_set_idx, len_df, open_side, np_datas, expiry):

    """
    v6_1 -> v6_2
        1. tr_set 을 p1, p2 에 가변적으로 기준하기 위해 ep_base_idx, tp_base_idx, out_base_idx 를 외부 참조하도록 함.
    """

    open, high, low, close, ep_arr = np_datas
    ep_base_idx, tp_base_idx, out_base_idx = tr_set_idx

    # selection_id = config.selection_id
    # allow_ep_in = 0 if config.ep_set.point2.use_point2 else 1
    entry_done = 0
    ep = None

    if entry_type == "LIMIT":
        fee = config.trader_set.limit_fee

        for e_j in range(op_idx + 1, len_df):
            # ------ index setting for dynamic options ------ #
            if not config.ep_set.static_ep:
                ep_base_idx = e_j  # dynamic_ep 를 위한 ep_index var.
                out_base_idx = e_j  # dynamic_out 를 위한 out_index var. - 조건식이 static_ep 와 같이 있는 이유 모름 => dynamic_lvrg 로 사료됨

            if not config.tp_set.static_tp:
                tp_base_idx = e_j

            # ------ expire_k & expire_tick ------ # - limit 사용하면 default 로 expire_k 가 존재해야함
            if expiry(res_df, config, op_idx, e_j, tp_base_idx, [high, low], open_side):
                break

            # ------ point2 ------ #
            # if not allow_ep_in:
            #     allow_ep_in, out_base_idx = ep_loc_point2(res_df, config, e_j, out_base_idx, side=OrderSide.SELL)
            #     if allow_ep_in:
            #       if config.ep_set.point2.entry_type == "LIMIT":
            #         ep_base_idx = e_j
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
                if high[e_j] >= ep_arr[ep_base_idx]:
                    entry_done = 1
                    ep = ep_arr[ep_base_idx]
                    if open[e_j] >= ep_arr[ep_base_idx]:  # open comp 는 결국, 수익률에 얹어주는 logic (반보수) -> 사용 보류
                        ep = open[e_j]
                    break
            else:
                if low[e_j] <= ep_arr[ep_base_idx]:
                    entry_done = 1
                    ep = ep_arr[ep_base_idx]
                    if open[e_j] <= ep_arr[ep_base_idx]:
                        ep = open[e_j]
                    break

        try:
            exec_idx = e_j

        except Exception as e:
            exec_idx = None  # 어차피, 외부에서 entry_done = 0 로 빠지면 continue 되기 때문에 의미 없음.
            print("error in check_entry e_j loop : {}".format(e))

    else:  # market entry
        exec_idx = op_idx + 1
        entry_done = 1
        ep = close[op_idx]
        fee = config.trader_set.market_fee

    return exec_idx, entry_done, ep, fee


def check_entry_v6_1(res_df, config, entry_type, op_idx, len_df, open_side, np_datas, expiry):

    """
    v6 -> v6_1
        1. expiry_tp 를 위한 function input 수정
    """

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
            if expiry(res_df, config, op_idx, e_j, tp_j, [high, low], open_side):
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


def check_limit_tp_exec_v3(res_df, config, open_i, i, tp_j, len_df, fee, open_side, exit_done, np_datas):
    """
    v1 -> v3
        1. 보수적 검증 적용 (체결률 100%)
    """

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

            if low[i] < tp_arr[tp_j]:  # 보수적 검증 (체결률 100%)
                # if low[i] <= tp_arr[tp_j]:  # and partial_tp_cnt == tp_i:  # we use static tp now
                # if low[i] <= tp_arr[i] <= h[i]: --> 이건 잘못되었음
                # partial_tp_cnt += 1 --> partial_tp 보류

                # 1. dynamic tp
                if tp_arr[i] != tp_arr[i - 1] and not config.tp_set.static_tp:
                    # tp limit 이 불가한 경우 - open 이 이미, tp 를 넘은 경우
                    if open[i] < tp_arr[i]:
                        tp = open[i]
                    # tp limit 이 가능한 경우 - open 이 아직, tp 를 넘지 않은 경우
                    else:
                        tp = tp_arr[i]

                # 2. static tp
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
            if high[i] > tp_arr[tp_j]:
                # if high[i] >= tp_arr[tp_j]:

                # 1. dynamic tp
                if tp_arr[i] != tp_arr[i - 1] and not config.tp_set.static_tp:
                    if open[i] > tp_arr[i]:
                        tp = open[i]
                    else:
                        tp = tp_arr[i]

                # 2. static tp
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


def check_limit_tp_exec_v2(res_df, config, open_i, i, tp_j, len_df, fee, open_side, exit_done, np_datas):

    """
    _ -> v2
        1. additional fee added.
    """

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

            # if low[i] < tp_arr[tp_j]:  # and partial_tp_cnt == tp_i:  # we use static tp now
            if low[i] <= tp_arr[tp_j]:  # and partial_tp_cnt == tp_i:  # we use static tp now
                # if low[i] <= tp_arr[i] <= h[i]: --> 이건 잘못되었음
                # partial_tp_cnt += 1 --> partial_tp 보류

                # 1. dynamic tp
                if tp_arr[i] != tp_arr[i - 1] and not config.tp_set.static_tp:
                    # tp limit 이 불가한 경우 - open 이 이미, tp 를 넘은 경우
                    if open[i] < tp_arr[i]:
                        tp = open[i]
                    # tp limit 이 가능한 경우 - open 이 아직, tp 를 넘지 않은 경우
                    else:
                        tp = tp_arr[i]

                # 2. static tp
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
            # if high[i] > tp_arr[tp_j]:
            if high[i] >= tp_arr[tp_j]:

                # 1. dynamic tp
                if tp_arr[i] != tp_arr[i - 1] and not config.tp_set.static_tp:
                    if open[i] > tp_arr[i]:
                        tp = open[i]
                    else:
                        tp = tp_arr[i]

                # 2. static tp
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
        fee += config.trader_set.limit_fee + 0.002  # just for Stock.

    return exit_done, tp, fee


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


def check_signal_out_v5(res_df, config, open_i, i, len_df, fee, open_side, cross_on, exit_done, np_datas):
    """
    v4 -> v5
        1. add fisher_exit.
    """
    _, _, _, close, np_timeidx = np_datas
    ex_p = None
    selection_id = config.selection_id

    # 1. timestamp
    if config.out_set.tf_exit != "None":
        if np_timeidx[i] % config.out_set.tf_exit == config.out_set.tf_exit - 1 and i != open_i:
            exit_done = -1

    # 2. fisher
    if config.out_set.fisher_exit:

        itv_num = itv_to_number(config.loc_set.point1.tf_entry)

        fisher_band = config.out_set.fisher_band
        fisher_band2 = config.out_set.fisher_band2

        if np_timeidx[i] % itv_num == itv_num - 1:

            fisher_ = res_df['fisher_{}30'.format(config.loc_set.point1.tf_entry)].to_numpy()

            if open_side == OrderSide.SELL:
                if (fisher_[i - itv_num] > -fisher_band) & (fisher_[i] <= -fisher_band):
                    exit_done = -1
                elif (fisher_[i - itv_num] < fisher_band2) & (fisher_[i] >= fisher_band2):
                    exit_done = -1
            else:
                if (fisher_[i - itv_num] < fisher_band) & (fisher_[i] >= fisher_band):
                    exit_done = -1
                elif (fisher_[i - itv_num] > fisher_band2) & (fisher_[i] <= fisher_band2):
                    exit_done = -1

    # 3. cci
    if config.out_set.cci_exit:
        cci_ = res_df['cci_T20'].to_numpy()

        if open_side == OrderSide.SELL:
            if (cci_[i - 1] >= -100) & (cci_[i] < -100):
                # if (cci_[i - 1] <= -100) & (cci_[i] > -100):
                exit_done = -1
        else:
            if (cci_[i - 1] <= 100) & (cci_[i] > 100):
                # if (cci_[i - 1] >= 100) & (cci_[i] < 100):
                exit_done = -1

    if exit_done:
        ex_p = close[i]
        fee += config.trader_set.market_fee

    return exit_done, cross_on, ex_p, fee


def check_signal_out_v4(res_df, config, open_i, i, len_df, fee, open_side, cross_on, exit_done, np_datas):
    """
    v3 -> v4
        1. remove unnecessary conditions.
    """
    _, _, _, close, np_timeidx = np_datas
    ex_p = None
    selection_id = config.selection_id

    # 1. timestamp
    if config.out_set.tf_exit != "None":
        if np_timeidx[i] % config.out_set.tf_exit == config.out_set.tf_exit - 1 and i != open_i:
            exit_done = -1

    # 3. cci
    if config.out_set.cci_exit:
        cci_ = res_df['cci_T20'].to_numpy()

        if open_side == OrderSide.SELL:
            if (cci_[i - 1] >= -100) & (cci_[i] < -100):
                # if (cci_[i - 1] <= -100) & (cci_[i] > -100):
                exit_done = -1
        else:
            if (cci_[i - 1] <= 100) & (cci_[i] > 100):
                # if (cci_[i - 1] >= 100) & (cci_[i] < 100):
                exit_done = -1

    if exit_done:
        ex_p = close[i]
        fee += config.trader_set.market_fee

    return exit_done, cross_on, ex_p, fee


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


def check_hl_out_v4(config, i, out_j, len_df, fee, open_side, exit_done, np_datas):
    """
    v3 -> v4
        1. Add non_out function.
    """

    open, high, low, close, out_arr, liqd_p = np_datas
    ex_p = None

    # 1. liquidation default check
    if open_side == OrderSide.SELL:
        if high[i] >= liqd_p:
            exit_done = 2
    else:
        if low[i] <= liqd_p:
            exit_done = 2

    if not config.out_set.non_out:
        # 2. hl_out
        if config.out_set.hl_out:
            if open_side == OrderSide.SELL:
                if high[i] >= out_arr[out_j]:
                    exit_done = -1
            else:
                if low[i] <= out_arr[out_j]:
                    exit_done = -1
        # 3. close_out
        else:
            if open_side == OrderSide.SELL:
                if close[i] >= out_arr[out_j]:
                    exit_done = -1
            else:
                if close[i] <= out_arr[out_j]:
                    exit_done = -1

    if exit_done:  # exit_done should not be zero in this phase
        if exit_done == 2:
            ex_p = liqd_p
        else:
            if config.out_set.hl_out:
                ex_p = out_arr[out_j]
            else:
                ex_p = close[i]

            # check open out execution
            if open_side == OrderSide.SELL:
                if open[i] >= out_arr[out_j]:
                    ex_p = open[i]
            else:
                if open[i] <= out_arr[out_j]:
                    ex_p = open[i]

        fee += config.trader_set.market_fee

    return exit_done, ex_p, fee


def check_hl_out_v3(config, i, out_j, len_df, fee, open_side, exit_done, np_datas):

    """
    v2 -> v3
        1. liquidation platform 도입함.
    """

    open, high, low, close, out_arr, liqd_p = np_datas
    ex_p = None

    # 1. liquidation default check
    if open_side == OrderSide.SELL:
        if high[i] >= liqd_p:
            exit_done = 2
    else:
        if low[i] <= liqd_p:
            exit_done = 2

    # 2-1. hl_out
    if config.out_set.hl_out:
        if open_side == OrderSide.SELL:
            if high[i] >= out_arr[out_j]:
                exit_done = -1
        else:
            if low[i] <= out_arr[out_j]:
                exit_done = -1
    # 2-2. close_out
    else:
        if open_side == OrderSide.SELL:
            if close[i] >= out_arr[out_j]:
                exit_done = -1
        else:
            if close[i] <= out_arr[out_j]:
                exit_done = -1

    if exit_done:  # exit_done should not be zero in this phase
        if exit_done == 2:
            ex_p = liqd_p
        else:
            if config.out_set.hl_out:
                ex_p = out_arr[out_j]
            else:
                ex_p = close[i]

            # check open out execution
            if open_side == OrderSide.SELL:
                if open[i] >= out_arr[out_j]:
                    ex_p = open[i]
            else:
                if open[i] <= out_arr[out_j]:
                    ex_p = open[i]

        fee += config.trader_set.market_fee

    return exit_done, ex_p, fee


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

#       Todo        #
#        1. back_id ajustment
#        2. title_position & tight_layout

from funcs.public.broker import sharpe_ratio, calc_tp_out_fee
from funcs.olds.funcs_indicator_candlescore import *
from funcs.public.idep import *
import matplotlib.pyplot as plt
import time
from matplotlib import gridspec

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def back_pr_check(res_df, png_save_path, utils_public, utils_list, config_list, key, show_plot=0, title_pos_y=0.5):

    multi_mode = 0
    strat_switch_idx = 1    # 0, 1, 2...
    override = 0
    public_override = 0

    if not multi_mode:  # override 하지않는 경우에 config1 만을 사용하니, config1 에 어떤 version 을 배치할지 선택
        utils_list[0] = utils_list[strat_switch_idx]
        config_list[0] = config_list[strat_switch_idx]

    config = config_list[0]  # custom base config, if use override -> set to config1
    
    tp_fee, out_fee = calc_tp_out_fee(config)   # Todo -> rev_pr 때문에 일단 이곳에도 선언함

    # ------- inversion set ------- #
    inversion = 0
    fdist_thresh = 1

    # ------- plot param ------- #
    fontsize = 10
    show_detail = 0
    title_position = (0.5, title_pos_y)

    # ------- temp param ------- #
    rsi_out_stratver = ['v7_3', '1_1']
    allow_osc_touch = 0
    rsi_gap = 5

    early_out_tpg = 0.36

    # ------- survey param ------- #
    itv_num_list = [1, 3, 5, 15]

    itv_list = ['15m', '30m', '1h', '4h']
    # itv_list = ['3m', '5m', '15m', '30m', '1h', '4h']

    # x_val_list = np.arange(, 2.0, 0.1)     # prcn 1
    x_val_list = np.arange(-0.69, -0.8, -0.01)  # prcn 2
    # x_val_list = np.arange(-0.695, -0.75, -0.005)    # prcn 3
    # x_val_list = np.arange(0.944, 0.945, 0.0001)    # prcn 4
    # x_val_list = np.arange(1, 10, 1)   # prcn -1
    # x_val_list = np.arange(210, 130, -5)   # prcn -2

    y_val_cols = ["wr", "sr", "frq", "dpf", "acc_pr", "sum_pr", "acc_mdd", "sum_mdd", "liqd", "min_pr", "tr", "dr"]
    y_rev_val_cols = ["wr", "sr", "acc_pr", "sum_pr", "acc_mdd", "sum_mdd", "min_pr"]

    #       Todo        #
    #        just paste below this line's end '\n' -------------------- #
    start_0 = time.time()

    np_timeidx = np.array(list(map(lambda x: intmin(x), res_df.index)))  # 이곳에 latency '3s'

    if override:
        res_df = public_indi(res_df, config, np_timeidx)
    else:
        if public_override:
            res_df = public_indi(res_df, config, np_timeidx)
        else:
            res_df = utils_public.public_indi(res_df, config, np_timeidx)

    # -------------------- entlist rtc & tr 은 중복되는 여부에 따라 user 가 flexible coding 해야할 것 -------------------- #
    if override:
        res_df = enlist_rtc(res_df, config, np_timeidx)
    else:
        for utils_, cfg_ in zip(utils_list, config_list):   # recursively
            res_df = utils_.enlist_rtc(res_df, cfg_, np_timeidx)
            if not multi_mode:
                break

    print("load_df ~ enlist_rtc elapsed time :", time.time() - start_0)

    survey_df = pd.DataFrame(index=x_val_list, columns=y_val_cols)
    short_survey_df = pd.DataFrame(index=x_val_list, columns=y_val_cols)
    long_survey_df = pd.DataFrame(index=x_val_list, columns=y_val_cols)
    rev_survey_df = pd.DataFrame(index=x_val_list, columns=y_rev_val_cols)
    rev_short_survey_df = pd.DataFrame(index=x_val_list, columns=y_rev_val_cols)
    rev_long_survey_df = pd.DataFrame(index=x_val_list, columns=y_rev_val_cols)

    for survey_i, just_loop in enumerate(range(1)):
        # for survey_i, config.loc_set.zone.tr_thresh in enumerate(x_val_list):
        # for survey_i, config.loc_set.zone.short_spread in enumerate(x_val_list):
        # for survey_i, config.loc_set.zone.long_spread in enumerate(x_val_list):
        # for survey_i, config.loc_set.zone.dt_k in enumerate(x_val_list):
        # for survey_i, config.loc_set.zone.dc_period in enumerate(x_val_list):
        # for survey_i, config.loc_set.zone.ei_k in enumerate(x_val_list):
        # for survey_i, config.loc_set.zone.dr_error in enumerate(x_val_list):
        # for survey_i, config.loc_set.zone.bbz_itv in enumerate(itv_list):
        # for survey_i, config.loc_set.zone.gap_mply in enumerate(x_val_list):
        # for survey_i, config.loc_set.zone.ad_idx in enumerate(x_val_list):
        # for survey_i, config.loc_set.point.tf_entry in enumerate(x_val_list):
        # for survey_i, config.loc_set.point.candle_ratio in enumerate(x_val_list):
        # for survey_i, config.loc_set.point.body_ratio in enumerate(x_val_list):
        # for survey_i, config.loc_set.point.candle_ratio2 in enumerate(x_val_list):
        # for survey_i, config.loc_set.point.body_ratio2 in enumerate(x_val_list):
        # for survey_i, config.loc_set.point.osc_band in enumerate(x_val_list):
        # for survey_i, config.loc_set.point2.ce_gap in enumerate(x_val_list):
        # for survey_i, config.tr_set.ep_gap in enumerate(x_val_list):
        # for survey_i, config.tr_set.out_gap in enumerate(x_val_list):
        # for survey_i, config.tr_set.tp_gap in enumerate(x_val_list):
        # for survey_i, config.lvrg_set.leverage in enumerate(x_val_list):
        # for survey_i, config.lvrg_set.target_pct in enumerate(x_val_list):
        # for survey_i, config.lvrg_set.target_pct in enumerate(x_val_list):
        # for survey_i, config.tp_set.decay_term in enumerate(x_val_list):
        # for survey_i, outg_dc_itv_num in enumerate(x_val_list):
        # for survey_i, exp_itv in enumerate(itv_list):
        # for survey_i, zone_dt_k in enumerate(x_val_list):
        # for survey_i, t_out_gap in enumerate(x_val_list):
        # for survey_i, zone_dc_period in enumerate(x_val_list):
        # for survey_i, early_out_tpg in enumerate(x_val_list):
        # for survey_i, ema_period in enumerate(x_val_list):

        start_0 = time.time()

        try:

            print("config.strat_version :", config.strat_version)
            print("config.loc_set.point.exp_itv :", config.loc_set.point.exp_itv)
            print("config.loc_set.point.tpg_itv1 :", config.loc_set.point.tpg_itv1)
            print("config.loc_set.point.tpg_itv0 :", config.loc_set.point.tpg_itv0)
            print("config.loc_set.point.outg_itv1 :", config.loc_set.point.outg_itv1)
            print("config.loc_set.point.outg_itv0 :", config.loc_set.point.outg_itv0)
            print("config.loc_set.point.outg_dc_period :", config.loc_set.point.outg_dc_period)
            print("-----------------------------------")
            # print("dtk_dc_itv :", dtk_dc_itv)
            # print("config.loc_set.dtk_dc_itv_num :", config.loc_set.dtk_dc_itv_num :",)
            print("config.loc_set.zone.short_spread :", config.loc_set.zone.short_spread)
            print("config.loc_set.zone.long_spread :", config.loc_set.zone.long_spread)
            print("config.loc_set.zone.tr_thresh :", config.loc_set.zone.tr_thresh)
            print("config.loc_set.zone.dtk_itv :", config.loc_set.zone.dtk_itv)
            print("config.loc_set.zone.dt_k :", config.loc_set.zone.dt_k)
            print("config.loc_set.zone.ei_k :", config.loc_set.zone.ei_k)
            print("config.loc_set.zone.dc_period :", config.loc_set.zone.dc_period)
            print("config.loc_set.zone.use_dtk_line :", config.loc_set.zone.use_dtk_line)

            print("config.loc_set.zone.zone_rejection :", config.loc_set.zone.zone_rejection)
            print("config.loc_set.zone.bbz_itv :", config.loc_set.zone.bbz_itv)
            print("config.loc_set.zone.gap_mply :", config.loc_set.zone.gap_mply)
            print("config.loc_set.zone.ad_idx :", config.loc_set.zone.ad_idx)
            print("config.loc_set.zone.zone_dt_k :", config.loc_set.zone.zone_dt_k)
            print("config.loc_set.zone.zone_dc_period :", config.loc_set.zone.zone_dc_period)
            # print("config.loc_set.open_shift :", config.loc_set.open_shift)
            print("-----------------------------------")
            # print("config.ep_set.dr_error :", config.ep_set.dr_error)
            print("config.loc_set.point.tf_entry :", config.loc_set.point.tf_entry)
            print("config.loc_set.point.htf_entry :", config.loc_set.point.htf_entry)
            print("config.loc_set.point.candle_ratio :", config.loc_set.point.candle_ratio)
            print("config.loc_set.point.body_ratio :", config.loc_set.point.body_ratio)
            print("config.loc_set.point.candle_ratio2 :", config.loc_set.point.candle_ratio2)
            print("config.loc_set.point.body_ratio2 :", config.loc_set.point.body_ratio2)
            print("config.loc_set.point.osc_band :", config.loc_set.point.osc_band)
            print("config.loc_set.point2.ce_gap :", config.loc_set.point2.ce_gap)
            print("config.tr_set.ep_gap :", config.tr_set.ep_gap)
            print("config.tr_set.tp_gap :", config.tr_set.tp_gap)
            print("config.tr_set.decay_gap :", config.tr_set.decay_gap)
            print("config.tr_set.out_gap :", config.tr_set.out_gap)
            print("config.tr_set.c_ep_gap :", config.tr_set.c_ep_gap)
            print("config.tr_set.t_out_gap :", config.tr_set.t_out_gap)
            print("-----------------------------------")
            print("config.lvrg_set.leverage :", config.lvrg_set.leverage)
            print("config.lvrg_set.static_lvrg :", config.lvrg_set.static_lvrg)
            print("config.lvrg_set.target_pct :", config.lvrg_set.target_pct)
            print("-----------------------------------")
            print("config.ep_set.entry_type :", config.ep_set.entry_type)
            print("config.tp_set.tp_type :", config.tp_set.tp_type)
            print("config.tp_set.static_tp :", config.tp_set.static_tp)
            print("config.tp_set.decay_term :", config.tp_set.decay_term)
            print("rsi_out_stratver :", rsi_out_stratver)
            print("config.out_set.use_out :", config.out_set.use_out)
            print("config.out_set.out_type :", config.out_set.out_type)

        except Exception as e:
            print(e)

        #       temp survey     #
        # if 'bb_upper_15m' in res_df.columns:
        #   res_df.drop(['bb_upper_15m', 'bb_lower_15m'], axis=1, inplace=True)
        # res_df = bb_level(res_df, '15m', config.loc_set.zone.gap_mply)

        # ema_period = 155
        # print("ema_period :", ema_period)

        # df_5T = to_htf(sliced_df1, '5T', offset='1h')
        # df_5T['ema_5m'] = ema(df_5T['close'], ema_period)   # ema formula issue
        # res_df.drop(['ema_5m'], axis=1, inplace=True, errors='ignore')
        # res_df = res_df.join(to_lower_tf_v2(res_df, df_5T, [-1]), how='inner')

        # rsi_upper = 50 + config.loc_set.point.osc_band
        # rsi_lower = 50 - config.loc_set.point.osc_band

        if override:
            res_df = enlist_tr(res_df, config, np_timeidx)
        else:
            for utils_, cfg_ in zip(utils_list, config_list):   # recursively
                res_df = utils_.enlist_tr(res_df, cfg_, np_timeidx)
                if not multi_mode:
                    break

        print("enlist_rtc ~ enlist_tr elapsed time :", time.time() - start_0)

        #       trading : 여기도 체결 결과에 대해 묘사함       #
        trade_list = []
        h_trade_list = []
        leverage_list = []
        fee_list = []
        short_fee_list = []
        long_fee_list = []
        open_list = []
        zone_list = []
        side_list = []
        strat_ver_list = []

        tp_ratio_list = []
        short_tp_ratio_list = []
        long_tp_ratio_list = []

        dr_list = []
        short_dr_list = []
        long_dr_list = []

        liqd_list = []
        short_liqd_list = []
        long_liqd_list = []

        nontp_liqd_list = []
        nontp_short_liqd_list = []
        nontp_long_liqd_list = []

        nontp_pr_list = []
        nontp_short_pr_list = []
        nontp_long_pr_list = []

        nontp_short_indexs = []
        nontp_long_indexs = []

        nontp_short_ep_list = []
        nontp_long_ep_list = []

        pr_list = []
        long_list = []
        short_list = []

        h_pr_list = []
        h_long_list = []
        h_short_list = []

        ep_tp_list = []
        h_ep_tp_list = []
        tp_state_list = []

        i = 0
        while 1:
            # for i in range(len(res_df)):

            run = 0
            open_side = None

            for utils_, cfg_ in zip(utils_list, config_list):

                #       entry_score     #
                if res_df['entry_{}'.format(cfg_.strat_version)][i] == cfg_.ep_set.short_entry_score:

                    #       ep_loc      #
                    if override or public_override:
                        res_df, open_side, zone = short_ep_loc(res_df, cfg_,
                                                            i,
                                                            np_timeidx, show_detail)
                    else:
                        res_df, open_side, zone = utils_public.short_ep_loc(res_df, cfg_,
                                                                          i,
                                                                          np_timeidx, show_detail)
                    if open_side is not None:   # 조건 만족시 바로 break
                        #       assign      #
                        config = cfg_
                        break

                #       entry_score     #
                elif res_df['entry_{}'.format(cfg_.strat_version)][i] == -cfg_.ep_set.short_entry_score:

                    #       ep_loc      #
                    if override or public_override:
                        res_df, open_side, zone = long_ep_loc(res_df, cfg_,
                                                               i,
                                                               np_timeidx, show_detail)
                    else:
                        res_df, open_side, zone = utils_public.long_ep_loc(res_df, cfg_,
                                                                            i,
                                                                            np_timeidx, show_detail)
                    if open_side is not None:
                        #       assign      #
                        config = cfg_
                        break

                if not multi_mode:
                    break

            if open_side is None:
                i += 1
                if i >= len(res_df):
                    break
                continue

            if open_side == utils_public.OrderSide.SELL:

                initial_i = i
                # print("short_ep_loc passed !")

                # --------------------- config 가 확정된 이후의 setting --------------------- #
                strat_version = config.strat_version

                # ------- tp / out fee calc ------- #
                tp_fee, out_fee = calc_tp_out_fee(config)

                # p_i 의 용도 모르겠음
                # if config.out_set.static_out:
                #     p_i = initial_i
                # else:
                #     p_i = i

                # ------- fee init ------- #
                if config.ep_set.entry_type == 'LIMIT':
                    fee = config.trader_set.limit_fee
                else:
                    fee = config.trader_set.market_fee

                # --------------- set partial tp --------------- #
                short_tps = [res_df['short_tp_{}'.format(strat_version)]]
                # short_tps = [short_tp2, short_tp] # org
                # short_tps = [short_tp, short_tp2]

                ep_j = initial_i
                out_j = initial_i

                # -------------- limit waiting : limit_out -------------- #

                if config.ep_set.entry_type == "LIMIT":

                    # allow_ep_in = 0 if strat_version in ['v5_2'] else 1
                    allow_ep_in = 0
                    entry_done = 0
                    entry_open = 0
                    prev_sar = None

                    # for e_j in range(i, len(res_df)): # entry_signal 이 open 기준 (해당 bar 에서 체결 가능함)
                    if i + 1 >= len(res_df):  # i should be checked if e_j starts from i+1
                        break
                    for e_j in range(i + 1, len(res_df)):  # entry signal이 close 기준 일 경우

                        if not config.ep_set.static_ep:
                            ep_j = e_j
                            out_j = e_j

                        if config.tp_set.static_tp:
                            # if config.ep_set.tpout_onexec:
                            #   tp_j = e_j
                            # else:
                            tp_j = initial_i
                        else:
                            tp_j = e_j

                            #             1. ep 설정
                        # -------------- np.inf ep -------------- #
                        # if short_ep.iloc[initial_i] == np.inf:
                        #   break

                        #     1. check ep_out     #
                        if res_df['low'].iloc[e_j] <= res_df['h_short_rtc_1_{}'.format(strat_version)].iloc[tp_j] - \
                                res_df['h_short_rtc_gap_{}'.format(strat_version)].iloc[
                                    tp_j] * config.loc_set.zone.ei_k:
                            break

                            # elif strat_version == 'v5_2':
                            # if res_df['low'].iloc[e_j] <= res_df['short_tp_{}'.format(strat_version)].iloc[tp_j]: # ep_out : tp_done
                            # # if np_timeidx[e_j] % config.loc_set.point.tf_entry == config.loc_set.point.tf_entry - 1:
                            #   break

                            # elif (res_df['rsi_%s' % config.loc_set.point.exp_itv].iloc[e_j - 1] >= 50 - config.loc_set.point.osc_band) & \
                            #                  (res_df['rsi_%s' % config.loc_set.point.exp_itv].iloc[e_j] < 50 - config.loc_set.point.osc_band):
                            #   break

                        # if config.loc_set.zone.c_itv_ticks != "None":

                        #   if np_timeidx[e_j] % config.loc_set.zone.c_itv_ticks == config.loc_set.zone.c_itv_ticks - 1:
                        #     break

                        #     2. ep_loc.point2
                        if override:
                            allow_ep_in, out_j = short_point2(res_df, config, e_j, out_j, allow_ep_in)
                        else:
                            allow_ep_in, out_j = utils_public.short_point2(res_df, config, e_j, out_j,
                                                                           allow_ep_in)  # not defined yet,

                        #     3. check ep_in       #
                        if allow_ep_in and res_df['high'].iloc[e_j] >= res_df['short_ep_{}'.format(strat_version)].iloc[
                            ep_j]:
                            entry_done = 1
                            # print("res_df['high'].iloc[e_j] :", res_df['high'].iloc[e_j])
                            # print("e_j :", e_j)

                            #     이미, e_j open 이 ep 보다 높은 경우, entry[ep_j] => -2 로 변경   #
                            if res_df['open'].iloc[e_j] >= res_df['short_ep_{}'.format(strat_version)].iloc[ep_j]:
                                entry_open = 1
                            break

                    i = e_j
                    # print("i = e_j :", i)

                    if entry_done:
                        pass

                    else:
                        i += 1
                        if i >= len(res_df):
                            break
                        continue

                # ----------------- end wait ----------------- #

                # if e_j - initial_i >= 200:
                #   print("e_j, initial_i :", e_j, initial_i)
                # print("e_j - initial_i :", e_j - initial_i)
                # print()

                open_list.append(initial_i)
                zone_list.append(zone)
                side_list.append('s')
                strat_ver_list.append(strat_version)

                #     e_j 라는 변수는 MARKET 에 있어서 정의되서는 안되는 변수임   #
                if config.ep_set.entry_type == 'MARKET':
                    # try:
                    #   ep_list = [res_df['close'].iloc[e_j]]
                    # except Exception as e:
                    #   # print('error in ep_list (initial) :', e)
                    ep_list = [res_df['close'].iloc[ep_j]]

                else:
                    if not entry_open:
                        ep_list = [res_df['short_ep_{}'.format(strat_version)].iloc[ep_j]]

                    else:
                        #   ep_j 는 항상 있음, LIMIT 인 경우 e_j 도 항상 존재함 --> dynamic_ep 여부에 따라 ep_j = e_j 가 되는 경우만 존재할 뿐임
                        #   따라서, ep_j 로 통일 가능함 (dynamic_ep 인 경우, ep_j = e_j 되어있음)
                        fee = config.trader_set.market_fee
                        ep_list = [res_df['open'].iloc[e_j]]  # --> 체결이 되는 e_j idx 기준으로 하는게 맞음

                if not config.lvrg_set.static_lvrg:

                    ep_ = ep_list[0]
                    out_ = res_df['short_out_{}'.format(strat_version)].iloc[out_j]
                    if override:
                        config.lvrg_set.leverage = lvrg_set(res_df, config, "SELL", ep_, out_, fee)
                    else:
                        config.lvrg_set.leverage = utils_public.lvrg_set(res_df, config, "SELL", ep_, out_, fee)

                    # -------------- leverage rejection -------------- #
                    if config.lvrg_set.leverage == None:
                        open_list.pop()
                        zone_list.pop()
                        side_list.pop()
                        strat_ver_list.pop()

                        i += 1
                        if i >= len(res_df):
                            break
                        continue

                leverage_list.append(config.lvrg_set.leverage)

                # try:
                if config.ep_set.entry_type == "MARKET":
                    ep_idx_list = [
                        ep_j]  # ep_j 는 ep_type 유관하게 존재하는 변수니까 try 에 걸어두는게 맞음 <-- # market 인데, e_j 변수가 할당된 경우 고려해야함
                else:
                    ep_idx_list = [e_j]

                out_idx_list = [out_j]

                # except Exception as e:
                #   # print('error in ep_idx_list :', e)
                #   ep_idx_list = [e_j]

                tp_list = []
                tp_idx_list = []

                partial_tp_cnt = 0
                hedge_cnt = 1

                h_ep, h_tp = None, None
                h_i, h_j = None, None

                trade_done = 0
                cross_on = 0
                out = 0
                # config.out_set.retouch

                #     Todo    #
                #      1. future_work : 상단의 retouch 와 겹침
                config.out_set.retouch = 0

                if i == len(res_df) - 1:  # if j start from i + 1
                    open_list.pop()
                    zone_list.pop()
                    side_list.pop()
                for j in range(i + 1, len(res_df)):

                    # for j in range(i, len(res_df)):

                    if config.tp_set.static_tp:
                        if not config.ep_set.static_ep and config.ep_set.entry_type == "LIMIT" and config.ep_set.tpout_onexec:
                            tp_j = ep_j  # tpout_onexec = using dynamic_ep --> using ep_j 에 대한 이유
                        else:
                            tp_j = initial_i
                    else:
                        tp_j = j

                    if config.out_set.static_out:
                        if not config.ep_set.static_ep and config.ep_set.entry_type == "LIMIT" and config.ep_set.tpout_onexec:
                            out_j = ep_j
                        # else:
                        #   out_j = initial_i   # --> referenced upper phase as initail_j / e_j (start of limit wait)
                    else:
                        out_j = j

                    # -------------- sub ep -------------- #
                    # if res_df['high'].iloc[j - 1] <= res_df['sar2'].iloc[j - 1] and res_df['high'].iloc[j] > res_df['sar2'].iloc[j]:

                    #   sub_ep = res_df['sar2'].iloc[j - 1]

                    #   if sub_ep < ep_list[-1]:
                    #     ep_list.append(sub_ep)
                    #     ep_idx_list.append(j)

                    # -------------- hedge only once -------------- #
                    #             일단, h_quantity 는 초기 진입과 동일하게 설정         #
                    # if res_df['high'].iloc[j] >= res_df['minor_ST2_Up'].iloc[j] and hedge_cnt == 1:
                    # if res_df['close'].iloc[j] >= res_df['minor_ST2_Up'].iloc[j] and hedge_cnt == 1:
                    # if res_df['close'].iloc[j] >= res_df['minor_ST3_Up'].iloc[j] and hedge_cnt == 1:

                    #   h_ep = res_df['close'].iloc[j]
                    #   hedge_cnt -= 1
                    #   h_i = j

                    # -------------- ultimate limit tp -------------- #
                    if not config.tp_set.non_tp:

                        #               1. by price line             #
                        if config.tp_set.tp_type == 'LIMIT' or config.tp_set.tp_type == "BOTH":

                            for s_i, short_tp_ in enumerate(short_tps):

                                #     decay adjustment    #
                                #     tp_j includes dynamic_j   #
                                try:
                                    if config.tr_set.decay_gap != "None":
                                        decay_share = (j - initial_i) // config.tp_set.decay_term
                                        decay_remain = (j - initial_i) % config.tp_set.decay_term
                                        if j != initial_i and decay_remain == 0:
                                            short_tp_.iloc[tp_j] += \
                                            res_df['h_short_rtc_gap_{}'.format(strat_version)].iloc[
                                                initial_i] * config.tr_set.decay_gap * decay_share

                                except:
                                    pass

                                if res_df['low'].iloc[j] <= short_tp_.iloc[
                                    tp_j] and partial_tp_cnt == s_i:  # we use static tp now
                                    # if res_df['low'].iloc[j] <= short_tp_.iloc[j]:
                                    # if res_df['low'].iloc[j] <= short_tp_.iloc[j] <= res_df['high'].iloc[j]: --> 이건 잘못되었음

                                    if s_i == len(short_tps) - 1:
                                        trade_done = 1

                                    partial_tp_cnt += 1

                                    #         dynamic tp        #
                                    # if 0:
                                    if short_tp_.iloc[j] != short_tp_.iloc[j - 1] and not config.tp_set.static_tp:

                                        #   tp limit 이 불가한 경우 = open 이 이미, tp 를 넘은 경우 #
                                        # if res_df['open'].iloc[j] < short_tp_.iloc[initial_i]:
                                        if res_df['open'].iloc[j] < short_tp_.iloc[j]:

                                            # tp = short_tp_.iloc[initial_i]
                                            tp = res_df['open'].iloc[j]

                                            if trade_done:
                                                tp_state_list.append("d-short_open {}".format(strat_version))

                                        #   tp limit 이 가능한 경우 = open 이 아직, tp 를 넘지 않은 경우 #
                                        else:

                                            # tp = short_tp_.iloc[initial_i]
                                            tp = short_tp_.iloc[j]
                                            # tp = res_df['open'].iloc[j]

                                            if trade_done:
                                                tp_state_list.append("d-short_tp {}".format(strat_version))

                                    #         static tp         #
                                    else:

                                        #   tp limit 이 불가한 경우 = open 이 이미, tp 를 넘은 경우 #
                                        #   non_inversion 의 경우, short_tp 가 가능함   #

                                        # if res_df['open'].iloc[j] < short_tp_.iloc[initial_i]:
                                        if res_df['open'].iloc[j] < short_tp_.iloc[tp_j]:

                                            # tp = short_tp_.iloc[initial_i]

                                            if config.tr_set.decay_gap != "None" and decay_remain == 0:
                                                tp = res_df['open'].iloc[
                                                    j]  # tp_j -> initial_i 를 가리키기 때문에 decay 는 한번만 진행되는게 맞음
                                            else:
                                                tp = short_tp_.iloc[tp_j]

                                            if trade_done:
                                                tp_state_list.append("s-short_tp {}".format(strat_version))

                                        #   tp limit 이 가능한 경우 = open 이 아직, tp 를 넘지 않은 경우 #
                                        else:

                                            # tp = short_tp_.iloc[initial_i]
                                            tp = short_tp_.iloc[tp_j]

                                            # tp = res_df['open'].iloc[j]

                                            if trade_done:
                                                tp_state_list.append("s-short_tp {}".format(strat_version))

                                    tp_list.append(tp)
                                    tp_idx_list.append(j)
                                    fee += config.trader_set.limit_fee

                        #           2. by signal        #
                        if config.tp_set.tp_type == 'MARKET' or (config.tp_set.tp_type == "BOTH" and not trade_done):

                            market_tp = 0

                            # -------------- sar tp -------------- #
                            # if (res_df['high'].iloc[j] >= res_df['sar2'].iloc[j]) & \
                            #   (res_df['high'].iloc[j - 1] < res_df['sar2'].iloc[j - 1]) & \
                            #   (res_df['high'].iloc[j - 2] < res_df['sar2'].iloc[j - 2]):

                            #       inversion     #
                            # if (res_df['low'].iloc[j] <= res_df['sar2'].iloc[j]) & \
                            #   (res_df['low'].iloc[j - 1] > res_df['sar2'].iloc[j - 1]) & \
                            #   (res_df['low'].iloc[j - 2] > res_df['sar2'].iloc[j - 2]):

                            # ----------- st short ----------- #
                            # if res_df['close'].iloc[j] <= res_df['short_tp'].iloc[tp_j]:

                            # -------------- sar pb tp -------------- #
                            # if res_df['low'].iloc[j] <= res_df['short_tp'].iloc[initial_i]:

                            # -------------- st tp -------------- #
                            # if res_df['close'].iloc[j] > res_df['middle_line'].iloc[j]:

                            # -------------- fisher tp -------------- #
                            # if entry[j] == 1:

                            # -------------- timestamp -------------- #
                            if config.tp_set.time_tp:
                                if np_timeidx[
                                    j] % config.loc_set.zone.c_itv_ticks == config.loc_set.zone.c_itv_ticks - 1 and \
                                        j - initial_i >= config.loc_set.zone.c_itv_ticks:
                                    market_tp = 1

                                    # -------------- rsi -------------- #
                            if strat_version in rsi_out_stratver:
                                if (res_df['rsi_%s' % config.loc_set.point.exp_itv].iloc[
                                        j - 1] >= 50 - config.loc_set.point.osc_band) & \
                                        (res_df['rsi_%s' % config.loc_set.point.exp_itv].iloc[
                                             j] < 50 - config.loc_set.point.osc_band):
                                    # if (res_df['rsi_%s' % config.loc_set.point.exp_itv].iloc[j - 1] <= 50 - config.loc_set.point.osc_band) & \
                                    #                  (res_df['rsi_%s' % config.loc_set.point.exp_itv].iloc[j] > 50 - config.loc_set.point.osc_band):
                                    market_tp = 1

                                # -------------- cci -------------- #
                                # if (res_df['cci_%s' % config.loc_set.point.exp_itv].iloc[j - 1] >= -config.loc_set.point.osc_band) & \
                                #                  (res_df['cci_%s' % config.loc_set.point.exp_itv].iloc[j] < -config.loc_set.point.osc_band):
                                #   market_tp = 1

                            # ---------------------------- early out ---------------------------- #

                            # #         rsi slight touch        #
                            if allow_osc_touch:
                                if (np.min(res_df['rsi_%s' % config.loc_set.point.exp_itv].iloc[
                                           initial_i:j]) < 50 - config.loc_set.point.osc_band + rsi_gap) & \
                                        (res_df['rsi_%s' % config.loc_set.point.exp_itv].iloc[j] >= 50):
                                    market_tp = 1

                                    #           tp early out          #
                            # # if (np.min(res_df['low'].iloc[e_j:j]) < res_df['short_tp'].iloc[tp_j]) & \
                            # if (np.min(res_df['low'].iloc[e_j:j]) < res_df['h_short_rtc_1'].iloc[tp_j] - res_df['h_short_rtc_gap'].iloc[tp_j] * early_out_tpg) & \
                            #   (res_df['close'].iloc[j] >= res_df['short_ep'].iloc[ep_j]):
                            #   market_tp = 1

                            # if strat_version == "v7":
                            #   if res_df['dc_upper_1m'].iloc[j] > res_df['dc_upper_5m'].iloc[j]:
                            #     market_tp = 1

                            #         bb_upper early out        # --> cross_on 기능은 ide latency 개선 여부에 해당되지 않음
                            if strat_version in ['v5_2']:
                                if res_df['close'].iloc[j] < res_df['bb_lower_5m'].iloc[j] < res_df['close'].iloc[
                                    j - 1]:
                                    cross_on = 1

                                if cross_on == 1 and res_df['close'].iloc[j] > res_df['bb_upper_5m'].iloc[j] > \
                                        res_df['close'].iloc[j - 1]:
                                    market_tp = 1

                            if market_tp:

                                tp = res_df['close'].iloc[j]
                                # tp = res_df['open'].iloc[j]
                                trade_done = 1

                                if trade_done:
                                    tp_state_list.append("short close tp")

                                tp_list.append(tp)
                                tp_idx_list.append(j)
                                fee += config.trader_set.market_fee

                    # -------------- out -------------- #
                    if not trade_done and config.out_set.use_out and j != len(res_df) - 1:

                        # -------------- macd -------------- #
                        # if res_df['macd_hist3'].iloc[j] > 0:  #  macd out
                        # if res_df['macd_hist3'].iloc[i] < 0 and res_df['macd_hist3'].iloc[j] > 0:

                        # -------------- st config.out_set.retouch -------------- #
                        # out = 1 상태면 동일 tick 에서 config.out_set.retouch 를 조사할 거기 때문에, 먼저 검사함
                        # 그리고, out 기준이 close 라 이게 맞음
                        # close 가 short_out 보다 올라가있는 상태일테니 low 를 조사하는게 맞음
                        # if out and res_df['low'].iloc[j] <= short_out.iloc[out_j]:
                        #   config.out_set.retouch = 1

                        # ------- 일정시간 이상, dynamic_out 적용 ------ #
                        try:
                            if j - out_idx >= config.out_set.retouch_out_period:
                                static_short_out = res_df['short_out_{}'.format(strat_version)].iloc[j]

                        except Exception as e:
                            pass

                            # ------- static out ------ #
                        try:
                            if out and res_df['low'].iloc[j] <= static_short_out:
                                config.out_set.retouch = 1
                        except Exception as e:
                            pass

                            # ------- config.out_set.retouch out ------ #
                        # if out and res_df['low'].iloc[j] <= short_out2.iloc[out_j]:
                        #   config.out_set.retouch = 1

                        # -------------- st -------------- #
                        # if res_df['close'].iloc[j] > res_df['middle_line'].iloc[j]:
                        # if res_df['close'].iloc[j] > res_df['minor_ST3_Up'].iloc[j]:
                        # if res_df['close'].iloc[j] > upper_middle.iloc[j]:
                        # if res_df['close'].iloc[j] > res_df['minor_ST1_Up'].iloc[j]:
                        if out == 0:
                            if config.out_set.hl_out:
                                if res_df['high'].iloc[j] >= res_df['short_out_{}'.format(strat_version)].iloc[
                                    out_j]:  # check out only once
                                    out = 1

                            else:
                                if res_df['close'].iloc[j] >= res_df['short_out_{}'.format(strat_version)].iloc[
                                    out_j]:  # check out only once
                                    out = 1

                            # out_idx = j
                            # static_short_out = short_out.iloc[out_j]
                            # if config.out_set.second_out:
                            # static_short_out = short_out.iloc[out_j] + res_df['st_gap'].iloc[out_j] * config.out_set.second_out_gap

                        # if out == 0 and res_df['high'].iloc[j] >= short_out.iloc[out_j]: # check out only once
                        #   out = 1

                        # -------------- sma -------------- #
                        # if res_df['close'].iloc[j] > res_df[sma].iloc[j]:

                        # -------------- sar -------------- #
                        # if res_df['close'].iloc[j] > res_df['minor_ST3_Up'].iloc[j] \
                        #   or res_df['sar2'].iloc[j] <= res_df['high'].iloc[j]:
                        # if res_df['close'].iloc[j] > short_out.iloc[initial_i]: # or \
                        #   out = 1
                        # res_df['sar2_uptrend'].iloc[j] == 1: # or \

                        # if res_df['sar2_uptrend'].iloc[j] == 1:

                        #   if prev_sar is None:
                        #     prev_sar = res_df['sar2'].iloc[j - 1]

                        #   if res_df['close'].iloc[j] > prev_sar:
                        #     out = 1

                        # else:
                        #   if res_df['close'].iloc[j] > res_df['sar2'].iloc[j]:
                        #     out = 1

                        # -------------- hl -------------- #
                        # if res_df['close'].iloc[j] > short_out.iloc[tp_j]:

                        # -------------- stoch -------------- #
                        # if res_df['stoch'].iloc[j - 2] >= res_df['stoch'].iloc[j - 1] and \
                        #   res_df['stoch'].iloc[j - 1] < res_df['stoch'].iloc[j] and \
                        #   res_df['stoch'].iloc[j - 1] <= stoch_lower:
                        #   out = 1

                        # config.out_set.retouch 1 경우, config.out_set.retouch 조건도 있어야함
                        if out:
                            if config.out_set.retouch:
                                if config.out_set.retouch:
                                    pass
                                else:
                                    continue

                            else:
                                pass

                            if config.out_set.price_restoration:
                                tp = res_df['short_out_{}'.format(strat_version)].iloc[out_j]
                                if config.out_set.second_out:
                                    tp = res_df['short_out2_{}'.format(strat_version)].iloc[out_j]

                                # if res_df['close'].iloc[j] > tp: # 이 경우를 protect 하는건 insane 임
                                #   tp = res_df['close'].iloc[j]

                            else:

                                if res_df['open'].iloc[j] >= res_df['short_out_{}'.format(strat_version)].iloc[out_j]:
                                    tp = res_df['open'].iloc[j]
                                else:
                                    if config.out_set.hl_out:
                                        tp = res_df['short_out_{}'.format(strat_version)].iloc[out_j]
                                    else:
                                        tp = res_df['close'].iloc[j]

                                # if not config.out_set.static_out:
                                #   if res_df['open'].iloc[j] >= res_df['short_out_{}'.format(strat_version)].iloc[out_j]: # close 기준이라 이런 조건을 못씀, 차라리 j 를 i 부터 시작
                                #     tp = res_df['open'].iloc[j]
                                #   else:
                                #     tp = res_df['close'].iloc[j]

                                # else:
                                #   tp = res_df['close'].iloc[j]

                            if config.out_set.retouch:  # out 과 open 비교
                                if config.out_set.second_out:
                                    if res_df['open'].iloc[j] <= res_df['short_out2_{}'.format(strat_version)].iloc[
                                        out_j]:
                                        tp = res_df['open'].iloc[j]
                                else:
                                    if res_df['open'].iloc[j] <= res_df['short_out_{}'.format(strat_version)].iloc[
                                        out_j]:
                                        tp = res_df['open'].iloc[j]

                                try:  # static_short_out 인 경우, open 도 고려한 tp set
                                    if res_df['open'].iloc[j] <= static_short_out:
                                        tp = res_df['open'].iloc[j]
                                    else:
                                        tp = static_short_out
                                except Exception as e:
                                    pass

                            trade_done = 1
                            tp_state_list.append("short close_out {}".format(strat_version))

                            tp_list.append(tp)
                            tp_idx_list.append(j)
                            fee += config.trader_set.market_fee

                    # -------------- non tp -------------- #
                    if j == len(res_df) - 1:
                        trade_done = 1
                        tp = res_df['close'].iloc[j]
                        tp_list.append(tp)
                        tp_idx_list.append(j)
                        fee += config.trader_set.market_fee

                    # -------------- append trade data -------------- #
                    if trade_done:

                        # --------------- tp_ratio info --------------- #
                        #         Todo        #
                        #          short_out 에 대한 정보는 존재함,
                        #          short_tp 에 대한 정보는 존재함,
                        #       => initial_i 기준으로 ,dynamic | static set 을 tp 와 out 에 각각 적용
                        #          config.lvrg_set.leverage 는 initial_i 기준으로 적용되니까
                        #          적용된 tp & out 으로 abs((tp - ep) / (ep - out)) 계산
                        try:
                            if config.out_set.use_out:
                                done_tp = res_df['short_tp_{}'.format(strat_version)].iloc[tp_j]
                                done_out = res_df['short_out_{}'.format(strat_version)].iloc[out_j]

                                if done_out <= ep_list[0]:  # loss > 1
                                    dr = np.nan
                                    tp_ratio = np.nan
                                else:
                                    dr = ((ep_list[0] - done_tp) / (done_out - ep_list[0]))
                                    tp_ratio = ((ep_list[0] - done_tp - tp_fee * ep_list[0]) / (
                                                done_out - ep_list[0] + out_fee * ep_list[0]))

                            else:
                                dr = np.nan
                                tp_ratio = np.nan


                        except Exception as e:
                            dr = np.nan
                            tp_ratio = np.nan

                        tp_ratio_list.append(tp_ratio)
                        short_tp_ratio_list.append(tp_ratio)
                        dr_list.append(dr)
                        short_dr_list.append(dr)

                        # -------------------- partial tp -------------------- #
                        #        1. len(tp_list) 에 대응하는 qty_list 를 만들어야함    #
                        #        2. temp_pr_list 를 만들어 총합 + 1 을 pr_list 에 저장      #
                        #        2-1. temp_pr = sum((ep / tp_list[i] - fee - 1) * qty_list[i])   #
                        #        3. temp_pr_list 의 첫 tp 에는 r_qty 를 할당함        #
                        qty_list = []
                        temp_pr_list = []
                        r_qty = 1
                        for q_i in range(len(tp_list) - 1, -1, -1):

                            if len(tp_list) == 1:
                                temp_qty = r_qty
                            else:
                                if q_i != 0:
                                    temp_qty = r_qty / config.tp_set.partial_qty_divider
                                else:
                                    temp_qty = r_qty

                            temp_pr = (ep_list[0] / tp_list[q_i] - fee - 1) * temp_qty * config.lvrg_set.leverage
                            # temp_pr = (ep_list[0] / tp_list[q_i] - fee - 1) * temp_qty
                            r_qty -= temp_qty

                            temp_pr_list.append(temp_pr)
                            qty_list.append(temp_qty)

                        # if len(temp_pr_list) == 1:
                        #   print("qty_list :", qty_list)
                        #   print("temp_pr_list :", temp_pr_list)

                        temp_pr = sum(temp_pr_list) + 1

                        # -------------------- sub ep -> pr calc -------------------- #
                        if len(ep_list) > 1:

                            p_ep_pr = []
                            for sub_ep_ in ep_list:
                                sub_pr = (sub_ep_ / tp - fee - 1) * config.lvrg_set.leverage
                                p_ep_pr.append(sub_pr)

                            temp_pr = sum(p_ep_pr) + 1

                            print("temp_pr :", temp_pr)

                        # ------------ hedge + non_hedge pr summation ------------ #
                        #         hedge pr direction is opposite to the origin       #
                        hedge_pr = 1
                        if hedge_cnt == 0:
                            #       hedge tp      #
                            h_tp = res_df['close'].iloc[j]
                            hedge_pr = (h_tp / h_ep - fee - 1) * config.lvrg_set.leverage  # hedge long
                            temp_pr += hedge_pr
                            h_j = j

                        # hh = max(res_df['high'].iloc[i:j + 1])
                        hh = max(res_df['high'].iloc[i:j])  # pos. 정리하기 바로 직전까지
                        short_liq = (ep_list[0] / hh - fee - 1) * config.lvrg_set.leverage + 1

                        if j != len(res_df) - 1:

                            # ep_tp_list.append((ep, tp_list))
                            ep_tp_list.append((ep_list, tp_list))
                            # trade_list.append([initial_i, i, j])
                            # trade_list.append((ep_idx_list, tp_idx_list))
                            trade_list.append((ep_idx_list, out_idx_list, tp_idx_list))

                            liqd_list.append(short_liq)
                            short_liqd_list.append(short_liq)

                            h_ep_tp_list.append(
                                (h_ep, h_tp))  # hedge 도 ep_tp_list 처럼 변경해주어야하는데 아직 안건드림, 딱히 사용할 일이 없어보여
                            h_trade_list.append([initial_i, h_i, h_j])

                            pr_list.append(temp_pr)
                            fee_list.append(fee)
                            short_list.append(temp_pr)
                            short_fee_list.append(fee)

                            h_pr_list.append(hedge_pr)
                            h_short_list.append(hedge_pr)

                            i = j
                            break

                        else:

                            # ep_tp_list.append((ep_list, tp_list))
                            # trade_list.append((ep_idx_list, tp_idx_list))
                            # plot_check 때문에, pr_list 까지 하게되면 acc_pr eval 이 꼬이게댐

                            # pr_list 를 넣지 않을거니까, open_list 에서 해당 idx 는 pop
                            open_list.pop()
                            zone_list.pop()
                            side_list.pop()
                            strat_ver_list.pop()

                            #         tp 미체결 survey        #
                            nontp_liqd_list.append(short_liq)
                            nontp_short_liqd_list.append(short_liq)
                            nontp_short_indexs.append(i)
                            nontp_short_ep_list.append(ep_list[0])

                            nontp_short_pr = (ep_list[0] / tp - fee - 1) * config.lvrg_set.leverage + 1
                            nontp_pr_list.append(nontp_short_pr)
                            nontp_short_pr_list.append(nontp_short_pr)


            #                  long  phase                #
            elif open_side == utils_public.OrderSide.BUY:

                initial_i = i
                # print("long_ep_loc passed !")

                strat_version = config.strat_version

                # ------- tp / out fee calc ------- #
                tp_fee, out_fee = calc_tp_out_fee(config)

                # ------- fee init ------- #
                if config.ep_set.entry_type == 'LIMIT':
                    fee = config.trader_set.limit_fee
                else:
                    fee = config.trader_set.market_fee

                # --------------- set partial tp --------------- #
                long_tps = [res_df['long_tp_{}'.format(strat_version)]]
                # long_tps = [long_tp2, long_tp]
                # long_tps = [long_tp, long_tp2]
                # print("i after long_ep_loc :", i)

                # if config.out_set.static_out:
                #     p_i = initial_i
                # else:
                #     p_i = i

                ep_j = initial_i
                out_j = initial_i

                # -------------- limit waiting const. -------------- #
                if config.ep_set.entry_type == "LIMIT":

                    # allow_ep_in = 0 if strat_version in ['v5_2'] else 1
                    allow_ep_in = 0
                    entry_done = 0
                    entry_open = 0
                    prev_sar = None

                    # for e_j in range(i, len(res_df)):

                    if i + 1 >= len(res_df):  # i should be checked if e_j starts from i+1
                        break
                    for e_j in range(i + 1, len(res_df)):  # entry 가 close 기준일 경우 사용 (open 기준일 경우 i 부터 시작해도 무방함)

                        if not config.ep_set.static_ep:
                            ep_j = e_j
                            out_j = e_j

                        if config.tp_set.static_tp:
                            # if config.ep_set.tpout_onexec:
                            #   tp_j = e_j
                            # else:
                            tp_j = initial_i
                        else:
                            tp_j = e_j

                            #          np.inf ep         #
                        # if long_ep.iloc[initial_i] == np.inf:
                        #   break

                        #     1. check ep_out     #
                        if res_df['high'].iloc[e_j] >= res_df['h_long_rtc_1_{}'.format(strat_version)].iloc[tp_j] + \
                                res_df['h_long_rtc_gap_{}'.format(strat_version)].iloc[tp_j] * config.loc_set.zone.ei_k:
                            # if res_df['high'].iloc[e_j] >= res_df['long_tp_{}'.format(strat_version)].iloc[tp_j]:
                            # if np_timeidx[e_j] % config.loc_set.point.tf_entry == config.loc_set.point.tf_entry - 1:
                            break

                            # elif (res_df['rsi_%s' % config.loc_set.point.exp_itv].iloc[e_j - 1] <= 50 + config.loc_set.point.osc_band) & \
                            #                  (res_df['rsi_%s' % config.loc_set.point.exp_itv].iloc[e_j] > 50 + config.loc_set.point.osc_band):
                            #   break

                        # if config.loc_set.zone.c_itv_ticks != "None":

                        #   if np_timeidx[e_j] % config.loc_set.zone.c_itv_ticks == config.loc_set.zone.c_itv_ticks - 1:
                        #     break

                        #     2. ep_loc.point2
                        if override:
                            allow_ep_in, out_j = long_point2(res_df, config, e_j, out_j, allow_ep_in)
                        else:
                            allow_ep_in, out_j = utils_public.long_point2(res_df, config, e_j, out_j,
                                                                          allow_ep_in)  # not defined yet,

                        #     3. check ep_in      #
                        if allow_ep_in and res_df['low'].iloc[e_j] <= res_df['long_ep_{}'.format(strat_version)].iloc[
                            ep_j]:
                            entry_done = 1
                            # print("e_j :", e_j)

                            #     이미, e_j open 이 ep 보다 낮은 경우, entry[initial_i] => -2 로 변경   #
                            if res_df['open'].iloc[e_j] <= res_df['long_ep_{}'.format(strat_version)].iloc[ep_j]:
                                entry_open = 1

                            break

                    i = e_j
                    # print("i = e_j :", i)

                    if entry_done:
                        pass
                        # print("i, entry_done :", i, entry_done)

                    else:
                        i += 1
                        if i >= len(res_df):
                            # print("i :", i)
                            break

                        # print("i in continue :", i)
                        continue

                # ---------------- end wait ---------------- #
                # if e_j - initial_i >= 200:
                #   print("e_j, initial_i :", e_j, initial_i)

                # print(i)

                open_list.append(initial_i)
                zone_list.append(zone)
                side_list.append('l')
                strat_ver_list.append(strat_version)

                if config.ep_set.entry_type == 'MARKET':
                    ep_list = [res_df['close'].iloc[ep_j]]
                else:
                    if not entry_open:
                        ep_list = [res_df['long_ep_{}'.format(strat_version)].iloc[
                                       ep_j]]  # dynamic_ep 인 경우에도 e_j 가 ep_j 로 대응되기 때문에 ep_j 만 사용해도 무관
                    else:
                        # try:
                        #   ep_list = [res_df['open'].iloc[e_j]]
                        # except Exception as e:
                        fee = config.trader_set.market_fee
                        ep_list = [res_df['open'].iloc[e_j]]  # --> 체결이 되는 e_j idx 기준으로 하는게 맞음

                if not config.lvrg_set.static_lvrg:

                    ep_ = ep_list[0]
                    out_ = res_df['long_out_{}'.format(strat_version)].iloc[out_j]
                    if override:
                        config.lvrg_set.leverage = lvrg_set(res_df, config, "BUY", ep_, out_, fee)
                    else:
                        config.lvrg_set.leverage = utils_public.lvrg_set(res_df, config, "BUY", ep_, out_, fee)

                    # -------------- leverage rejection -------------- #
                    if config.lvrg_set.leverage == None:
                        open_list.pop()
                        zone_list.pop()
                        side_list.pop()
                        strat_ver_list.pop()

                        i += 1
                        if i >= len(res_df):
                            break
                        continue

                leverage_list.append(config.lvrg_set.leverage)

                if config.ep_set.entry_type == "MARKET":
                    ep_idx_list = [ep_j]
                else:
                    ep_idx_list = [e_j]

                out_idx_list = [out_j]

                tp_list = []
                tp_idx_list = []

                partial_tp_cnt = 0
                hedge_cnt = 1

                h_ep, h_tp = None, None
                h_i, h_j = None, None

                trade_done = 0
                cross_on = 0
                out = 0
                config.out_set.retouch = 0

                if i == len(res_df) - 1:  # if j start from i + 1
                    open_list.pop()
                    zone_list.pop()
                    side_list.pop()
                    strat_ver_list.pop()

                for j in range(i + 1, len(res_df)):

                    # for j in range(i, len(res_df)):

                    if config.tp_set.static_tp:
                        if not config.ep_set.static_ep and config.ep_set.entry_type == "LIMIT" and config.ep_set.tpout_onexec:
                            tp_j = ep_j
                        else:
                            tp_j = initial_i
                    else:
                        tp_j = j

                    if config.out_set.static_out:
                        if not config.ep_set.static_ep and config.ep_set.entry_type == "LIMIT" and config.ep_set.tpout_onexec:
                            out_j = ep_j
                        # else:
                        #   out_j = initial_i
                    else:
                        out_j = j

                        # -------------- hedge only once -------------- #
                    #             일단, h_quantity 는 초기 진입과 동일하게 설정         #
                    # if res_df['low'].iloc[j] <= res_df['minor_ST2_Down'].iloc[j] and hedge_cnt == 1:
                    # if res_df['close'].iloc[j] <= res_df['minor_ST2_Down'].iloc[j] and hedge_cnt == 1:
                    # if res_df['close'].iloc[j] <= res_df['minor_ST3_Down'].iloc[j] and hedge_cnt == 1:

                    #   h_ep = res_df['close'].iloc[j]
                    #   hedge_cnt -= 1
                    #   h_i = j

                    # -------------- sub ep -------------- #
                    # if res_df['low'].iloc[j - 1] >= res_df['sar2'].iloc[j - 1] and res_df['low'].iloc[j] < res_df['sar2'].iloc[j]:

                    #   sub_ep = res_df['sar2'].iloc[j - 1]

                    #   if sub_ep > ep_list[-1]:
                    #     ep_list.append(sub_ep)
                    #     ep_idx_list.append(j)

                    # -------------- ultimate tp -------------- #
                    if not config.tp_set.non_tp:
                        #            1. by level          #
                        if config.tp_set.tp_type == "LIMIT" or config.tp_set.tp_type == "BOTH":

                            for l_i, long_tp_ in enumerate(long_tps):

                                #     decay adjustment    #
                                #     tp_j includes dynamic_j   #
                                try:
                                    if config.tr_set.decay_gap != "None":
                                        decay_share = (j - initial_i) // config.tp_set.decay_term
                                        decay_remain = (j - initial_i) % config.tp_set.decay_term
                                        if j != initial_i and decay_remain == 0:
                                            long_tp_.iloc[tp_j] -= \
                                            res_df['h_long_rtc_gap_{}'.format(strat_version)].iloc[
                                                initial_i] * config.tr_set.decay_gap * decay_share

                                except:
                                    pass

                                if res_df['high'].iloc[j] >= long_tp_.iloc[tp_j] and partial_tp_cnt == l_i:
                                    # if res_df['high'].iloc[j] >= long_tp.iloc[j]:

                                    if l_i == len(long_tps) - 1:
                                        trade_done = 1

                                    partial_tp_cnt += 1

                                    #         dynamic tp        #
                                    # if 0:
                                    if long_tp_.iloc[j] != long_tp_.iloc[j - 1] and not config.tp_set.static_tp:

                                        #   tp limit 이 불가한 경우 = open 이 이미, tp 를 넘은 경우 #
                                        # if res_df['open'].iloc[j] >= long_tp_.iloc[initial_i]:
                                        if res_df['open'].iloc[j] >= long_tp_.iloc[j]:

                                            # tp = long_tp_.iloc[initial_i]
                                            tp = res_df['open'].iloc[j]

                                            if trade_done:
                                                tp_state_list.append("d-long_open {}".format(strat_version))


                                        #   tp limit 이 가능한 경우 = open 이 아직, tp 를 넘지 않은 경우 #
                                        else:

                                            # tp = long_tp_.iloc[initial_i]
                                            tp = long_tp_.iloc[j]
                                            # tp = res_df['open'].iloc[j]

                                            if trade_done:
                                                tp_state_list.append("d-long_tp {}".format(strat_version))

                                    #         static tp         #
                                    else:

                                        #   tp limit 이 불가한 경우 = open 이 이미, tp 를 넘은 경우 #
                                        #   non_inversion 의 경우, short_tp 가 가능함   #

                                        if res_df['open'].iloc[j] >= long_tp_.iloc[tp_j]:
                                            # if res_df['open'].iloc[j] >= long_tp_.iloc[initial_i]:

                                            # tp = long_tp_.iloc[initial_i]

                                            if config.tr_set.decay_gap != "None" and decay_remain == 0:
                                                tp = res_df['open'].iloc[j]
                                            else:
                                                tp = long_tp_.iloc[tp_j]

                                            if trade_done:
                                                tp_state_list.append("s-long_tp {}".format(strat_version))


                                        #   tp limit 이 가능한 경우 = open 이 아직, tp 를 넘지 않은 경우 #
                                        else:

                                            # tp = long_tp_.iloc[initial_i]
                                            tp = long_tp_.iloc[tp_j]

                                            # tp = res_df['open'].iloc[j]

                                            if trade_done:
                                                tp_state_list.append("s-long_tp {}".format(strat_version))

                                    tp_list.append(tp)
                                    tp_idx_list.append(j)
                                    fee += config.trader_set.limit_fee

                        #           2. by time        #
                        if config.tp_set.tp_type == 'MARKET' or (config.tp_set.tp_type == "BOTH" and not trade_done):

                            market_tp = 0

                            # -------------- sar tp -------------- #
                            # if (res_df['low'].iloc[j] <= res_df['sar2'].iloc[j]) & \
                            #   (res_df['low'].iloc[j - 1] > res_df['sar2'].iloc[j - 1]) & \
                            #   (res_df['low'].iloc[j - 2] > res_df['sar2'].iloc[j - 2]):

                            #       inversion     #
                            # if (res_df['high'].iloc[j] >= res_df['sar2'].iloc[j]) & \
                            #   (res_df['high'].iloc[j - 1] < res_df['sar2'].iloc[j - 1]) & \
                            #   (res_df['high'].iloc[j - 2] < res_df['sar2'].iloc[j - 2]):

                            # ----------- st long ----------- #
                            # if res_df['close'].iloc[j] >= res_df['long_tp'].iloc[tp_j]:

                            # -------------- sar pb tp -------------- #
                            # if res_df['high'].iloc[j] >= res_df['long_tp'].iloc[initial_i]:

                            # -------------- st tp -------------- #
                            # if res_df['close'].iloc[j] < res_df['middle_line'].iloc[j]:

                            # -------------- fisher tp -------------- #
                            # if entry[j] == -1:

                            # -------------- timestamp -------------- #
                            if config.tp_set.time_tp:
                                if np_timeidx[
                                    j] % config.loc_set.zone.c_itv_ticks == config.loc_set.zone.c_itv_ticks - 1 and \
                                        j - initial_i >= config.loc_set.zone.c_itv_ticks:
                                    market_tp = 1

                            # -------------- rsi -------------- #
                            if strat_version in rsi_out_stratver:
                                if (res_df['rsi_%s' % config.loc_set.point.exp_itv].iloc[
                                        j - 1] <= 50 + config.loc_set.point.osc_band) & \
                                        (res_df['rsi_%s' % config.loc_set.point.exp_itv].iloc[
                                             j] > 50 + config.loc_set.point.osc_band):
                                    # if (res_df['rsi_%s' % config.loc_set.point.exp_itv].iloc[j - 1] >= 50 + config.loc_set.point.osc_band) & \
                                    #                  (res_df['rsi_%s' % config.loc_set.point.exp_itv].iloc[j] < 50 + config.loc_set.point.osc_band):
                                    market_tp = 1

                                # -------------- cci -------------- #
                                # if (res_df['cci_%s' % config.loc_set.point.exp_itv].iloc[j - 1] <= config.loc_set.point.osc_band) & \
                                #                  (res_df['cci_%s' % config.loc_set.point.exp_itv].iloc[j] > config.loc_set.point.osc_band):
                                #   market_tp = 1

                            # ---------------------------- early out phase ---------------------------- #

                            #        osc slight touch     #
                            if allow_osc_touch:
                                if (np.max(res_df['rsi_%s' % config.loc_set.point.exp_itv].iloc[
                                           initial_i:j]) > 50 + config.loc_set.point.osc_band - rsi_gap) & \
                                        (res_df['rsi_%s' % config.loc_set.point.exp_itv].iloc[j] <= 50):
                                    market_tp = 1

                            #         tp early out        #
                            # # if (np.max(res_df['high'].iloc[e_j:j]) > res_df['long_tp'].iloc[tp_j]) & \
                            # if (np.max(res_df['high'].iloc[e_j:j]) > res_df['h_long_rtc_1'].iloc[tp_j] + res_df['h_long_rtc_gap'].iloc[tp_j] * early_out_tpg) & \
                            #   (res_df['close'].iloc[j] <= res_df['long_ep'].iloc[ep_j]):
                            #   market_tp = 1

                            # if strat_version == "v7":
                            #   if res_df['dc_lower_1m'].iloc[j] < res_df['dc_lower_5m'].iloc[j]:
                            #     market_tp = 1

                            #         bb_upper early out        #
                            if strat_version in ['v5_2']:
                                if res_df['close'].iloc[j] > res_df['bb_upper_5m'].iloc[j] > res_df['close'].iloc[
                                    j - 1]:
                                    cross_on = 1

                                if cross_on == 1 and res_df['close'].iloc[j] < res_df['bb_lower_5m'].iloc[j] < \
                                        res_df['close'].iloc[j - 1]:
                                    market_tp = 1

                            if market_tp:

                                tp = res_df['close'].iloc[j]
                                # tp = res_df['open'].iloc[j]
                                trade_done = 1

                                if trade_done:
                                    tp_state_list.append("long close tp {}".format(strat_version))
                                    # print("early_out passed !")

                                tp_list.append(tp)
                                tp_idx_list.append(j)
                                fee += config.trader_set.market_fee

                    # -------------- out -------------- #
                    if not trade_done and config.out_set.use_out and j != len(res_df) - 1:

                        # -------------- macd -------------- #
                        # if res_df['macd_hist3'].iloc[j] < 0:
                        # # if res_df['macd_hist3'].iloc[i] > 0 and res_df['macd_hist3'].iloc[j] < 0:

                        # -------------- st config.out_set.retouch -------------- #
                        # out = 1 상태면 동일 tick 에서 config.out_set.retouch 를 조사할 거기 때문에, 먼저 검사함
                        # 그리고, out 기준이 close 라 이게 맞음
                        # close 가 long_out 보다 내려가있는 상태일테니 high 를 조사하는게 맞음
                        # if out and res_df['high'].iloc[j] >= long_out.iloc[out_j]:
                        #   config.out_set.retouch = 1

                        # ------- 일정시간 이상, dynamic_out 적용 ------ #
                        try:
                            if j - out_idx >= config.out_set.retouch_out_period:
                                static_long_out = res_df['long_out_{}'.format(strat_version)].iloc[j]

                        except Exception as e:
                            pass

                            # ------- static out ------ #
                        try:
                            if out and res_df['high'].iloc[j] >= static_long_out:
                                config.out_set.retouch = 1
                        except Exception as e:
                            pass

                            # ------- config.out_set.retouch out ------ #
                        # if out and res_df['high'].iloc[j] >= long_out2.iloc[out_j]:
                        #   config.out_set.retouch = 1

                        # -------------- st -------------- #
                        # if res_df['close'].iloc[j] < res_df['middle_line'].iloc[j]:
                        # if res_df['close'].iloc[j] < res_df['minor_ST3_Down'].iloc[j]:
                        # if res_df['close'].iloc[j] < lower_middle.iloc[j]:
                        # if res_df['close'].iloc[j] < res_df['minor_ST1_Down'].iloc[j]:
                        if out == 0:
                            if config.out_set.hl_out:
                                if res_df['low'].iloc[j] <= res_df['long_out_{}'.format(strat_version)].iloc[
                                    out_j]:  # check out only once
                                    out = 1

                            else:
                                if res_df['close'].iloc[j] <= res_df['long_out_{}'.format(strat_version)].iloc[
                                    out_j]:  # check out only once
                                    out = 1

                            # out_idx = j
                            # static_long_out = long_out.iloc[out_j]
                            # if config.out_set.second_out:
                            # static_long_out = long_out.iloc[out_j] - res_df['st_gap'].iloc[out_j] * config.out_set.second_out_gap

                        # if out == 0 and res_df['low'].iloc[j] <= long_out.iloc[out_j]: # check out only once
                        #   out = 1

                        # -------------- sma -------------- #
                        # if res_df['close'].iloc[j] < res_df[sma].iloc[j]:

                        # -------------- sar -------------- #
                        # if res_df['close'].iloc[j] < res_df['minor_ST3_Down'].iloc[j] \
                        #   or res_df['sar2'].iloc[j] >= res_df['low'].iloc[j]:
                        # if res_df['close'].iloc[j] < long_out.iloc[initial_i]: # or \
                        #   #  res_df['close'].iloc[j] < res_df['sar2'].iloc[j]:
                        #   #  res_df['sar2_uptrend'].iloc[j] == 0 or \
                        #   out = 1

                        # if res_df['sar2_uptrend'].iloc[j] == 0:

                        #     if prev_sar is None:
                        #       prev_sar = res_df['sar2'].iloc[j - 1]

                        #     if res_df['close'].iloc[j] < prev_sar:
                        #       out = 1

                        # else:
                        #   if res_df['close'].iloc[j] < res_df['sar2'].iloc[j]:
                        #     out = 1

                        # -------------- hl -------------- #
                        # if res_df['close'].iloc[j] < long_out.iloc[tp_j]:

                        # -------------- stoch -------------- #
                        # if res_df['stoch'].iloc[j - 2] <= res_df['stoch'].iloc[j - 1] and \
                        #   res_df['stoch'].iloc[j - 1] > res_df['stoch'].iloc[j] and \
                        #   res_df['stoch'].iloc[j - 1] >= stoch_upper:
                        #   out = 1

                        # config.out_set.retouch 1 경우, config.out_set.retouch 조건도 있어야함
                        if out:
                            if config.out_set.retouch:
                                if config.out_set.retouch:
                                    pass
                                else:
                                    continue

                            else:
                                pass

                            if config.out_set.price_restoration:
                                tp = res_df['long_out_{}'.format(strat_version)].iloc[out_j]
                                if config.out_set.second_out:
                                    tp = res_df['long_out2_{}'.format(strat_version)].iloc[out_j]

                                # if res_df['close'].iloc[j] < tp: # 이 경우를 protect 하는건 insane 임
                                # # if res_df['high'].iloc[j] < tp: # --> config.out_set.hl_out 사용시 이 조건은 valid 함
                                #   tp = res_df['close'].iloc[j]

                            else:

                                if res_df['open'].iloc[j] <= res_df['long_out_{}'.format(strat_version)].iloc[out_j]:
                                    tp = res_df['open'].iloc[j]
                                else:
                                    if config.out_set.hl_out:
                                        tp = res_df['long_out_{}'.format(strat_version)].iloc[out_j]
                                    else:
                                        tp = res_df['close'].iloc[j]

                                # if not config.out_set.static_out:
                                #   if res_df['open'].iloc[j] <= res_df['long_out'].iloc[out_j]: # dynamic close out 의 open 고려
                                #     tp = res_df['open'].iloc[j]
                                #   else:
                                #     tp = res_df['close'].iloc[j]

                                # else:
                                #   tp = res_df['close'].iloc[j]

                            if config.out_set.retouch:  # out 과 open 비교
                                if config.out_set.second_out:  # long_out = sell
                                    # config.out_set.second_out 은 기본적으로 limit 이라 이 구조가 가능함
                                    if res_df['open'].iloc[j] >= res_df['long_out2_{}'.format(strat_version)].iloc[
                                        out_j]:  # dynamic_out 일 경우 고려해야함
                                        tp = res_df['open'].iloc[j]
                                else:
                                    if res_df['open'].iloc[j] >= res_df['long_out_{}'.format(strat_version)].iloc[
                                        out_j]:  # dynamic_out 일 경우 고려해야함
                                        tp = res_df['open'].iloc[j]

                                try:
                                    if res_df['open'].iloc[j] >= static_long_out:
                                        tp = res_df['open'].iloc[j]
                                    else:
                                        tp = static_long_out
                                except Exception as e:
                                    pass

                            # tp = res_df['open'].iloc[j]
                            tp_state_list.append("long close_out {}".format(strat_version))
                            trade_done = 1

                            tp_list.append(tp)
                            tp_idx_list.append(j)
                            fee += config.trader_set.market_fee

                    # -------------- non tp -------------- #
                    if j == len(res_df) - 1:
                        trade_done = 1
                        tp = res_df['close'].iloc[j]
                        tp_list.append(tp)
                        tp_idx_list.append(j)
                        fee += config.trader_set.market_fee

                    if trade_done:

                        # --------------- tp_ratio info --------------- #
                        try:
                            if config.out_set.use_out:
                                done_tp = res_df['long_tp_{}'.format(strat_version)].iloc[tp_j]
                                done_out = res_df['long_out_{}'.format(strat_version)].iloc[out_j]

                                if done_out >= ep_list[0]:  # loss >= 1
                                    tp_ratio = np.nan
                                    dr = np.nan
                                    # print("loss >= 1")
                                else:
                                    tp_ratio = ((done_tp - ep_list[0] - tp_fee * ep_list[0]) / (
                                                ep_list[0] - done_out + out_fee * ep_list[0]))
                                    dr = ((done_tp - ep_list[0]) / (ep_list[0] - done_out))

                            else:
                                dr = np.nan
                                tp_ratio = np.nan

                        except Exception as e:
                            print("error in tr phase :", e)
                            dr = np.nan
                            tp_ratio = np.nan

                        tp_ratio_list.append(tp_ratio)
                        long_tp_ratio_list.append(tp_ratio)
                        dr_list.append(dr)
                        long_dr_list.append(dr)

                        qty_list = []
                        temp_pr_list = []
                        r_qty = 1
                        for q_i in range(len(tp_list) - 1, -1, -1):

                            if len(tp_list) == 1:
                                temp_qty = r_qty
                            else:
                                if q_i != 0:
                                    temp_qty = r_qty / config.tp_set.partial_qty_divider
                                else:
                                    temp_qty = r_qty

                            # temp_pr = (tp_list[q_i] / ep_list[0] - fee - 1) * temp_qty
                            temp_pr = (tp_list[q_i] / ep_list[0] - fee - 1) * temp_qty * config.lvrg_set.leverage
                            r_qty -= temp_qty

                            temp_pr_list.append(temp_pr)

                        temp_pr = sum(temp_pr_list) + 1

                        # -------------------- sub ep -> pr calc -------------------- #
                        if len(ep_list) > 1:

                            p_ep_pr = []
                            for sub_ep_ in ep_list:
                                sub_pr = (tp / sub_ep_ - fee - 1) * config.lvrg_set.leverage
                                p_ep_pr.append(sub_pr)

                            temp_pr = sum(p_ep_pr) + 1

                            print("temp_pr :", temp_pr)

                        # ------------ hedge + non_hedge pr summation ------------ #
                        #         hedge pr direction is opposite to the origin       #
                        hedge_pr = 1
                        if hedge_cnt == 0:
                            #       hedge tp      #
                            h_tp = res_df['close'].iloc[j]
                            hedge_pr = (h_ep / h_tp - fee - 1) * config.lvrg_set.leverage  # hedge short
                            temp_pr += hedge_pr
                            h_j = j

                        # ll = min(res_df['low'].iloc[i:j + 1])
                        ll = min(res_df['low'].iloc[i:j])  # pos. 정리하기 바로 직전까지
                        long_liq = (ll / ep_list[0] - fee - 1) * config.lvrg_set.leverage + 1

                        if j != len(res_df) - 1:

                            ep_tp_list.append((ep_list, tp_list))
                            # trade_list.append((ep_idx_list, tp_idx_list))
                            trade_list.append((ep_idx_list, out_idx_list, tp_idx_list))

                            liqd_list.append(long_liq)
                            long_liqd_list.append(long_liq)

                            h_ep_tp_list.append((h_ep, h_tp))
                            h_trade_list.append([initial_i, h_i, h_j])

                            pr_list.append(temp_pr)
                            fee_list.append(fee)
                            long_list.append(temp_pr)
                            long_fee_list.append(fee)

                            h_pr_list.append(hedge_pr)
                            h_long_list.append(hedge_pr)

                            i = j
                            break

                        else:

                            # ep_tp_list.append((ep_list, tp_list))
                            # trade_list.append((ep_idx_list, tp_idx_list))

                            # pr_list 를 넣지 않을거니까, open_list 에서 해당 idx 는 pop
                            open_list.pop()
                            zone_list.pop()
                            side_list.pop()
                            strat_ver_list.pop()

                            #         tp 미체결 survey        #
                            nontp_liqd_list.append(long_liq)
                            nontp_long_liqd_list.append(long_liq)
                            nontp_long_indexs.append(i)
                            nontp_long_ep_list.append(ep_list[0])

                            nontp_long_pr = (tp / ep_list[0] - fee - 1) * config.lvrg_set.leverage + 1
                            nontp_pr_list.append(nontp_long_pr)
                            nontp_long_pr_list.append(nontp_long_pr)

                        if len(open_list) > len(trade_list):
                            print('debug from index :', i)
                            print(len(open_list), len(trade_list))
                            print("len(res_df) :", len(res_df))
                            assert len(open_list) == len(trade_list), 'stopped'

            i += 1  # if entry starts with prev trade's close, do not use it !
            # print("i in end :", i)
            if i >= len(res_df):
                break

        # -------------------- result analysis -------------------- #
        # try:
        print("elapsed_time :", time.time() - start_0)

        plt.style.use('default')
        # mpl.rcParams.update(mpl.rcParamsDefault)

        fig = plt.figure(figsize=(14, 10))

        gs = gridspec.GridSpec(nrows=3,  # row 몇 개
                               ncols=3,  # col 몇 개
                               height_ratios=[10, 10, 1]
                               )
        # plt.figure(figsize=(16, 12))
        # plt.figure(figsize=(12, 8))
        # plt.figure(figsize=(10, 6))
        plt.suptitle(key)

        try:
            np_pr = np.array(pr_list)

            sr = sharpe_ratio(np_pr)

            dpf = (len(res_df) / 1440) / len(np_pr)

            np_zone_list = np.array(zone_list)
            # np_pr_list = np.array(pr_list)
            np_side_list = np.array(side_list)

            t_w = np.sum(np.where((np_zone_list == 't') & (np_pr > 1), 1, 0))
            c_w = np.sum(np.where((np_zone_list == 'c') & (np_pr > 1), 1, 0))
            t_ls = np.sum(np.where((np_zone_list == 't') & (np_pr < 1), 1, 0))
            c_ls = np.sum(np.where((np_zone_list == 'c') & (np_pr < 1), 1, 0))

            # np_pr = (np.array(pr_list) - 1) * config.lvrg_set.leverage + 1

            # ----- fake_pr ----- #
            # np_pr = np.where(np_pr > 1, 1 + (np_pr - 1) * 3, np_pr)

            total_pr = np.cumprod(np_pr)

            for_sum_pr = np_pr - 1
            for_sum_pr[0] = 1
            sum_pr = np.cumsum(for_sum_pr)
            sum_pr = np.where(sum_pr < 0, 0, sum_pr)

            wr = len(np_pr[np_pr > 1]) / len(np_pr[np_pr != 1])

            total_rollmax_pr = np.maximum.accumulate(total_pr)
            total_acc_mdd = np.max((total_rollmax_pr - total_pr) / total_rollmax_pr)
            total_rollmax_sumpr = np.maximum.accumulate(sum_pr)
            total_sum_mdd = np.max((total_rollmax_sumpr - sum_pr) / total_rollmax_sumpr)

            np_tp_ratio_list = np.array(tp_ratio_list)  # 초기에 tr 을 정하는거라 mean 사용하는게 맞음
            mean_tr = np.mean(np_tp_ratio_list[np.isnan(np_tp_ratio_list) == 0])

            np_dr_list = np.array(dr_list)  # 초기에 tr 을 정하는거라 mean 사용하는게 맞음
            mean_dr = np.mean(np_dr_list[np.isnan(np_dr_list) == 0])

            # pr_gap = (np_pr - 1) / config.lvrg_set.leverage + fee
            # tp_gap_ = pr_gap[pr_gap > 0]
            # # mean_config.tr_set.tp_gap = np.mean(pr_gap[pr_gap > 0])
            # mean_ls_gap = np.mean(pr_gap[pr_gap < 0])

            # ---- profit fee ratio ---- #
            # mean_pgfr = np.mean((tp_gap_ - fee) / abs(tp_gap_ + fee))

            plt.subplot(gs[0])
            plt.plot(total_pr)
            plt.plot(sum_pr, color='gold')
            if len(nontp_liqd_list) != 0:
                plt.title(
                    "wr : %.3f\n len(td) : %s\n dpf : %.3f\n min_pr : %.3f\n acc_pr : %.3f\n sum_pr : %.3f\n sr : %.3f"
                    % (wr, len(np_pr[np_pr != 1]), dpf, np.min(np_pr), total_pr[-1], sum_pr[-1], sr) + \
                    "\n acc_mdd : -%.3f\n sum_mdd : -%.3f\n leverage %s\n liqd : %.3f\n mean_tr : %.3f\n mean_dr : %.3f"
                    % (total_acc_mdd, total_sum_mdd, config.lvrg_set.leverage, min(liqd_list), mean_tr, mean_dr) + \
                    "\n nontp_liqd_cnt : %s\nnontp_liqd : %.3f\nontp_liqd_pr : %.3f\n tw cw tls cls : %s %s %s %s"
                    % (len(nontp_liqd_list), min(nontp_liqd_list), min(nontp_pr_list), t_w, c_w, t_ls, c_ls),
                    position=title_position, fontsize=fontsize)
            else:
                plt.title(
                    "wr : %.3f\n len(td) : %s\n dpf : %.3f\n min_pr : %.3f\n acc_pr : %.3f\n sum_pr : %.3f\n sr : %.3f"
                    % (wr, len(np_pr[np_pr != 1]), dpf, np.min(np_pr), total_pr[-1], sum_pr[-1], sr) + \
                    "\n acc_mdd : -%.3f\n sum_mdd : -%.3f\n leverage %s\n liqd : %.3f\n mean_tr : %.3f\n mean_dr : %.3f"
                    % (total_acc_mdd, total_sum_mdd, config.lvrg_set.leverage, min(liqd_list), mean_tr, mean_dr) + \
                    "\n nontp_liqd_cnt : %s\n tw cw tls cls : %s %s %s %s"
                    % (len(nontp_liqd_list), t_w, c_w, t_ls, c_ls),
                    position=title_position, fontsize=fontsize)
            # plt.show()

            survey_df.iloc[survey_i] = wr, sr, len(np_pr[np_pr != 1]), dpf, \
                                       total_pr[-1], sum_pr[-1], total_acc_mdd, total_sum_mdd, min(liqd_list), np.min(
                np_pr), mean_tr, mean_dr

            print('supblot231 passed')

        except Exception as e:
            print("error in 231 :", e)

        try:
            #         short only      #
            short_np_pr = np.array(short_list)

            short_sr = sharpe_ratio(short_np_pr)

            short_dpf = (len(res_df) / 1440) / len(short_np_pr)

            short_total_pr = np.cumprod(short_np_pr)

            short_for_sum_pr = short_np_pr - 1
            short_for_sum_pr[0] = 1
            short_sum_pr = np.cumsum(short_for_sum_pr)
            short_sum_pr = np.where(short_sum_pr < 0, 0, short_sum_pr)

            short_wr = len(short_np_pr[short_np_pr > 1]) / len(short_np_pr[short_np_pr != 1])

            t_w_s = np.sum(np.where((np_zone_list == 't') & (np_pr > 1) & (np_side_list == 's'), 1, 0))
            c_w_s = np.sum(np.where((np_zone_list == 'c') & (np_pr > 1) & (np_side_list == 's'), 1, 0))
            t_ls_s = np.sum(np.where((np_zone_list == 't') & (np_pr < 1) & (np_side_list == 's'), 1, 0))
            c_ls_s = np.sum(np.where((np_zone_list == 'c') & (np_pr < 1) & (np_side_list == 's'), 1, 0))

            short_rollmax_pr = np.maximum.accumulate(short_total_pr)
            short_acc_mdd = np.max((short_rollmax_pr - short_total_pr) / short_rollmax_pr)
            short_rollmax_sumpr = np.maximum.accumulate(short_sum_pr)
            short_sum_mdd = np.max((short_rollmax_sumpr - short_sum_pr) / short_rollmax_sumpr)

            np_short_tp_ratio_list = np.array(short_tp_ratio_list)
            mean_short_tr = np.mean(np_short_tp_ratio_list[np.isnan(np_short_tp_ratio_list) == 0])

            np_short_dr_list = np.array(short_dr_list)
            mean_short_dr = np.mean(np_short_dr_list[np.isnan(np_short_dr_list) == 0])

            # short_pr_gap = (short_np_pr - 1) / config.lvrg_set.leverage + fee
            # short_tp_gap = short_pr_gap[short_pr_gap > 0]
            # # mean_short_tp_gap = np.mean(short_pr_gap[short_pr_gap > 0])
            # # mean_short_ls_gap = np.mean(short_pr_gap[short_pr_gap < 0])

            # mean_short_pgfr = np.mean((short_tp_gap - fee) / abs(short_tp_gap + fee))

            # plt.subplot(232)
            plt.subplot(gs[1])
            plt.plot(short_total_pr)
            plt.plot(short_sum_pr, color='gold')
            if len(nontp_short_liqd_list) != 0:

                max_nontp_short_term = len(res_df) - nontp_short_indexs[0]

                plt.title(
                    "wr : %.3f\nlen(short_td) : %s\n dpf : %.3f\n min_pr : %.3f\n acc_pr : %.3f\n sum_pr : %.3f\n sr : %.3f"
                    % (short_wr, len(short_np_pr[short_np_pr != 1]), short_dpf, np.min(short_np_pr), short_total_pr[-1],
                       short_sum_pr[-1], short_sr) + \
                    "\n acc_mdd : -%.3f\n sum_mdd : -%.3f\n leverage %s\n liqd : %.3f\n mean_tr : %.3f\n mean_dr : %.3f"
                    % (short_acc_mdd, short_sum_mdd, config.lvrg_set.leverage, min(short_liqd_list), mean_short_tr,
                       mean_short_dr) + \
                    "\n nontp_liqd_cnt : %s\n nontp_liqd : %.3f\n nontp_liqd_pr : %.3f\n max_nontp_term : %s\n tw cw tls cls : %s %s %s %s"
                    % (len(nontp_short_liqd_list), min(nontp_short_liqd_list), min(nontp_short_pr_list),
                       max_nontp_short_term, t_w_s, c_w_s, t_ls_s, c_ls_s),
                    position=title_position, fontsize=fontsize)
            else:
                plt.title(
                    "wr : %.3f\nlen(short_td) : %s\n dpf : %.3f\n min_pr : %.3f\n acc_pr : %.3f\n sum_pr : %.3f\n sr : %.3f"
                    % (short_wr, len(short_np_pr[short_np_pr != 1]), short_dpf, np.min(short_np_pr), short_total_pr[-1],
                       short_sum_pr[-1], short_sr) + \
                    "\n acc_mdd : -%.3f\n sum_mdd : -%.3f\n leverage %s\n liqd : %.3f\n mean_tr : %.3f\n mean_dr : %.3f"
                    % (short_acc_mdd, short_sum_mdd, config.lvrg_set.leverage, min(short_liqd_list), mean_short_tr,
                       mean_short_dr) + \
                    "\n nontp_liqd_cnt : %s\n tw cw tls cls : %s %s %s %s"
                    % (len(nontp_short_liqd_list), t_w_s, c_w_s, t_ls_s, c_ls_s),
                    position=title_position, fontsize=fontsize)

            short_survey_df.iloc[survey_i] = short_wr, short_sr, len(short_np_pr[short_np_pr != 1]), short_dpf, \
                                             short_total_pr[-1], short_sum_pr[-1], short_acc_mdd, short_sum_mdd, min(
                short_liqd_list), np.min(short_np_pr), mean_short_tr, mean_short_dr

            print('supblot232 passed')

        except Exception as e:
            print("error in 232 :", e)

        try:
            #         long only      #
            long_np_pr = np.array(long_list)
            # long_np_pr = (np.array(long_list) - 1) * config.lvrg_set.leverage + 1

            long_sr = sharpe_ratio(long_np_pr)

            long_dpf = (len(res_df) / 1440) / len(long_np_pr)

            long_total_pr = np.cumprod(long_np_pr)

            long_for_sum_pr = long_np_pr - 1
            long_for_sum_pr[0] = 1
            long_sum_pr = np.cumsum(long_for_sum_pr)
            long_sum_pr = np.where(long_sum_pr < 0, 0, long_sum_pr)

            long_wr = len(long_np_pr[long_np_pr > 1]) / len(long_np_pr[long_np_pr != 1])

            t_w_l = np.sum(np.where((np_zone_list == 't') & (np_pr > 1) & (np_side_list == 'l'), 1, 0))
            c_w_l = np.sum(np.where((np_zone_list == 'c') & (np_pr > 1) & (np_side_list == 'l'), 1, 0))
            t_ls_l = np.sum(np.where((np_zone_list == 't') & (np_pr < 1) & (np_side_list == 'l'), 1, 0))
            c_ls_l = np.sum(np.where((np_zone_list == 'c') & (np_pr < 1) & (np_side_list == 'l'), 1, 0))

            long_rollmax_pr = np.maximum.accumulate(long_total_pr)
            long_acc_mdd = np.max((long_rollmax_pr - long_total_pr) / long_rollmax_pr)
            long_rollmax_sumpr = np.maximum.accumulate(long_sum_pr)
            long_sum_mdd = np.max((long_rollmax_sumpr - long_sum_pr) / long_rollmax_sumpr)

            np_long_tp_ratio_list = np.array(long_tp_ratio_list)
            mean_long_tr = np.mean(np_long_tp_ratio_list[np.isnan(np_long_tp_ratio_list) == 0])

            np_long_dr_list = np.array(long_dr_list)
            mean_long_dr = np.mean(np_long_dr_list[np.isnan(np_long_dr_list) == 0])

            # long_pr_gap = (long_np_pr - 1) / config.lvrg_set.leverage + fee
            # long_tp_gap = long_pr_gap[long_pr_gap > 0]
            # # mean_long_tp_gap = np.mean(long_pr_gap[long_pr_gap > 0])
            # # mean_long_ls_gap = np.mean(long_pr_gap[long_pr_gap < 0])

            # mean_long_pgfr = np.mean((long_tp_gap - fee) / abs(long_tp_gap + fee))

            plt.subplot(gs[2])
            plt.plot(long_total_pr)
            plt.plot(long_sum_pr, color='gold')
            if len(nontp_long_liqd_list) != 0:

                max_nontp_long_term = len(res_df) - nontp_long_indexs[0]

                plt.title(
                    "wr : %.3f\nlen(long_td) : %s\n dpf : %.3f\n min_pr : %.3f\n acc_pr : %.3f\n sum_pr : %.3f\n sr : %.3f"
                    % (long_wr, len(long_np_pr[long_np_pr != 1]), long_dpf, np.min(long_np_pr), long_total_pr[-1],
                       long_sum_pr[-1], long_sr) + \
                    "\n acc_mdd : -%.3f\n sum_mdd : -%.3f\n leverage %s\n liqd : %.3f\n mean_tr : %.3f\n mean_dr : %.3f"
                    % (long_acc_mdd, long_sum_mdd, config.lvrg_set.leverage, min(long_liqd_list), mean_long_tr,
                       mean_long_dr) + \
                    "\n nontp_liqd_cnt : %s\n nontp_liqd : %.3f\n nontp_liqd_pr : %.3f\n max_nontp_term : %s\n tw cw tls cls : %s %s %s %s"
                    % (
                    len(nontp_long_liqd_list), min(nontp_long_liqd_list), min(nontp_long_pr_list), max_nontp_long_term,
                    t_w_l, c_w_l, t_ls_l, c_ls_l),
                    position=title_position, fontsize=fontsize)
            else:
                plt.title(
                    "wr : %.3f\nlen(long_td) : %s\n dpf : %.3f\n min_pr : %.3f\n acc_pr : %.3f\n sum_pr : %.3f\n sr : %.3f"
                    % (long_wr, len(long_np_pr[long_np_pr != 1]), long_dpf, np.min(long_np_pr), long_total_pr[-1],
                       long_sum_pr[-1], long_sr) + \
                    "\n acc_mdd : -%.3f\n sum_mdd : -%.3f\n leverage %s\n liqd : %.3f\n mean_tr : %.3f\n mean_dr : %.3f"
                    % (long_acc_mdd, long_sum_mdd, config.lvrg_set.leverage, min(long_liqd_list), mean_long_tr,
                       mean_long_dr) + \
                    "\n nontp_liqd_cnt : %s\n tw cw tls cls : %s %s %s %s"
                    % (len(nontp_long_liqd_list), t_w_l, c_w_l, t_ls_l, c_ls_l),
                    position=title_position, fontsize=fontsize)

            long_survey_df.iloc[survey_i] = long_wr, long_sr, len(long_np_pr[long_np_pr != 1]), long_dpf, \
                                            long_total_pr[-1], long_sum_pr[-1], long_acc_mdd, long_sum_mdd, min(
                long_liqd_list), np.min(long_np_pr), mean_long_tr, mean_long_dr

            print('supblot233 passed')

        except Exception as e:
            print("error in 233 :", e)

        try:
            #     reversion adjustment      #
            # rev_np_pr = 1 / (np.array(pr_list) + fee) - fee
            rev_fee = tp_fee + out_fee - np.array(fee_list)
            rev_np_pr = (1 / ((np.array(pr_list) - 1) / config.lvrg_set.leverage + np.array(
                fee_list) + 1) - rev_fee - 1) * config.lvrg_set.leverage + 1
            # rev_np_pr = (1 / (np.array(pr_list) + fee) - fee - 1) * config.lvrg_set.leverage + 1

            rev_sr = sharpe_ratio(rev_np_pr)

            rev_total_pr = np.cumprod(rev_np_pr)
            rev_wr = len(rev_np_pr[rev_np_pr > 1]) / len(rev_np_pr[rev_np_pr != 1])

            rev_total_for_sum_pr = rev_np_pr - 1
            rev_total_for_sum_pr[0] = 1
            rev_total_sum_pr = np.cumsum(rev_total_for_sum_pr)
            rev_total_sum_pr = np.where(rev_total_sum_pr < 0, 0, rev_total_sum_pr)

            rev_rollmax_pr = np.maximum.accumulate(rev_total_pr)
            rev_acc_mdd = np.max((rev_rollmax_pr - rev_total_pr) / rev_rollmax_pr)
            rev_rollmax_sumpr = np.maximum.accumulate(rev_total_sum_pr)
            rev_sum_mdd = np.max((rev_rollmax_sumpr - rev_total_sum_pr) / rev_rollmax_sumpr)

            plt.subplot(gs[3])

            plt.plot(rev_total_pr)
            plt.plot(rev_total_sum_pr, color='gold')

            plt.title(
                "wr : %.3f\n sr : %.3f\n acc_pr : %.3f\n sum_pr : %.3f\n acc_mdd : -%.3f\n sum_mdd : -%.3f\n min_pr : %.3f"
                % (rev_wr, rev_sr, rev_total_pr[-1], rev_total_sum_pr[-1],
                   rev_acc_mdd, rev_sum_mdd, np.min(rev_np_pr)), fontsize=fontsize)

            rev_survey_df.iloc[survey_i] = rev_wr, rev_sr, rev_total_pr[-1], rev_total_sum_pr[
                -1], rev_acc_mdd, rev_sum_mdd, np.min(rev_np_pr)

        except Exception as e:
            print("error in 234 :", e)

        try:
            #         short       #
            # rev_short_np_pr = 1 / (np.array(short_list) + fee) - fee
            rev_short_fee = tp_fee + out_fee - np.array(short_fee_list)
            rev_short_np_pr = (1 / ((np.array(short_list) - 1) / config.lvrg_set.leverage + np.array(
                short_fee_list) + 1) - rev_short_fee - 1) * config.lvrg_set.leverage + 1
            # rev_short_np_pr = (1 / (np.array(short_list) + fee) - fee - 1) * config.lvrg_set.leverage + 1

            rev_short_sr = sharpe_ratio(rev_short_np_pr)

            short_rev_total_pr = np.cumprod(rev_short_np_pr)
            rev_short_wr = len(rev_short_np_pr[rev_short_np_pr > 1]) / len(rev_short_np_pr[rev_short_np_pr != 1])

            rev_short_for_sum_pr = rev_short_np_pr - 1
            rev_short_for_sum_pr[0] = 1
            short_rev_sum_pr = np.cumsum(rev_short_for_sum_pr)
            short_rev_sum_pr = np.where(short_rev_sum_pr < 0, 0, short_rev_sum_pr)

            short_rev_rollmax_pr = np.maximum.accumulate(short_rev_total_pr)
            short_rev_acc_mdd = np.max((short_rev_rollmax_pr - short_rev_total_pr) / short_rev_rollmax_pr)
            short_rev_rollmax_sumpr = np.maximum.accumulate(short_rev_sum_pr)
            short_rev_sum_mdd = np.max((short_rev_rollmax_sumpr - short_rev_sum_pr) / short_rev_rollmax_sumpr)

            plt.subplot(gs[4])

            plt.plot(short_rev_total_pr)
            plt.plot(short_rev_sum_pr, color='gold')

            plt.title(
                "wr : %.3f\n sr : %.3f\n acc_pr : %.3f\n sum_pr : %.3f\n acc_mdd : -%.3f\n sum_mdd : -%.3f\n min_pr : %.3f"
                % (rev_short_wr, rev_short_sr, short_rev_total_pr[-1], short_rev_sum_pr[-1],
                   short_rev_acc_mdd, short_rev_sum_mdd, np.min(rev_short_np_pr)), fontsize=fontsize)

            rev_short_survey_df.iloc[survey_i] = rev_short_wr, rev_short_sr, short_rev_total_pr[-1], short_rev_sum_pr[
                -1], short_rev_acc_mdd, short_rev_sum_mdd, np.min(rev_short_np_pr)

        except Exception as e:
            print("error in 235 :", e)

        try:
            #         long       #
            # rev_long_np_pr = 1 / (np.array(long_list) + fee) - fee
            rev_long_fee = tp_fee + out_fee - np.array(long_fee_list)
            rev_long_np_pr = (1 / ((np.array(long_list) - 1) / config.lvrg_set.leverage + np.array(
                long_fee_list) + 1) - rev_long_fee - 1) * config.lvrg_set.leverage + 1

            rev_long_sr = sharpe_ratio(rev_long_np_pr)

            long_rev_total_pr = np.cumprod(rev_long_np_pr)
            rev_long_wr = len(rev_long_np_pr[rev_long_np_pr > 1]) / len(rev_long_np_pr[rev_long_np_pr != 1])

            rev_long_for_sum_pr = rev_long_np_pr - 1
            rev_long_for_sum_pr[0] = 1
            long_rev_sum_pr = np.cumsum(rev_long_for_sum_pr)
            long_rev_sum_pr = np.where(long_rev_sum_pr < 0, 0, long_rev_sum_pr)

            long_rev_rollmax_pr = np.maximum.accumulate(long_rev_total_pr)
            long_rev_acc_mdd = np.max((long_rev_rollmax_pr - long_rev_total_pr) / long_rev_rollmax_pr)
            long_rev_rollmax_sumpr = np.maximum.accumulate(long_rev_sum_pr)
            long_rev_sum_mdd = np.max((long_rev_rollmax_sumpr - long_rev_sum_pr) / long_rev_rollmax_sumpr)

            plt.subplot(gs[5])

            plt.plot(long_rev_total_pr)
            plt.plot(long_rev_sum_pr, color='gold')

            plt.title(
                "wr : %.3f\n sr : %.3f\n acc_pr : %.3f\n sum_pr : %.3f\n acc_mdd : -%.3f\n sum_mdd : -%.3f\n min_pr : %.3f"
                % (rev_long_wr, rev_long_sr, long_rev_total_pr[-1], long_rev_sum_pr[-1],
                   long_rev_acc_mdd, long_rev_sum_mdd, np.min(rev_long_np_pr)), fontsize=fontsize)

            rev_long_survey_df.iloc[survey_i] = rev_long_wr, rev_long_sr, long_rev_total_pr[-1], long_rev_sum_pr[
                -1], long_rev_acc_mdd, long_rev_sum_mdd, np.min(rev_long_np_pr)

        except Exception as e:
            print("error in 236 :", e)

        if show_plot:

            try:
                frq_dev, s_frq_dev, l_frq_dev = frq_dev_plot(res_df, trade_list, side_list, plot=False)
                plt.subplot(gs[6])
                plt.plot(frq_dev)

                plt.subplot(gs[7])
                plt.plot(s_frq_dev)

                plt.subplot(gs[8])
                plt.plot(l_frq_dev)

            except Exception as e:
                print("error in frq_dev_plot :", e)

            plt.show()

        try:

            h_np_pr = np.array(h_pr_list)
            # h_rev_np_pr = 1 / (np.array(h_pr_list) + fee) - fee    # define, for plot_check below cell
            h_rev_np_pr = (1 / (
                        (np.array(h_pr_list) - 1) / config.lvrg_set.leverage + np.array(fee_list) + 1) - np.array(
                fee_list) - 1) * config.lvrg_set.leverage + 1

            # --------------------- h pr plot --------------------- #
            if len(h_np_pr[h_np_pr != 1]) != 0:

                plt.figure(figsize=(16, 12))
                plt.suptitle(key + " hedge")

                h_total_pr = np.cumprod(h_np_pr)
                h_wr = len(h_np_pr[h_np_pr > 1]) / len(h_np_pr[h_np_pr != 1])

                plt.subplot(gs[0])

                plt.plot(h_total_pr)
                plt.title("wr : %.3f\n min_pr : %.3f\n acc_pr : %.3f\n leverage %s" % (
                h_wr, np.min(h_np_pr), h_total_pr[-1], config.lvrg_set.leverage))
                # plt.show()

                #         short only      #
                h_short_np_pr = np.array(h_short_list)

                short_h_total_pr = np.cumprod(h_short_np_pr)
                h_short_wr = len(h_short_np_pr[h_short_np_pr > 1]) / len(h_short_np_pr[h_short_np_pr != 1])

                plt.subplot(gs[1])
                plt.plot(short_h_total_pr)
                plt.title("wr : %.3f\n min_pr : %.3f\n acc_pr : %.3f\n leverage %s" % (
                h_short_wr, np.min(h_short_np_pr), short_h_total_pr[-1], config.lvrg_set.leverage))

                #         long only      #
                h_long_np_pr = np.array(h_long_list)

                long_h_total_pr = np.cumprod(h_long_np_pr)
                h_long_wr = len(h_long_np_pr[h_long_np_pr > 1]) / len(h_long_np_pr[h_long_np_pr != 1])

                plt.subplot(gs[2])
                plt.plot(long_h_total_pr)
                plt.title("wr : %.3f\n min_pr : %.3f\n acc_pr : %.3f\n leverage %s" % (
                h_long_wr, np.min(h_long_np_pr), long_h_total_pr[-1], config.lvrg_set.leverage))

                #     reversion adjustment      #

                h_rev_total_pr = np.cumprod(h_rev_np_pr)
                h_rev_wr = len(h_rev_np_pr[h_rev_np_pr > 1]) / len(h_rev_np_pr[h_rev_np_pr != 1])

                plt.subplot(gs[3])
                plt.plot(h_rev_total_pr)
                plt.title("wr : %.3f\n min_pr : %.3f\n acc_pr : %.3f\n leverage %s" % (
                h_rev_wr, np.min(h_rev_np_pr), h_rev_total_pr[-1], config.lvrg_set.leverage))

                #         short       #
                # h_rev_short_np_pr = 1 / (np.array(h_short_list) + fee) - fee
                h_rev_short_np_pr = (1 / ((np.array(h_short_list) - 1) / config.lvrg_set.leverage + np.array(
                    short_fee_list) + 1) - np.array(short_fee_list) - 1) * config.lvrg_set.leverage + 1

                short_h_rev_total_pr = np.cumprod(h_rev_short_np_pr)
                h_rev_short_wr = len(h_rev_short_np_pr[h_rev_short_np_pr > 1]) / len(
                    h_rev_short_np_pr[h_rev_short_np_pr != 1])

                plt.subplot(gs[4])
                plt.plot(short_h_rev_total_pr)
                plt.title("wr : %.3f\n min_pr : %.3f\n acc_pr : %.3f\n leverage %s" % (
                h_rev_short_wr, np.min(h_rev_short_np_pr), short_h_rev_total_pr[-1], config.lvrg_set.leverage))

                #         long       #
                # h_rev_long_np_pr = 1 / (np.array(h_long_list) + fee) - fee
                h_rev_long_np_pr = (1 / ((np.array(h_long_list) - 1) / config.lvrg_set.leverage + np.array(
                    long_fee_list) + 1) - np.array(long_fee_list) - 1) * config.lvrg_set.leverage + 1

                long_h_rev_total_pr = np.cumprod(h_rev_long_np_pr)
                h_rev_long_wr = len(h_rev_long_np_pr[h_rev_long_np_pr > 1]) / len(
                    h_rev_long_np_pr[h_rev_long_np_pr != 1])

                plt.subplot(gs[5])
                plt.plot(long_h_rev_total_pr)
                plt.title("wr : %.3f\n min_pr : %.3f\n acc_pr : %.3f\n leverage %s" % (
                h_rev_long_wr, np.min(h_rev_long_np_pr), long_h_rev_total_pr[-1], config.lvrg_set.leverage))

                if show_plot:
                    plt.show()

        except Exception as e:
            print('error in h_pr plot :', e)

        print()

    if not show_plot:
        plt.savefig(png_save_path)
        print("back_pr.png saved !")

    return open_list, ep_tp_list, trade_list, side_list, strat_ver_list

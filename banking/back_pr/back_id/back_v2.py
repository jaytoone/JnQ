#       Todo        #
#        1. back_id ajustment
#         a. remove "key" variable
#        2. define as a function
#        3. adds some libraries
#        4. plt.show() off
#        5. title_position & tight_layout

from funcs.olds.funcs_indicator_candlescore import *
import matplotlib.pyplot as plt

fee = 0

show_log = 0

basic_st_number = 5  # 5 : 30m

# ------- ep set ------- #
static_ep = 1
# entry_type = 'limit'

entry_type = 'market'

ep_gap = 0.5  # st_gap is critera # 지정한 line (middle) 부터 st_gap 적용
# ep_gap = 0  # st_gap is critera #

ep_protect_gap = 0.25  # ep scheduling 을 위한 protect gap (과한 추격매를 방지하기 위함임)
ep_protect_gap = 0.5

# ep_out_gap = 0.5 # ep ~ out 사이의 gap, 이 안에 들 경우 진입하도록 (max_tr 을 위해 적용함) --> 필요없어질 것
entry_incycle = 0

# ------- out set ------- #
use_out = 1
static_out = 1

hl_out = 1

price_restoration = 1  # if use hl_out, turn on ps

retouch = 0  # out_line 에 대한 retouch 를 out 으로 한건지에 대한 여부
retouch_out_period = 500
second_out = 0  # 기존 out 과는 다른 second_out 사용 여부

out_gap = 0.25
# out_gap = 1

approval_st_gap = 1.5

second_out_gap = 0.5  # static_out 을 위한 gap

# ------- tp set ------- #
non_tp = 0  # without tp set

pgfr = 0.3
# pgfr = 0.8

# exit_type = 'limit'
exit_type = 'market'
static_tp = 1

tp_gap = 0.25  # st_gap is critera
tp_gap = 2.5  # st_gap is critera

p_qty_divider = 1.5

# ------- fee calc ------- #
if entry_type == 'limit':
    fee += 0.0002
else:
    fee += 0.0004

if exit_type == 'non_tp' or exit_type == 'market':
    fee += 0.0004
else:
    fee += 0.0002

print('fee :', fee)
# fee = 0.0006
# print('mod fee :', fee)


# ------- lvrg set ------- #

static_lvrg = 1
target_pct = 0.05
# target_pct = 0.01
# target_pct = 0.005
# target_pct = 0.001

# hl_lookback = 10


# ------- inversion set ------- #
inversion = 1

# if inversion:
# short_entry = [1, 2]
# long_entry = [-1, -2]
# else:
short_entry = [-1, -2]
long_entry = [1, 2]

tp_out_ratio = 0.7

fdist_thresh = 1

# ----------------- indicator ----------------- #
# ------- shift_size ------- #
cloud_shift_size = 1
sma_shift_size = 1
close_shift_size = 1

# ------- lb ------- #
# cloud_lookback = 30
cloud_lookback = 69
# cloud_lookback = 150
# cloud_lookback = 10

sma_lookback = 100
# sma_lookback = 100

sar_lookback = 5

# ------- indi. params ------- #

senkoua_list = ['senkou_a1', 'senkou_a2', 'senkou_a3', 'senkou_a4', 'senkou_a5']
senkoub_list = ['senkou_b1', 'senkou_b2', 'senkou_b3', 'senkou_b4', 'senkou_b5']

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def back_pr_check(res_df, dir_path, lvrg):
    # for survey_i, just_loop in enumerate(range(1)):
    for survey_i, config.loc_set.dt_k in enumerate(x_val_list):
        # for survey_i, config.loc_set.ei_k in enumerate(x_val_list):
        # for survey_i, config.loc_set.spread in enumerate(x_val_list):
        # for survey_i, config.loc_set.bbz_itv in enumerate(itv_list):
        # for survey_i, config.loc_set.dtk_dc_itv_num in enumerate(x_val_list):
        # for survey_i, config.tr_set.ep_gap in enumerate(x_val_list):
        # for survey_i, config.tr_set.out_gap in enumerate(x_val_list):
        # for survey_i, config.tr_set.tp_gap in enumerate(x_val_list):
        # for survey_i, config.ep_set.tf_entry in enumerate(x_val_list):
        # for survey_i, config.ep_set.dr_error in enumerate(x_val_list):
        # for survey_i, config.lvrg_set.leverage in enumerate(x_val_list):
        # for survey_i, config.lvrg_set.target_pct in enumerate(x_val_list):
        # for survey_i, outg_dc_itv_num in enumerate(x_val_list):
        # for survey_i, exp_itv in enumerate(itv_list):
        # for survey_i, dt_k2 in enumerate(x_val_list):
        # for survey_i, out_gap2 in enumerate(x_val_list):

        print("config.loc_set.tpg_itv :", config.loc_set.tpg_itv)
        print("config.loc_set.outg_itv :", config.loc_set.outg_itv)
        print("config.loc_set.dtk_itv :", config.loc_set.dtk_itv)
        # print("dtk_dc_itv :", dtk_dc_itv)
        print("config.loc_set.dtk_dc_itv_num :", config.loc_set.dtk_dc_itv_num)
        print("config.loc_set.dc_period :", config.loc_set.dc_period)
        print("config.loc_set.use_dtk_line :", config.loc_set.use_dtk_line)
        print("config.loc_set.bbz_itv :", config.loc_set.bbz_itv)
        # print("config.loc_set.dtk_itv2 :", config.loc_set.dtk_itv2)

        # tp_lb_period = 100
        # res_df['low_lb'] = res_df['low'].rolling(tp_lb_period).min()
        # res_df['high_lb'] = res_df['high'].rolling(tp_lb_period).max()
        # config.tr_set.out_gap = config.tr_set.tp_gap / config.loc_set.spread

        print("config.loc_set.dt_k :", config.loc_set.dt_k)
        print("config.loc_set.ei_k :", config.loc_set.ei_k)
        print("config.loc_set.spread :", config.loc_set.spread)
        # print("config.ep_set.dr_error :", config.ep_set.dr_error)
        print("config.ep_set.tf_entry :", config.ep_set.tf_entry)
        print("config.ep_set.htf_entry :", config.ep_set.htf_entry)
        print("config.tr_set.ep_gap :", config.tr_set.ep_gap)
        print("config.tr_set.out_gap :", config.tr_set.out_gap)
        print("config.tr_set.tp_gap :", config.tr_set.tp_gap)
        print("config.lvrg_set.leverage :", config.lvrg_set.leverage)
        print("config.lvrg_set.target_pct :", config.lvrg_set.target_pct)

        print("outg_dc_itv_num :", outg_dc_itv_num)
        print("outg_dc_period :", outg_dc_period)
        print("exp_itv :", exp_itv)
        print("spread_ep_gap :", spread_ep_gap)
        print("ep_gap2 :", ep_gap2)
        print("dt_k2 :", dt_k2)
        print("out_gap2 :", out_gap2)

        # res_df['sma_1m'] = res_df['close'].rolling(sma_period).mean()
        # res_df = bb_level(res_df, '5m', bbg)

        res_df['entry'] = np.zeros(len(res_df))
        res_df['h_entry'] = np.zeros(len(res_df))

        # -------- set ep level -------- #

        #       limit ver.     #
        #        bb ep         #
        res_df['short_ep'] = res_df['h_short_rtc_1'] + res_df['h_short_rtc_gap'] * config.tr_set.ep_gap
        res_df['long_ep'] = res_df['h_long_rtc_1'] - res_df['h_long_rtc_gap'] * config.tr_set.ep_gap

        res_df['short_ep_org'] = res_df['short_ep'].copy()
        res_df['long_ep_org'] = res_df['long_ep'].copy()

        res_df['short_ep2'] = res_df['h_short_rtc_1'] + res_df['h_short_rtc_gap'] * ep_gap2
        res_df['long_ep2'] = res_df['h_long_rtc_1'] - res_df['h_long_rtc_gap'] * ep_gap2

        res_df['short_tr_ep'] = res_df['h_short_rtc_1'] + res_df['h_short_rtc_gap'] * spread_ep_gap
        res_df['long_tr_ep'] = res_df['h_long_rtc_1'] - res_df['h_long_rtc_gap'] * spread_ep_gap

        #       market ver.     #
        if config.ep_set.entry_type == "MARKET":
            res_df['short_ep'] = res_df['close']
            res_df['long_ep'] = res_df['close']

        # ---------------------------------------- short = -1 ---------------------------------------- #
        # ---------------- ep_time  ---------------- #
        #        bb level entry      #
        # res_df['entry'] = np.where((res_df['close'].shift(config.ep_set.tf_entry * 1) >= res_df['bb_lower_%s' % exp_itv]) &
        res_df['entry'] = np.where((res_df['open'] >= res_df['bb_lower_%s' % exp_itv]) &
                                   # (res_df['close'].shift(config.ep_set.tf_entry * 1) >= res_df['bb_lower_%s' % exp_itv]) &
                                   (res_df['close'] < res_df['bb_lower_%s' % exp_itv])
                                   , res_df['entry'] - 1, res_df['entry'])

        res_df['entry'] = np.where((res_df['entry'] < 0) &
                                   (np_timeidx % config.ep_set.tf_entry == (config.ep_set.tf_entry - 1))
                                   , res_df['entry'] - 1, res_df['entry'])

        # res_df['entry'] = np.where((res_df['open'] >= res_df['cloud_bline_%s' % cb_itv]) &
        #                 # (res_df['close'].shift(config.ep_set.tf_entry * 1) <= res_df['cloud_bline_%s' % cb_itv]) &
        #                 (res_df['close'] < res_df['cloud_bline_%s' % cb_itv])
        #                 , res_df['entry'] - 1, res_df['entry'])

        # res_df['entry'] = np.where((res_df['open'] >= res_df['bb_lower_1m']) &
        #                 # (res_df['close'].shift(config.ep_set.tf_entry * 1) <= res_df['bb_lower_1m']) &
        #                 (res_df['close'] < res_df['bb_lower_1m'])
        #                 , res_df['entry'] - 1, res_df['entry'])

        res_df['h_entry'] = np.where(  # (res_df['open'] >= res_df['bb_lower_%s' % config.loc_set.dtk_itv]) &
            (res_df['close'].shift(config.ep_set.htf_entry * 1) >= res_df['bb_lower_%s' % config.loc_set.dtk_itv]) &
            (res_df['close'] < res_df['bb_lower_%s' % config.loc_set.dtk_itv]) &
            (np_timeidx % config.ep_set.htf_entry == (config.ep_set.htf_entry - 1))
            , res_df['h_entry'] - 1, res_df['h_entry'])

        # ---------------------------------------- long = 1 ---------------------------------------- #
        # ---------------------- ep_time ---------------------- #

        #      bb level entry      #
        # res_df['entry'] = np.where((res_df['close'].shift(config.ep_set.tf_entry * 1) <= res_df['bb_upper_%s' % exp_itv]) &
        res_df['entry'] = np.where((res_df['open'] <= res_df['bb_upper_%s' % exp_itv]) &
                                   # (res_df['close'].shift(config.ep_set.tf_entry * 1) <= res_df['bb_upper_%s' % exp_itv]) &
                                   (res_df['close'] > res_df['bb_upper_%s' % exp_itv])
                                   , res_df['entry'] + 1, res_df['entry'])

        res_df['entry'] = np.where((res_df['entry'] > 0) &
                                   (np_timeidx % config.ep_set.tf_entry == (config.ep_set.tf_entry - 1))
                                   , res_df['entry'] + 1, res_df['entry'])

        # res_df['entry'] = np.where((res_df['open'] <= res_df['cloud_bline_%s' % cb_itv]) &
        #                   # (res_df['close'].shift(config.ep_set.tf_entry * 1) >= res_df['cloud_bline_%s' % cb_itv]) &
        #                   (res_df['close'] > res_df['cloud_bline_%s' % cb_itv])
        #                 , res_df['entry'] + 1, res_df['entry'])

        # res_df['entry'] = np.where((res_df['open'] <= res_df['bb_upper_1m']) &
        #                   # (res_df['close'].shift(config.ep_set.tf_entry * 1) >= res_df['bb_upper_1m']) &
        #                   (res_df['close'] > res_df['bb_upper_1m'])
        #                 , res_df['entry'] + 1, res_df['entry'])

        res_df['h_entry'] = np.where(  # (res_df['open'] <= res_df['bb_upper_%s' % config.loc_set.dtk_itv]) &
            (res_df['close'].shift(config.ep_set.htf_entry * 1) <= res_df['bb_upper_%s' % config.loc_set.dtk_itv]) &
            (res_df['close'] > res_df['bb_upper_%s' % config.loc_set.dtk_itv]) &
            (np_timeidx % config.ep_set.htf_entry == (config.ep_set.htf_entry - 1))
            , res_df['h_entry'] + 1, res_df['h_entry'])

        # ------------------------------ rtc tp & out ------------------------------ #
        # --------------- bb rtc out --------------- #

        if outg_dc_itv_num != "None":
            res_df['short_rtc_0'] = res_df['high'].rolling(outg_dc_period * outg_dc_itv_num).max()
            res_df['long_rtc_0'] = res_df['low'].rolling(outg_dc_period * outg_dc_itv_num).min()

        res_df['short_out'] = res_df['short_rtc_0'] + res_df['short_rtc_gap'] * config.tr_set.out_gap
        res_df['long_out'] = res_df['long_rtc_0'] - res_df['long_rtc_gap'] * config.tr_set.out_gap

        res_df['short_out_org'] = res_df['short_out'].copy()
        res_df['long_out_org'] = res_df['long_out'].copy()

        res_df['short_out2'] = res_df['short_rtc_0'] + res_df['short_rtc_gap'] * out_gap2
        res_df['long_out2'] = res_df['long_rtc_0'] - res_df['long_rtc_gap'] * out_gap2

        # ------------------------------ tp ------------------------------ #
        # --------------- bb rtc tp --------------- #
        res_df['short_tp'] = res_df['h_short_rtc_1'] - res_df['h_short_rtc_gap'] * config.tr_set.tp_gap
        res_df['long_tp'] = res_df['h_long_rtc_1'] + res_df['h_long_rtc_gap'] * config.tr_set.tp_gap

        # --------------- set tp_line / dtk_line --------------- #
        # res_df['short_tp_1'] = np.where(res_df['h_entry'] == -1, res_df['short_rtc_1'], np.nan)
        # res_df['short_tp_1'] = ffill(res_df['short_tp_1'].values.reshape(1, -1)).reshape(-1, 1)
        # res_df['short_tp_gap'] = np.where(res_df['h_entry'] == -1, res_df['h_short_rtc_gap'], np.nan)  # ltf_gap 은 out 을 위한 gap 임
        # res_df['short_tp_gap'] = ffill(res_df['short_tp_gap'].values.reshape(1, -1)).reshape(-1, 1)

        # res_df['long_tp_1'] = np.where(res_df['h_entry'] == 1, res_df['long_rtc_1'], np.nan)
        # res_df['long_tp_1'] = ffill(res_df['long_tp_1'].values.reshape(1, -1)).reshape(-1, 1)
        # res_df['long_tp_gap'] = np.where(res_df['h_entry'] == 1, res_df['h_long_rtc_gap'], np.nan)
        # res_df['long_tp_gap'] = ffill(res_df['long_tp_gap'].values.reshape(1, -1)).reshape(-1, 1)

        # res_df['h_short_tp_1'] = np.where(res_df['h_entry'] == -1, res_df['h_short_rtc_1'], np.nan)
        # res_df['h_short_tp_1'] = ffill(res_df['h_short_tp_1'].values.reshape(1, -1)).reshape(-1, 1)
        # res_df['h_short_tp_gap'] = np.where(res_df['h_entry'] == -1, res_df['h_short_rtc_gap'], np.nan)
        # res_df['h_short_tp_gap'] = ffill(res_df['h_short_tp_gap'].values.reshape(1, -1)).reshape(-1, 1)

        # res_df['h_long_tp_1'] = np.where(res_df['h_entry'] == 1, res_df['h_long_rtc_1'], np.nan)
        # res_df['h_long_tp_1'] = ffill(res_df['h_long_tp_1'].values.reshape(1, -1)).reshape(-1, 1)
        # res_df['h_long_tp_gap'] = np.where(res_df['h_entry'] == 1, res_df['h_long_rtc_gap'], np.nan)
        # res_df['h_long_tp_gap'] = ffill(res_df['h_long_tp_gap'].values.reshape(1, -1)).reshape(-1, 1)

        if config.loc_set.use_dtk_line:
            res_df['short_dtk_1'] = np.where(res_df['h_entry'] == -1, res_df['short_dtk_1'], np.nan)
            res_df['short_dtk_1'] = ffill(res_df['short_dtk_1'].values.reshape(1, -1)).reshape(-1, 1)
            res_df['short_dtk_gap'] = np.where(res_df['h_entry'] == -1, res_df['short_dtk_gap'], np.nan)
            res_df['short_dtk_gap'] = ffill(res_df['short_dtk_gap'].values.reshape(1, -1)).reshape(-1, 1)

            res_df['long_dtk_1'] = np.where(res_df['h_entry'] == 1, res_df['long_dtk_1'], np.nan)
            res_df['long_dtk_1'] = ffill(res_df['long_dtk_1'].values.reshape(1, -1)).reshape(-1, 1)
            res_df['long_dtk_gap'] = np.where(res_df['h_entry'] == 1, res_df['long_dtk_gap'], np.nan)
            res_df['long_dtk_gap'] = ffill(res_df['long_dtk_gap'].values.reshape(1, -1)).reshape(-1, 1)

        res_df['dc_upper_v2'] = res_df['high'].rolling(config.loc_set.dc_period * config.loc_set.dtk_dc_itv_num).max()
        res_df['dc_lower_v2'] = res_df['low'].rolling(config.loc_set.dc_period * config.loc_set.dtk_dc_itv_num).min()

        # --------------- set partial tp --------------- #
        short_tps = [res_df['short_tp']]
        long_tps = [res_df['long_tp']]

        # short_tps = [short_tp2]
        # long_tps = [long_tp2]

        # short_tps = [short_tp2, short_tp] # org
        # long_tps = [long_tp2, long_tp]

        # short_tps = [short_tp, short_tp2]
        # long_tps = [long_tp, long_tp2]

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

            # ------- fee init ------- #
            if config.ep_set.entry_type == 'LIMIT':
                fee = config.init_set.limit_fee
            else:
                fee = config.init_set.market_fee

            # ------- mr_score init ------- #
            mr_const_cnt = 0
            mr_score = 0
            zone = np.nan

            if res_df['entry'][i] == config.ep_set.short_entry_score:

                # print("i in short :", i)

                initial_i = i

                if config.out_set.static_out:
                    p_i = initial_i
                else:
                    p_i = i

                # -------------- tr scheduling -------------- #
                if config.loc_set.spread != "None":
                    mr_const_cnt += 1

                    if spread_ep_gap != "None":
                        if (res_df['short_tr_ep'].iloc[i] - res_df['short_tp'].iloc[i] - tp_fee *
                            res_df['short_tr_ep'].iloc[i]) / (
                                res_df['short_out'].iloc[i] - res_df['short_tr_ep'].iloc[i] + out_fee *
                                res_df['short_tr_ep'].iloc[i]) >= config.loc_set.spread:
                            mr_score += 1

                    else:
                        if (res_df['short_ep'].iloc[i] - res_df['short_tp'].iloc[i] - tp_fee * res_df['short_ep'].iloc[
                            i]) / (res_df['short_out'].iloc[i] - res_df['short_ep'].iloc[i] + out_fee *
                                   res_df['short_ep'].iloc[i]) >= config.loc_set.spread:
                            # if (res_df['short_ep'].iloc[i] - res_df['short_tp'].iloc[i] - tp_fee * res_df['short_ep'].iloc[i]) / (res_df['short_out'].iloc[i] - res_df['short_ep'].iloc[i] + tp_fee * res_df['short_ep'].iloc[i]) >= config.loc_set.spread:
                            # if (res_df['short_ep'].iloc[i] - res_df['short_tp'].iloc[i] - tp_fee * res_df['short_ep'].iloc[i]) / (res_df['short_out'].iloc[i] - res_df['short_ep'].iloc[i] + out_fee * res_df['short_ep'].iloc[i]) == config.loc_set.spread:
                            # if (res_df['short_ep'].iloc[i] - res_df['short_tp'].iloc[i] - out_fee * res_df['short_ep'].iloc[i]) / (res_df['short_out'].iloc[i] - res_df['short_ep'].iloc[i] + out_fee * res_df['short_ep'].iloc[i]) >= config.loc_set.spread:
                            mr_score += 1

                # -------------- dtk -------------- #
                if config.loc_set.dt_k != "None":
                    mr_const_cnt += 1
                    # if res_df['dc_lower_%s' % dtk_dc_itv].iloc[i] >= res_df['short_rtc_1'].iloc[i] - res_df['h_short_rtc_gap'].iloc[i] * config.loc_set.dt_k:
                    #     dtk_v1 & v2 platform     #
                    if dtk_dc_itv != "None":
                        if res_df['dc_lower_%s' % dtk_dc_itv].iloc[i] >= res_df['short_dtk_1'].iloc[i] - \
                                res_df['short_dtk_gap'].iloc[i] * config.loc_set.dt_k:
                            mr_score += 1

                            #     dc_v2   #
                    else:
                        if res_df['dc_lower_v2'].iloc[i] >= res_df['short_dtk_1'].iloc[i] - \
                                res_df['short_dtk_gap'].iloc[i] * config.loc_set.dt_k:
                            # if res_df['dc_lower_v2'].iloc[i] >= res_df['short_dtk_1'].iloc[i] - res_df['short_dtk_gap'].iloc[i] * config.loc_set.dt_k and \
                            # res_df['dc_upper_v2'].iloc[i] <= res_df['long_dtk_1'].iloc[i] + res_df['long_dtk_gap'].iloc[i] * config.loc_set.dt_k:

                            mr_score += 1

                            # -------------- bb_zone  -------------- #
                # if config.loc_set.bbz_itv != "None":
                #   mr_const_cnt += 1
                #   if res_df['close'].iloc[i] < res_df['bb_lower_%s' % config.loc_set.bbz_itv].iloc[i]:
                #     mr_score += 1

                #           ep2 on csd_zone          #
                if ep_gap2 != "None":
                    #         c_zone        #
                    # if res_df['close'].iloc[i] > res_df['bb_lower_%s' % config.loc_set.bbz_itv].iloc[i]:
                    if res_df['dc_upper_v2'].iloc[i] > res_df['long_dtk_plot_1'].iloc[i] + \
                            res_df['long_dtk_plot_gap'].iloc[i] * dt_k2:
                        res_df['short_ep'] = res_df['short_ep2']
                        res_df['short_out'] = res_df['short_out_org']
                        zone = 'c'

                    #         t_zone        #
                    else:
                        res_df['short_ep'] = res_df['short_ep_org']
                        res_df['short_out'] = res_df['short_out2']
                        zone = 't'

                # -------------- mr_score summation -------------- #
                if mr_score == mr_const_cnt:
                    pass

                else:
                    i += 1
                    if i >= len(res_df):
                        break
                    continue

                ep_j = initial_i  # dynamic ep 를 위한 var.

                # -------------- limit waiting : limit_out -------------- #

                if config.ep_set.entry_type == "LIMIT":

                    entry_done = 0
                    entry_open = 0
                    prev_sar = None

                    # for e_j in range(i, len(res_df)): # entry_signal 이 open 기준 (해당 bar 에서 체결 가능함)
                    if i + 1 >= len(res_df):  # i should be checked if e_j starts from i+1
                        break
                    for e_j in range(i + 1, len(res_df)):  # entry signal이 close 기준 일 경우

                        if not config.ep_set.static_ep:
                            ep_j = e_j

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

                        #             1. check ep_out      #
                        if config.loc_set.ei_k != "None":
                            if res_df['low'].iloc[e_j] <= res_df['h_short_rtc_1'].iloc[tp_j] - \
                                    res_df['h_short_rtc_gap'].iloc[tp_j] * config.loc_set.ei_k:
                                # if res_df['low'].iloc[e_j] <= res_df['short_tp'].iloc[tp_j]: # ep_out : tp_done
                                # if np_timeidx[e_j] % config.ep_set.tf_entry == config.ep_set.tf_entry - 1:
                                break

                        #             2. check ep_in       #
                        if res_df['high'].iloc[e_j] >= res_df['short_ep'].iloc[ep_j]:
                            entry_done = 1
                            # print("res_df['high'].iloc[e_j] :", res_df['high'].iloc[e_j])
                            # print("e_j :", e_j)

                            #     이미, e_j open 이 ep 보다 높은 경우, entry[ep_j] => -2 로 변경   #
                            if res_df['open'].iloc[e_j] >= res_df['short_ep'].iloc[ep_j]:
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

                if config.ep_set.entry_type is 'MARKET':
                    try:
                        ep_list = [res_df['close'].iloc[e_j]]
                    except Exception as e:
                        # print('error in ep_list (initial) :', e)
                        ep_list = [res_df['close'].iloc[ep_j]]

                else:
                    if not entry_open:
                        ep_list = [res_df['short_ep'].iloc[ep_j]]

                    #     Todo    #
                    #      1. entry_score version 으로 재정의해야함
                    #      2. below phase exists for open_price entry
                    else:
                        #   e_j 가 있는 경우,
                        try:
                            ep_list = [res_df['open'].iloc[e_j]]
                        except Exception as e:
                            ep_list = [res_df['open'].iloc[ep_j]]

                if not config.lvrg_set.static_lvrg:
                    config.lvrg_set.leverage = config.lvrg_set.target_pct / (
                                res_df['short_out'].iloc[ep_j] / res_df['short_ep'].iloc[ep_j] - 1 - (
                                    fee + config.init_set.market_fee))
                    # config.lvrg_set.leverage = config.lvrg_set.target_pct / (res_df['short_out'].iloc[ep_j] / res_df['short_ep_org'].iloc[ep_j] - 1 - (fee + config.init_set.market_fee))

                    # config.lvrg_set.leverage = config.lvrg_set.target_pct / abs(res_df['short_ep'].iloc[ep_j] / res_df['short_out'].iloc[ep_j] - 1 - (fee + config.init_set.market_fee))
                    # config.lvrg_set.leverage = config.lvrg_set.target_pct / abs(res_df['short_ep_org'].iloc[ep_j] / res_df['short_out'].iloc[ep_j] - 1 - (fee + config.init_set.market_fee))

                    if not config.lvrg_set.allow_float:
                        config.lvrg_set.leverage = int(config.lvrg_set.leverage)

                    # -------------- leverage rejection -------------- #
                    if config.lvrg_set.leverage < 1 and config.lvrg_set.lvrg_rejection:
                        open_list.pop()
                        zone_list.pop()
                        side_list.pop()
                        i += 1
                        if i >= len(res_df):
                            break
                        continue

                    config.lvrg_set.leverage = max(config.lvrg_set.leverage, 1)
                    leverage_list.append(config.lvrg_set.leverage)

                config.lvrg_set.leverage = min(50, config.lvrg_set.leverage)

                try:
                    ep_idx_list = [e_j]
                except Exception as e:
                    # print('error in ep_idx_list :', e)
                    ep_idx_list = [ep_j]

                tp_list = []
                tp_idx_list = []

                partial_tp_cnt = 0
                hedge_cnt = 1

                h_ep, h_tp = None, None
                h_i, h_j = None, None

                trade_done = False
                out = False
                # config.out_set.retouch

                #     Todo    #
                #      1. future_work : 상단의 retouch 와 겹침
                config.out_set.retouch = False

                if i == len(res_df) - 1:  # if j start from i + 1
                    open_list.pop()
                    zone_list.pop()
                    side_list.pop()
                for j in range(i + 1, len(res_df)):

                    # for j in range(i, len(res_df)):

                    if config.tp_set.static_tp:
                        if not config.ep_set.static_ep and config.ep_set.entry_type == "LIMIT" and config.ep_set.tpout_onexec:
                            tp_j = e_j
                        else:
                            tp_j = initial_i
                    else:
                        tp_j = j

                    if config.out_set.static_out:
                        if not config.ep_set.static_ep and config.ep_set.entry_type == "LIMIT" and config.ep_set.tpout_onexec:
                            out_j = e_j
                        else:
                            out_j = initial_i
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
                        if config.tp_set.tp_type != 'MARKET':

                            for s_i, short_tp_ in enumerate(short_tps):

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
                                                tp_state_list.append("d-short_open")

                                        #   tp limit 이 가능한 경우 = open 이 아직, tp 를 넘지 않은 경우 #
                                        else:

                                            # tp = short_tp_.iloc[initial_i]
                                            tp = short_tp_.iloc[j]
                                            # tp = res_df['open'].iloc[j]

                                            if trade_done:
                                                tp_state_list.append("d-short_tp")

                                    #         static tp         #
                                    else:

                                        #   tp limit 이 불가한 경우 = open 이 이미, tp 를 넘은 경우 #
                                        #   non_inversion 의 경우, short_tp 가 가능함   #

                                        # if res_df['open'].iloc[j] < short_tp_.iloc[initial_i]:
                                        if res_df['open'].iloc[j] < short_tp_.iloc[tp_j]:

                                            # tp = short_tp_.iloc[initial_i]
                                            tp = short_tp_.iloc[tp_j]

                                            # tp = res_df['open'].iloc[j]

                                            if trade_done:
                                                tp_state_list.append("s-short_tp")

                                        #   tp limit 이 가능한 경우 = open 이 아직, tp 를 넘지 않은 경우 #
                                        else:

                                            # tp = short_tp_.iloc[initial_i]
                                            tp = short_tp_.iloc[tp_j]

                                            # tp = res_df['open'].iloc[j]

                                            if trade_done:
                                                tp_state_list.append("s-short_tp")

                                    tp_list.append(tp)
                                    tp_idx_list.append(j)
                                    fee += config.init_set.limit_fee


                        #           2. by signal        #
                        else:

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
                            if np_timeidx[j] % config.ep_set.tf_entry == config.ep_set.tf_entry - 1:

                                tp = res_df['close'].iloc[j]
                                # tp = res_df['open'].iloc[j]
                                trade_done = 1

                                if trade_done:
                                    tp_state_list.append("short close tp")

                                tp_list.append(tp)
                                tp_idx_list.append(j)
                                fee += config.init_set.market_fee

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
                                static_short_out = res_df['short_out'].iloc[j]

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
                                if res_df['high'].iloc[j] >= res_df['short_out'].iloc[out_j]:  # check out only once
                                    out = 1

                            else:
                                if res_df['close'].iloc[j] >= res_df['short_out'].iloc[out_j]:  # check out only once
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
                                tp = res_df['short_out'].iloc[out_j]
                                if config.out_set.second_out:
                                    tp = res_df['short_out2'].iloc[out_j]

                                # if res_df['close'].iloc[j] > tp: # 이 경우를 protect 하는건 insane 임
                                #   tp = res_df['close'].iloc[j]

                            else:

                                if res_df['open'].iloc[j] >= res_df['short_out'].iloc[out_j]:
                                    tp = res_df['open'].iloc[j]
                                else:
                                    if config.out_set.hl_out:
                                        tp = res_df['short_out'].iloc[out_j]
                                    else:
                                        tp = res_df['close'].iloc[j]

                                # if not config.out_set.static_out:
                                #   if res_df['open'].iloc[j] >= res_df['short_out'].iloc[out_j]: # close 기준이라 이런 조건을 못씀, 차라리 j 를 i 부터 시작
                                #     tp = res_df['open'].iloc[j]
                                #   else:
                                #     tp = res_df['close'].iloc[j]

                                # else:
                                #   tp = res_df['close'].iloc[j]

                            if config.out_set.retouch:  # out 과 open 비교
                                if config.out_set.second_out:
                                    if res_df['open'].iloc[j] <= res_df['short_out2'].iloc[out_j]:
                                        tp = res_df['open'].iloc[j]
                                else:
                                    if res_df['open'].iloc[j] <= res_df['short_out'].iloc[out_j]:
                                        tp = res_df['open'].iloc[j]

                                try:  # static_short_out 인 경우, open 도 고려한 tp set
                                    if res_df['open'].iloc[j] <= static_short_out:
                                        tp = res_df['open'].iloc[j]
                                    else:
                                        tp = static_short_out
                                except Exception as e:
                                    pass

                            trade_done = 1
                            tp_state_list.append("short close_out")

                            tp_list.append(tp)
                            tp_idx_list.append(j)
                            fee += config.init_set.market_fee

                    # -------------- non tp -------------- #
                    if j == len(res_df) - 1:
                        trade_done = 1
                        tp = res_df['close'].iloc[j]
                        tp_list.append(tp)
                        tp_idx_list.append(j)
                        fee += config.init_set.market_fee

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
                            done_tp = res_df['short_tp'].iloc[ep_j]
                            done_out = res_df['short_out'].iloc[ep_j]

                            if done_out <= ep_list[0]:  # loss > 1
                                dr = np.nan
                                tp_ratio = np.nan
                            else:
                                dr = ((ep_list[0] - done_tp) / (done_out - ep_list[0]))
                                tp_ratio = ((ep_list[0] - done_tp - tp_fee * ep_list[0]) / (
                                            done_out - ep_list[0] + out_fee * ep_list[0]))

                        except Exception as e:
                            # pass
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

                        hh = max(res_df['high'].iloc[i:j + 1])
                        short_liq = (ep_list[0] / hh - fee - 1) * config.lvrg_set.leverage + 1

                        if j != len(res_df) - 1:

                            # ep_tp_list.append((ep, tp_list))
                            ep_tp_list.append((ep_list, tp_list))
                            # trade_list.append([initial_i, i, j])
                            trade_list.append((ep_idx_list, tp_idx_list))

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

                            #         tp 미체결 survey        #
                            nontp_liqd_list.append(short_liq)
                            nontp_short_liqd_list.append(short_liq)
                            nontp_short_indexs.append(i)
                            nontp_short_ep_list.append(ep_list[0])

                            nontp_short_pr = (ep_list[0] / tp - fee - 1) * config.lvrg_set.leverage + 1
                            nontp_pr_list.append(nontp_short_pr)
                            nontp_short_pr_list.append(nontp_short_pr)


            #                  long  phase                #
            elif res_df['entry'][i] == -config.ep_set.short_entry_score:

                initial_i = i

                if config.out_set.static_out:
                    p_i = initial_i
                else:
                    p_i = i

                # -------------- tr scheduling -------------- #
                if config.loc_set.spread != "None":

                    mr_const_cnt += 1
                    if spread_ep_gap != "None":
                        if (res_df['long_tp'].iloc[i] - res_df['long_tr_ep'].iloc[i] - tp_fee *
                            res_df['long_tr_ep'].iloc[i]) / (
                                res_df['long_tr_ep'].iloc[i] - res_df['long_out'].iloc[i] + out_fee *
                                res_df['long_tr_ep'].iloc[i]) >= config.loc_set.spread:
                            mr_score += 1

                    else:
                        if (res_df['long_tp'].iloc[i] - res_df['long_ep'].iloc[i] - tp_fee * res_df['long_ep'].iloc[
                            i]) / (res_df['long_ep'].iloc[i] - res_df['long_out'].iloc[i] + out_fee *
                                   res_df['long_ep'].iloc[i]) >= config.loc_set.spread:
                            # if (res_df['long_tp'].iloc[i] - res_df['long_ep'].iloc[i] - tp_fee * res_df['long_ep'].iloc[i]) / (res_df['long_ep'].iloc[i] - res_df['long_out'].iloc[i] + tp_fee * res_df['long_ep'].iloc[i]) >= config.loc_set.spread:
                            # if (res_df['long_tp'].iloc[i] - res_df['long_ep'].iloc[i] - tp_fee * res_df['long_ep'].iloc[i]) / (res_df['long_ep'].iloc[i] - res_df['long_out'].iloc[i] + out_fee * res_df['long_ep'].iloc[i]) == config.loc_set.spread:
                            # if (res_df['long_tp'].iloc[i] - res_df['long_ep'].iloc[i] - out_fee * res_df['long_ep'].iloc[i]) / (res_df['long_ep'].iloc[i] - res_df['long_out'].iloc[i] + out_fee * res_df['long_ep'].iloc[i]) >= config.loc_set.spread:
                            mr_score += 1

                # -------------- dtk -------------- #
                if config.loc_set.dt_k != "None":

                    mr_const_cnt += 1
                    # if res_df['dc_upper_%s' % dtk_dc_itv].iloc[i] <= res_df['long_rtc_1'].iloc[i] + res_df['long_rtc_gap'].iloc[i] * config.loc_set.dt_k:
                    #     dtk_v1 & v2 platform    #
                    if dtk_dc_itv != "None":
                        if res_df['dc_upper_%s' % dtk_dc_itv].iloc[i] <= res_df['long_dtk_1'].iloc[i] + \
                                res_df['long_dtk_gap'].iloc[i] * config.loc_set.dt_k:
                            mr_score += 1

                    else:
                        #     dc_v2     #
                        if res_df['dc_upper_v2'].iloc[i] <= res_df['long_dtk_1'].iloc[i] + res_df['long_dtk_gap'].iloc[
                            i] * config.loc_set.dt_k:
                            # if res_df['dc_upper_v2'].iloc[i] <= res_df['long_dtk_1'].iloc[i] + res_df['long_dtk_gap'].iloc[i] * config.loc_set.dt_k and \
                            #   res_df['dc_lower_v2'].iloc[i] >= res_df['short_dtk_1'].iloc[i] - res_df['short_dtk_gap'].iloc[i] * config.loc_set.dt_k:

                            mr_score += 1

                            # -------------- bb_zone  -------------- #
                # if config.loc_set.bbz_itv != "None":
                #   mr_const_cnt += 1
                #   if res_df['close'].iloc[i] > res_df['bb_upper_%s' % config.loc_set.bbz_itv].iloc[i]:
                #     mr_score += 1

                #           ep2 on csd_zone          #
                if ep_gap2 != "None":
                    #         c_zone        #
                    # if res_df['close'].iloc[i] < res_df['bb_upper_%s' % config.loc_set.bbz_itv].iloc[i]:
                    if res_df['dc_lower_v2'].iloc[i] < res_df['short_dtk_plot_1'].iloc[i] - \
                            res_df['short_dtk_plot_gap'].iloc[i] * dt_k2:
                        res_df['long_ep'] = res_df['long_ep2']
                        res_df['long_out'] = res_df['long_out_org']
                        zone = 'c'

                        # mr_const_cnt += 1
                        # dc_lb_period = 100
                        # if np.sum((res_df['dc_upper_15m'] > res_df['dc_upper_15m'].shift(15)).iloc[i - dc_lb_period:i]) == 0:
                        #   mr_score += 1


                    #         t_zone        #
                    else:
                        res_df['long_ep'] = res_df['long_ep_org']
                        res_df['long_out'] = res_df['long_out2']
                        zone = 't'

                # -------------- mr_score summation -------------- #
                if mr_score == mr_const_cnt:
                    pass

                else:
                    i += 1
                    if i >= len(res_df):
                        break
                    continue

                # print("i after mrcheck :", i)

                ep_j = initial_i

                # -------------- limit waiting const. -------------- #
                if config.ep_set.entry_type == "LIMIT":

                    entry_done = 0
                    entry_open = 0
                    prev_sar = None

                    # for e_j in range(i, len(res_df)):

                    if i + 1 >= len(res_df):  # i should be checked if e_j starts from i+1
                        break
                    for e_j in range(i + 1, len(res_df)):  # entry 가 close 기준일 경우 사용 (open 기준일 경우 i 부터 시작해도 무방함)

                        if not config.ep_set.static_ep:
                            ep_j = e_j

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

                        #          1. check ep_out          #
                        if config.loc_set.ei_k != "None":
                            if res_df['high'].iloc[e_j] >= res_df['h_long_rtc_1'].iloc[tp_j] + \
                                    res_df['h_long_rtc_gap'].iloc[tp_j] * config.loc_set.ei_k:
                                # if res_df['high'].iloc[e_j] >= res_df['long_tp'].iloc[tp_j]:
                                # if np_timeidx[e_j] % config.ep_set.tf_entry == config.ep_set.tf_entry - 1:
                                break

                        #          2. check ep_in          #
                        if res_df['low'].iloc[e_j] <= res_df['long_ep'].iloc[ep_j]:
                            entry_done = 1
                            # print("e_j :", e_j)

                            #     이미, e_j open 이 ep 보다 낮은 경우, entry[initial_i] => -2 로 변경   #
                            if res_df['open'].iloc[e_j] <= res_df['long_ep'].iloc[ep_j]:
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

                if config.ep_set.entry_type is 'MARKET':
                    try:
                        ep_list = [res_df['close'].iloc[e_j]]
                    except Exception as e:
                        # print('error in ep_list (initial) :', e)
                        ep_list = [res_df['close'].iloc[ep_j]]
                else:
                    if not entry_open:
                        ep_list = [res_df['long_ep'].iloc[ep_j]]
                    else:
                        try:
                            ep_list = [res_df['open'].iloc[e_j]]
                        except Exception as e:
                            ep_list = [res_df['open'].iloc[ep_j]]

                if not config.lvrg_set.static_lvrg:
                    config.lvrg_set.leverage = config.lvrg_set.target_pct / (
                                res_df['long_ep'].iloc[ep_j] / res_df['long_out'].iloc[ep_j] - 1 - (
                                    fee + config.init_set.market_fee))
                    # config.lvrg_set.leverage = config.lvrg_set.target_pct / (res_df['long_ep_org'].iloc[ep_j] / res_df['long_out'].iloc[ep_j] - 1 - (fee + config.init_set.market_fee))

                    # config.lvrg_set.leverage = config.lvrg_set.target_pct / abs(res_df['long_out'].iloc[ep_j] / res_df['long_ep'].iloc[ep_j] - 1 - (fee + config.init_set.market_fee))
                    # config.lvrg_set.leverage = config.lvrg_set.target_pct / abs(res_df['long_out'].iloc[ep_j] / res_df['long_ep_org'].iloc[ep_j] - 1 - (fee + config.init_set.market_fee))

                    if not config.lvrg_set.allow_float:
                        config.lvrg_set.leverage = int(config.lvrg_set.leverage)

                    # -------------- leverage rejection -------------- #
                    if config.lvrg_set.leverage < 1 and config.lvrg_set.lvrg_rejection:
                        open_list.pop()
                        zone_list.pop()
                        side_list.pop()
                        i += 1
                        if i >= len(res_df):
                            break
                        continue

                    config.lvrg_set.leverage = max(config.lvrg_set.leverage, 1)
                    leverage_list.append(config.lvrg_set.leverage)

                config.lvrg_set.leverage = min(50, config.lvrg_set.leverage)

                try:
                    ep_idx_list = [e_j]
                except Exception as e:
                    # print('error in ep_idx_list :', e)
                    ep_idx_list = [ep_j]

                tp_list = []
                tp_idx_list = []

                partial_tp_cnt = 0
                hedge_cnt = 1

                h_ep, h_tp = None, None
                h_i, h_j = None, None

                trade_done = False
                out = False
                config.out_set.retouch = False

                if i == len(res_df) - 1:  # if j start from i + 1
                    open_list.pop()
                    zone_list.pop()
                    side_list.pop()

                for j in range(i + 1, len(res_df)):

                    # for j in range(i, len(res_df)):

                    if config.tp_set.static_tp:
                        if not config.ep_set.static_ep and config.ep_set.entry_type == "LIMIT" and config.ep_set.tpout_onexec:
                            tp_j = e_j
                        else:
                            tp_j = initial_i
                    else:
                        tp_j = j

                    if config.out_set.static_out:
                        if not config.ep_set.static_ep and config.ep_set.entry_type == "LIMIT" and config.ep_set.tpout_onexec:
                            out_j = e_j
                        else:
                            out_j = initial_i
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
                        if config.tp_set.tp_type != 'MARKET':

                            for l_i, long_tp_ in enumerate(long_tps):

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
                                                tp_state_list.append("d-long_open")


                                        #   tp limit 이 가능한 경우 = open 이 아직, tp 를 넘지 않은 경우 #
                                        else:

                                            # tp = long_tp_.iloc[initial_i]
                                            tp = long_tp_.iloc[j]
                                            # tp = res_df['open'].iloc[j]

                                            if trade_done:
                                                tp_state_list.append("d-long_tp")

                                    #         static tp         #
                                    else:

                                        #   tp limit 이 불가한 경우 = open 이 이미, tp 를 넘은 경우 #
                                        #   non_inversion 의 경우, short_tp 가 가능함   #

                                        if res_df['open'].iloc[j] >= long_tp_.iloc[tp_j]:
                                            # if res_df['open'].iloc[j] >= long_tp_.iloc[initial_i]:

                                            # tp = long_tp_.iloc[initial_i]
                                            tp = long_tp_.iloc[tp_j]

                                            # tp = res_df['open'].iloc[j]

                                            if trade_done:
                                                tp_state_list.append("s-long_tp")


                                        #   tp limit 이 가능한 경우 = open 이 아직, tp 를 넘지 않은 경우 #
                                        else:

                                            # tp = long_tp_.iloc[initial_i]
                                            tp = long_tp_.iloc[tp_j]

                                            # tp = res_df['open'].iloc[j]

                                            if trade_done:
                                                tp_state_list.append("s-long_tp")

                                    tp_list.append(tp)
                                    tp_idx_list.append(j)
                                    fee += config.init_set.limit_fee

                        #           2. by time        #
                        else:

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
                            if np_timeidx[j] % config.ep_set.tf_entry == config.ep_set.tf_entry - 1:

                                tp = res_df['close'].iloc[j]
                                # tp = res_df['open'].iloc[j]
                                trade_done = 1

                                if trade_done:
                                    tp_state_list.append("long close tp")

                                tp_list.append(tp)
                                tp_idx_list.append(j)
                                fee += config.init_set.market_fee

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
                                static_long_out = res_df['long_out'].iloc[j]

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
                                if res_df['low'].iloc[j] <= res_df['long_out'].iloc[out_j]:  # check out only once
                                    out = 1

                            else:
                                if res_df['close'].iloc[j] <= res_df['long_out'].iloc[out_j]:  # check out only once
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
                                tp = res_df['long_out'].iloc[out_j]
                                if config.out_set.second_out:
                                    tp = res_df['long_out2'].iloc[out_j]

                                # if res_df['close'].iloc[j] < tp: # 이 경우를 protect 하는건 insane 임
                                # # if res_df['high'].iloc[j] < tp: # --> config.out_set.hl_out 사용시 이 조건은 valid 함
                                #   tp = res_df['close'].iloc[j]

                            else:

                                if res_df['open'].iloc[j] <= res_df['long_out'].iloc[out_j]:
                                    tp = res_df['open'].iloc[j]
                                else:
                                    if config.out_set.hl_out:
                                        tp = res_df['long_out'].iloc[out_j]
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
                                    if res_df['open'].iloc[j] >= res_df['long_out2'].iloc[
                                        out_j]:  # dynamic_out 일 경우 고려해야함
                                        tp = res_df['open'].iloc[j]
                                else:
                                    if res_df['open'].iloc[j] >= res_df['long_out'].iloc[
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
                            tp_state_list.append("long close_out")
                            trade_done = 1

                            tp_list.append(tp)
                            tp_idx_list.append(j)
                            fee += config.init_set.market_fee

                    # -------------- non tp -------------- #
                    if j == len(res_df) - 1:
                        trade_done = 1
                        tp = res_df['close'].iloc[j]
                        tp_list.append(tp)
                        tp_idx_list.append(j)
                        fee += config.init_set.market_fee

                    if trade_done:

                        # --------------- tp_ratio info --------------- #
                        try:
                            done_tp = res_df['long_tp'].iloc[ep_j]
                            done_out = res_df['long_out'].iloc[ep_j]

                            if done_out >= ep_list[0]:  # loss >= 1
                                tp_ratio = np.nan
                                dr = np.nan
                            else:
                                tp_ratio = ((done_tp - ep_list[0] - tp_fee * ep_list[0]) / (
                                            ep_list[0] - done_out + out_fee * ep_list[0]))
                                dr = ((done_tp - ep_list[0]) / (ep_list[0] - done_out))

                        except Exception as e:
                            # pass
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

                        ll = min(res_df['low'].iloc[i:j + 1])
                        long_liq = (ll / ep_list[0] - fee - 1) * config.lvrg_set.leverage + 1

                        if j != len(res_df) - 1:

                            ep_tp_list.append((ep_list, tp_list))
                            trade_list.append((ep_idx_list, tp_idx_list))

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
        plt.style.use('default')
        # mpl.rcParams.update(mpl.rcParamsDefault)

        plt.figure(figsize=(16, 12))
        # plt.figure(figsize=(12, 8))
        # plt.figure(figsize=(10, 6))
        plt.suptitle(key)

        try:
            np_pr = np.array(pr_list)
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

            # mean_profit = np.mean(np_pr[np_pr > 1])
            # mean_loss = np.mean(np_pr[np_pr < 1])
            # cumprod_profit = np.cumprod(np_pr[np_pr > 1])[-1]
            # cumprod_loss = np.cumprod(np_pr[np_pr < 1])[-1]
            # pr_tr = cumprod_profit * cumprod_loss

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

            # plt.subplot(121)
            plt.subplot(231)
            plt.plot(total_pr)
            plt.plot(sum_pr, color='gold')
            if len(nontp_liqd_list) != 0:
                plt.title("wr : %.3f\nlen(td) : %s\nmin_pr : %.3f\nacc_pr : %.3f\n sum_pr : %.3f"
                          % (wr, len(np_pr[np_pr != 1]), np.min(np_pr), total_pr[-1], sum_pr[-1]) + \
                          "\n leverage %s\nliqd : %.3f\nmean_tr : %.3f\n mean_dr : %.3f\n nontp_liqd_cnt : %s\nnontp_liqd : %.3f\nontp_liqd_pr : %.3f\n tw cw tls cls : %s %s %s %s"
                          % (config.lvrg_set.leverage, min(liqd_list), mean_tr, mean_dr, len(nontp_liqd_list),
                             min(nontp_liqd_list), min(nontp_pr_list), t_w, c_w, t_ls, c_ls))
            else:
                plt.title("wr : %.3f\nlen(td) : %s\nmin_pr : %.3f\nacc_pr : %.3f\n sum_pr : %.3f"
                          % (wr, len(np_pr[np_pr != 1]), np.min(np_pr), total_pr[-1], sum_pr[-1]) + \
                          "\n leverage %s\nliqd : %.3f\nmean_tr : %.3f\n mean_dr : %.3f\n nontp_liqd_cnt : %s\n tw cw tls cls : %s %s %s %s"
                          % (config.lvrg_set.leverage, min(liqd_list), mean_tr, mean_dr, len(nontp_liqd_list), t_w, c_w,
                             t_ls, c_ls))
            # plt.show()

            survey_df.iloc[survey_i] = wr, len(np_pr[np_pr != 1]), np.min(np_pr), total_pr[-1], sum_pr[-1], min(
                liqd_list), mean_tr, mean_dr

            print('supblot231 passed')

        except Exception as e:
            print("error in 231 :", e)

        try:
            #         short only      #
            short_np_pr = np.array(short_list)

            total_short_pr = np.cumprod(short_np_pr)

            short_for_sum_pr = short_np_pr - 1
            short_for_sum_pr[0] = 1
            short_sum_pr = np.cumsum(short_for_sum_pr)
            short_sum_pr = np.where(short_sum_pr < 0, 0, short_sum_pr)

            short_wr = len(short_np_pr[short_np_pr > 1]) / len(short_np_pr[short_np_pr != 1])

            t_w_s = np.sum(np.where((np_zone_list == 't') & (np_pr > 1) & (np_side_list == 's'), 1, 0))
            c_w_s = np.sum(np.where((np_zone_list == 'c') & (np_pr > 1) & (np_side_list == 's'), 1, 0))
            t_ls_s = np.sum(np.where((np_zone_list == 't') & (np_pr < 1) & (np_side_list == 's'), 1, 0))
            c_ls_s = np.sum(np.where((np_zone_list == 'c') & (np_pr < 1) & (np_side_list == 's'), 1, 0))
            # short_cumprod_profit = np.cumprod(short_np_pr[short_np_pr > 1])[-1]
            # short_cumprod_loss = np.cumprod(short_np_pr[short_np_pr < 1])[-1]
            # short_pr_tr = short_cumprod_profit * short_cumprod_loss

            np_short_tp_ratio_list = np.array(short_tp_ratio_list)
            mean_short_tr = np.mean(np_short_tp_ratio_list[np.isnan(np_short_tp_ratio_list) == 0])

            np_short_dr_list = np.array(short_dr_list)
            mean_short_dr = np.mean(np_short_dr_list[np.isnan(np_short_dr_list) == 0])

            # short_pr_gap = (short_np_pr - 1) / config.lvrg_set.leverage + fee
            # short_tp_gap = short_pr_gap[short_pr_gap > 0]
            # # mean_short_tp_gap = np.mean(short_pr_gap[short_pr_gap > 0])
            # # mean_short_ls_gap = np.mean(short_pr_gap[short_pr_gap < 0])

            # mean_short_pgfr = np.mean((short_tp_gap - fee) / abs(short_tp_gap + fee))

            plt.subplot(232)
            plt.plot(total_short_pr)
            plt.plot(short_sum_pr, color='gold')
            if len(nontp_short_liqd_list) != 0:

                max_nontp_short_term = len(res_df) - nontp_short_indexs[0]

                plt.title("wr : %.3f\nlen(short_td) : %s\nmin_pr : %.3f\nacc_pr : %.3f\n sum_pr : %.3f"
                          % (short_wr, len(short_np_pr[short_np_pr != 1]), np.min(short_np_pr), total_short_pr[-1],
                             short_sum_pr[-1]) + \
                          "\n leverage %s\nliqd : %.3f\nmean_tr : %.3f\n mean_dr : %.3f\n nontp_short_liqd_cnt : %s\nnontp_short_liqd : %.3f\nontp_short_liqd_pr : %.3f\nmax_nontp_short_term : %s\n tw cw tls cls : %s %s %s %s"
                          % (config.lvrg_set.leverage, min(short_liqd_list), mean_short_tr, mean_short_dr,
                             len(nontp_short_liqd_list), min(nontp_short_liqd_list), min(nontp_short_pr_list),
                             max_nontp_short_term, t_w_s, c_w_s, t_ls_s, c_ls_s))
            else:
                plt.title("wr : %.3f\nlen(short_td) : %s\nmin_pr : %.3f\nacc_pr : %.3f\n sum_pr : %.3f"
                          % (short_wr, len(short_np_pr[short_np_pr != 1]), np.min(short_np_pr), total_short_pr[-1],
                             short_sum_pr[-1]) + \
                          "\n leverage %s\nliqd : %.3f\nmean_tr : %.3f\n mean_dr : %.3f\n nontp_short_liqd_cnt : %s\n tw cw tls cls : %s %s %s %s"
                          % (config.lvrg_set.leverage, min(short_liqd_list), mean_short_tr, mean_short_dr,
                             len(nontp_short_liqd_list), t_w_s, c_w_s, t_ls_s, c_ls_s))

            short_survey_df.iloc[survey_i] = short_wr, len(short_np_pr[short_np_pr != 1]), np.min(short_np_pr), \
                                             total_short_pr[-1], short_sum_pr[-1], min(
                short_liqd_list), mean_short_tr, mean_short_dr

            print('supblot232 passed')

        except Exception as e:
            print("error in 232 :", e)

        try:
            #         long only      #
            long_np_pr = np.array(long_list)
            # long_np_pr = (np.array(long_list) - 1) * config.lvrg_set.leverage + 1

            total_long_pr = np.cumprod(long_np_pr)

            long_for_sum_pr = long_np_pr - 1
            long_for_sum_pr[0] = 1
            long_sum_pr = np.cumsum(long_for_sum_pr)
            long_sum_pr = np.where(long_sum_pr < 0, 0, long_sum_pr)

            long_wr = len(long_np_pr[long_np_pr > 1]) / len(long_np_pr[long_np_pr != 1])

            t_w_l = np.sum(np.where((np_zone_list == 't') & (np_pr > 1) & (np_side_list == 'l'), 1, 0))
            c_w_l = np.sum(np.where((np_zone_list == 'c') & (np_pr > 1) & (np_side_list == 'l'), 1, 0))
            t_ls_l = np.sum(np.where((np_zone_list == 't') & (np_pr < 1) & (np_side_list == 'l'), 1, 0))
            c_ls_l = np.sum(np.where((np_zone_list == 'c') & (np_pr < 1) & (np_side_list == 'l'), 1, 0))
            # long_cumprod_profit = np.cumprod(long_np_pr[long_np_pr > 1])[-1]
            # long_cumprod_loss = np.cumprod(long_np_pr[long_np_pr < 1])[-1]
            # long_pr_tr = long_cumprod_profit * long_cumprod_loss

            np_long_tp_ratio_list = np.array(long_tp_ratio_list)
            mean_long_tr = np.mean(np_long_tp_ratio_list[np.isnan(np_long_tp_ratio_list) == 0])

            np_long_dr_list = np.array(long_dr_list)
            mean_long_dr = np.mean(np_long_dr_list[np.isnan(np_long_dr_list) == 0])

            # long_pr_gap = (long_np_pr - 1) / config.lvrg_set.leverage + fee
            # long_tp_gap = long_pr_gap[long_pr_gap > 0]
            # # mean_long_tp_gap = np.mean(long_pr_gap[long_pr_gap > 0])
            # # mean_long_ls_gap = np.mean(long_pr_gap[long_pr_gap < 0])

            # mean_long_pgfr = np.mean((long_tp_gap - fee) / abs(long_tp_gap + fee))

            plt.subplot(233)
            plt.plot(total_long_pr)
            plt.plot(long_sum_pr, color='gold')
            if len(nontp_long_liqd_list) != 0:

                max_nontp_long_term = len(res_df) - nontp_long_indexs[0]

                plt.title("wr : %.3f\nlen(long_td) : %s\nmin_pr : %.3f\nacc_pr : %.3f\n sum_pr : %.3f"
                          % (long_wr, len(long_np_pr[long_np_pr != 1]), np.min(long_np_pr), total_long_pr[-1],
                             long_sum_pr[-1]) + \
                          "\n leverage %s\nliqd : %.3f\nmean_tr : %.3f\n mean_dr : %.3f\n nontp_long_liqd_cnt : %s\nnontp_long_liqd : %.3f\nontp_long_liqd_pr : %.3f\nmax_nontp_long_term : %s\n tw cw tls cls : %s %s %s %s"
                          % (config.lvrg_set.leverage, min(long_liqd_list), mean_long_tr, mean_long_dr,
                             len(nontp_long_liqd_list), min(nontp_long_liqd_list), min(nontp_long_pr_list),
                             max_nontp_long_term, t_w_l, c_w_l, t_ls_l, c_ls_l))
            else:
                plt.title("wr : %.3f\nlen(long_td) : %s\nmin_pr : %.3f\nacc_pr : %.3f\n sum_pr : %.3f"
                          % (long_wr, len(long_np_pr[long_np_pr != 1]), np.min(long_np_pr), total_long_pr[-1],
                             long_sum_pr[-1]) + \
                          "\n leverage %s\nliqd : %.3f\nmean_tr : %.3f\n mean_dr : %.3f\n nontp_long_liqd_cnt : %s\n tw cw tls cls : %s %s %s %s"
                          % (config.lvrg_set.leverage, min(long_liqd_list), mean_long_tr, mean_long_dr,
                             len(nontp_long_liqd_list), t_w_l, c_w_l, t_ls_l, c_ls_l))

            long_survey_df.iloc[survey_i] = long_wr, len(long_np_pr[long_np_pr != 1]), np.min(long_np_pr), \
                                            total_long_pr[-1], long_sum_pr[-1], min(
                long_liqd_list), mean_long_tr, mean_long_dr

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

            rev_total_pr = np.cumprod(rev_np_pr)
            rev_wr = len(rev_np_pr[rev_np_pr > 1]) / len(rev_np_pr[rev_np_pr != 1])

            rev_total_for_sum_pr = rev_np_pr - 1
            rev_total_for_sum_pr[0] = 1
            rev_total_sum_pr = np.cumsum(rev_total_for_sum_pr)
            rev_total_sum_pr = np.where(rev_total_sum_pr < 0, 0, rev_total_sum_pr)

            # plt.subplot(122)
            plt.subplot(234)
            plt.plot(rev_total_pr)
            plt.plot(rev_total_sum_pr, color='gold')

            plt.title("wr : %.3f\nmin_pr : %.3f\nacc_pr : %.3f\n sum_pr : %.3f" % (
            rev_wr, np.min(rev_np_pr), rev_total_pr[-1], rev_total_sum_pr[-1]))

            rev_survey_df.iloc[survey_i] = rev_wr, np.min(rev_np_pr), rev_total_pr[-1], rev_total_sum_pr[-1]

        except Exception as e:
            print("error in 234 :", e)

        try:
            #         short       #
            # rev_short_np_pr = 1 / (np.array(short_list) + fee) - fee
            rev_short_fee = tp_fee + out_fee - np.array(short_fee_list)
            rev_short_np_pr = (1 / ((np.array(short_list) - 1) / config.lvrg_set.leverage + np.array(
                short_fee_list) + 1) - rev_short_fee - 1) * config.lvrg_set.leverage + 1
            # rev_short_np_pr = (1 / (np.array(short_list) + fee) - fee - 1) * config.lvrg_set.leverage + 1

            rev_total_short_pr = np.cumprod(rev_short_np_pr)
            rev_short_wr = len(rev_short_np_pr[rev_short_np_pr > 1]) / len(rev_short_np_pr[rev_short_np_pr != 1])

            rev_short_for_sum_pr = rev_short_np_pr - 1
            rev_short_for_sum_pr[0] = 1
            rev_short_sum_pr = np.cumsum(rev_short_for_sum_pr)
            rev_short_sum_pr = np.where(rev_short_sum_pr < 0, 0, rev_short_sum_pr)

            # plt.subplot(122)
            plt.subplot(235)
            plt.plot(rev_total_short_pr)
            plt.plot(rev_short_sum_pr, color='gold')

            plt.title("wr : %.3f\nmin_pr : %.3f\nacc_pr : %.3f\n sum_pr : %.3f" % (
            rev_short_wr, np.min(rev_short_np_pr), rev_total_short_pr[-1], rev_short_sum_pr[-1]))

            rev_short_survey_df.iloc[survey_i] = rev_short_wr, np.min(rev_short_np_pr), rev_total_short_pr[-1], \
                                                 rev_short_sum_pr[-1]

        except Exception as e:
            print("error in 235 :", e)

        try:
            #         long       #
            # rev_long_np_pr = 1 / (np.array(long_list) + fee) - fee
            rev_long_fee = tp_fee + out_fee - np.array(long_fee_list)
            rev_long_np_pr = (1 / ((np.array(long_list) - 1) / config.lvrg_set.leverage + np.array(
                long_fee_list) + 1) - rev_long_fee - 1) * config.lvrg_set.leverage + 1

            rev_total_long_pr = np.cumprod(rev_long_np_pr)
            rev_long_wr = len(rev_long_np_pr[rev_long_np_pr > 1]) / len(rev_long_np_pr[rev_long_np_pr != 1])

            rev_long_for_sum_pr = rev_long_np_pr - 1
            rev_long_for_sum_pr[0] = 1
            rev_long_sum_pr = np.cumsum(rev_long_for_sum_pr)
            rev_long_sum_pr = np.where(rev_long_sum_pr < 0, 0, rev_long_sum_pr)

            # plt.subplot(122)
            plt.subplot(236)
            plt.plot(rev_total_long_pr)
            plt.plot(rev_long_sum_pr, color='gold')

            plt.title("wr : %.3f\nmin_pr : %.3f\nacc_pr : %.3f\n sum_pr : %.3f" % (
            rev_long_wr, np.min(rev_long_np_pr), rev_total_long_pr[-1], rev_long_sum_pr[-1]))

            rev_long_survey_df.iloc[survey_i] = rev_long_wr, np.min(rev_long_np_pr), rev_total_long_pr[-1], \
                                                rev_long_sum_pr[-1]

        except Exception as e:
            print("error in 236 :", e)

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

                # plt.subplot(121)
                plt.subplot(231)
                plt.plot(h_total_pr)
                plt.title("wr : %.3f\nmin_pr : %.3f\nacc_pr : %.3f\n leverage %s" % (
                h_wr, np.min(h_np_pr), h_total_pr[-1], config.lvrg_set.leverage))
                # plt.show()

                #         short only      #
                h_short_np_pr = np.array(h_short_list)

                h_total_short_pr = np.cumprod(h_short_np_pr)
                h_short_wr = len(h_short_np_pr[h_short_np_pr > 1]) / len(h_short_np_pr[h_short_np_pr != 1])

                plt.subplot(232)
                plt.plot(h_total_short_pr)
                plt.title("wr : %.3f\nmin_pr : %.3f\nacc_pr : %.3f\n leverage %s" % (
                h_short_wr, np.min(h_short_np_pr), h_total_short_pr[-1], config.lvrg_set.leverage))

                #         long only      #
                h_long_np_pr = np.array(h_long_list)

                h_total_long_pr = np.cumprod(h_long_np_pr)
                h_long_wr = len(h_long_np_pr[h_long_np_pr > 1]) / len(h_long_np_pr[h_long_np_pr != 1])

                plt.subplot(233)
                plt.plot(h_total_long_pr)
                plt.title("wr : %.3f\nmin_pr : %.3f\nacc_pr : %.3f\n leverage %s" % (
                h_long_wr, np.min(h_long_np_pr), h_total_long_pr[-1], config.lvrg_set.leverage))

                #     reversion adjustment      #

                h_rev_total_pr = np.cumprod(h_rev_np_pr)
                h_rev_wr = len(h_rev_np_pr[h_rev_np_pr > 1]) / len(h_rev_np_pr[h_rev_np_pr != 1])

                # plt.subplot(122)
                plt.subplot(234)
                plt.plot(h_rev_total_pr)
                plt.title("wr : %.3f\nmin_pr : %.3f\nacc_pr : %.3f\n leverage %s" % (
                h_rev_wr, np.min(h_rev_np_pr), h_rev_total_pr[-1], config.lvrg_set.leverage))

                #         short       #
                # h_rev_short_np_pr = 1 / (np.array(h_short_list) + fee) - fee
                h_rev_short_np_pr = (1 / ((np.array(h_short_list) - 1) / config.lvrg_set.leverage + np.array(
                    short_fee_list) + 1) - np.array(short_fee_list) - 1) * config.lvrg_set.leverage + 1

                h_rev_total_short_pr = np.cumprod(h_rev_short_np_pr)
                h_rev_short_wr = len(h_rev_short_np_pr[h_rev_short_np_pr > 1]) / len(
                    h_rev_short_np_pr[h_rev_short_np_pr != 1])

                # plt.subplot(122)
                plt.subplot(235)
                plt.plot(h_rev_total_short_pr)
                plt.title("wr : %.3f\nmin_pr : %.3f\nacc_pr : %.3f\n leverage %s" % (
                h_rev_short_wr, np.min(h_rev_short_np_pr), h_rev_total_short_pr[-1], config.lvrg_set.leverage))

                #         long       #
                # h_rev_long_np_pr = 1 / (np.array(h_long_list) + fee) - fee
                h_rev_long_np_pr = (1 / ((np.array(h_long_list) - 1) / config.lvrg_set.leverage + np.array(
                    long_fee_list) + 1) - np.array(long_fee_list) - 1) * config.lvrg_set.leverage + 1

                h_rev_total_long_pr = np.cumprod(h_rev_long_np_pr)
                h_rev_long_wr = len(h_rev_long_np_pr[h_rev_long_np_pr > 1]) / len(
                    h_rev_long_np_pr[h_rev_long_np_pr != 1])

                # plt.subplot(122)
                plt.subplot(236)
                plt.plot(h_rev_total_long_pr)
                plt.title("wr : %.3f\nmin_pr : %.3f\nacc_pr : %.3f\n leverage %s" % (
                h_rev_long_wr, np.min(h_rev_long_np_pr), h_rev_total_long_pr[-1], config.lvrg_set.leverage))

                plt.show()

        # break # pair loop

        except Exception as e:
            print('error in pr plot :', e)

        print()

        break  # indi. loop

    plt.savefig(dir_path + "/back_pr/back_pr.png")
    print("back_pr.png saved !")

    return ep_tp_list, trade_list
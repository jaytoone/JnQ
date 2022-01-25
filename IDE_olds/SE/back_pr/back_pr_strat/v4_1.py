#       Todo        #
#        1. back_id ajustment
#         a. remove "key" variable
#        2. define as a function
#        3. adds some libraries
#        4. plt.show() off
#        5. title_position & tight_layout

import numpy as np
from funcs.funcs_indicator_candlescore import *
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

    # for senkou_a, senkou_b in zip(senkoua_list, senkoub_list):
    for pgfr in np.arange(0.3, 1, 0.1):
        # for entry_incycle in range(0, 5):
        # for tp_gap in np.arange(1., 2.5, 0.2):
        # for out_gap in np.arange(0.25, 1., 0.05):
        # for cloud_lookback in np.arange(5, 100, 3):
        # tp_gap = 1
        # pgfr = 0.7
        print("out_gap :", out_gap)
        print("tp_gap :", tp_gap)
        print("pgfr :", pgfr)
        print("entry_incycle :", entry_incycle)

        # print("sma4_period :", sma4_period)
        # print("cloud_lookback :", cloud_lookback)

        # -------------------- additional indicators -------------------- #

        #           ichimoku            #
        # res_df['senkou_a1'], res_df['senkou_b1'] = ichimoku(res_df)

        # # second_df['senkou_a'], second_df['senkou_b'] = ichimoku(second_df)
        # # res_df = res_df.join( pd.DataFrame(index=res_df.index, data=to_lower_tf(res_df, second_df, [-2, -1]), columns=['senkou_a2', 'senkou_b2']))

        # # #           cloud displacement           #
        # cloud_cnt = 0
        # for col_n in res_df.columns:
        #   if 'senkou' in col_n:
        #     cloud_cnt += 1
        # print("cloud_cnt :", cloud_cnt)

        # res_df.iloc[:, -cloud_cnt:] = res_df.iloc[:, -cloud_cnt:].shift(26 - 1)

        # senkou_a, senkou_b = 'senkou_a1', 'senkou_b1'

        # cloud_top = np.max(res_df[[senkou_a, senkou_b]], axis=1)
        # cloud_bottom = np.min(res_df[[senkou_a, senkou_b]], axis=1)

        # under_top = res_df['close'].shift(cloud_shift_size) <= cloud_top.shift(cloud_shift_size)
        # over_top = res_df['close'].shift(cloud_shift_size) >= cloud_top.shift(cloud_shift_size)

        # over_bottom = res_df['close'].shift(cloud_shift_size) >= cloud_bottom.shift(cloud_shift_size)
        # under_bottom = res_df['close'].shift(cloud_shift_size) >= cloud_bottom.shift(cloud_shift_size)

        # --------------- sma --------------- #
        # res_df['sma_1m'] = res_df['close'].rolling(sma_period).mean()
        res_df['sma_1m'] = res_df['close'].rolling(60).mean()

        # --------------- ema --------------- #
        res_df['ema5_1m'] = ema(res_df['close'], 5).shift(1)

        # --------------- cloud bline --------------- #
        res_df['cloud_bline_1m'] = cloud_bline(res_df, 26).shift(1)

        # ------------------------------ htf data ------------------------------ #
        # second_df = pd.read_excel(date_path2 + key.replace("_bbline_backi2", ""), index_col=0)
        # # third_df = pd.read_excel(date_path3 + key.replace("_bbline_backi2", ""), index_col=0)
        # # fourth_df = pd.read_excel(date_path4 + key.replace("_4h1d_backi2", ""), index_col=0)
        # # fifth_df = pd.read_excel(date_path5 + key.replace("_multipline_backi2", ""), index_col=0)
        #
        # # --------------- cloud bline --------------- #
        # second_df['cloud_bline_3m'] = cloud_bline(second_df, 26)
        # res_df = res_df.join(
        #     pd.DataFrame(index=res_df.index, data=to_lower_tf(res_df, second_df, [-1]), columns=['cloud_bline_3m']))
        # third_df['cloud_bline_5m'] = cloud_bline(third_df, 26)
        # res_df = res_df.join(pd.DataFrame(index=res_df.index, data=to_lower_tf(res_df, third_df, [-1]), columns=['cloud_bline_5m']))

        # ---------------------------------------- short = -1 ---------------------------------------- #

        # --------------- timestamp entry --------------- #
        # entry = np.where((intmin(res_df.index) in [0, 30])
        #                       , -1, 0)

        # # --------------- ema_roc entry --------------- #
        # entry = np.where((res_df['ema_roc'].shift(1) >= 0) &
        #                       (res_df['ema_roc'] < 0)
        #                       , -1, 0)

        # --------------- st entry --------------- #
        # entry = np.where((res_df['close'].shift(1) <= short_ep) &
        #                       (res_df['high'] >= short_ep)
        #                       , -1, 0)
        # # entry = np.where((res_df['high'].shift(1) <= upper_middle) &
        # # entry = np.where((res_df['high'].shift(1) <= res_df['middle_line']) &
        # #                     (res_df['high'] >= short_ep)
        # #                     , -1, 0)

        # entry = np.where((res_df['close'].shift(1) > short_ep)
        #                   , -2, entry)

        # --------------- st level entry --------------- #
        entry = np.where((res_df['close'].shift(1) >= res_df['st_lower_5m']) &
                         (res_df['close'] < res_df['st_lower_5m'])
                         , -1, 0)

        # entry = np.where((res_df['close'].shift(1) >= res_df['lower_middle5']) &
        #                 (res_df['close'] < res_df['lower_middle5'])
        #                 , -1, 0)

        # --------------- bb level entry --------------- #
        # entry = np.where((res_df['close'].shift(1) >= res_df['bb_lower_1m']) &
        #                 (res_df['close'] < res_df['bb_lower_1m'])
        #                 , -1, 0)
        # entry = np.where((res_df['close'].shift(1) >= res_df['bb_lower_1m'].shift(1)) &
        #                 (res_df['close'] < res_df['bb_lower_1m'])
        #                 , -1, entry)

        # --------------- cbline rolling entry --------------- #
        # entry = np.where((res_df['close'].shift(1) >= res_df['cloud_bline1']) &
        #                 (res_df['close'] < res_df['cloud_bline1'])
        #                 , -1, 0)

        # --------------- ema rolling entry --------------- #
        # entry = np.where((res_df['close'].shift(1) >= res_df['ema1']) &
        #                 # (long_ep <= res_df['high']) &
        #                 (res_df['close'] < res_df['ema1'])
        #                 , -1, 0)

        # --------------- mmh_st rolling ep --------------- #
        # entry = np.where((res_df['close'].shift(1) >= res_df['mmh_st2'].shift(1)) &
        #                 # (long_ep <= res_df['high']) &
        #                 (res_df['close'] < res_df['mmh_st2']) &
        #                 # (res_df['norm_st_trend'] == 1)
        #                 (res_df['norm_st_trend'] == -1)
        #                 , -1, 0)

        # print("len(entry[entry==-1]) :", len(entry[entry==-1]))
        # break

        # ---------------------------------------- long = 1 ---------------------------------------- #

        # --------------- timestamp entry --------------- #
        # entry = np.where((np.array(intmin(res_df.index)) in [0, 30])

        # int_min_ts = np.array(list(map(lambda x :intmin(x), res_df.index)))
        # # entry = np.where((intmin(res_df.index) == 0)
        # entry = np.where((int_min_ts == 0) | (int_min_ts == 30)
        #                       , 1, 0)

        # print("int_min_ts :", int_min_ts)
        # print("entry :", entry)
        # break

        # --------------- ema_roc entry --------------- #
        # entry = np.where((res_df['ema_roc'].shift(1) <= 0) &
        #                       (res_df['ema_roc'] > 0)
        #                       , 1, entry)

        # --------------- st entry --------------- #
        # entry = np.where((res_df['close'].shift(1) >= long_ep) &
        #                   (res_df['low'] <= long_ep)
        #                   , 1, entry)
        # # entry = np.where((res_df['low'].shift(1) >= lower_middle) &
        # # entry = np.where((res_df['low'].shift(1) >= res_df['middle_line']) &
        # #                   (res_df['low'] <= long_ep)
        # #                   , 1, entry)

        # entry = np.where((res_df['close'].shift(1) < long_ep)
        #                   , 2, entry)

        # entry = np.where((res_df['close'].shift(1) <= long_ep) &
        #                   # (long_ep <= res_df['high']) &
        #                   (res_df['close'] >= long_ep)
        #                   , 1, entry)

        # --------------- st level entry --------------- #
        entry = np.where((res_df['close'].shift(1) <= res_df['st_upper_5m']) &
                         (res_df['close'] > res_df['st_upper_5m'])
                         , 1, entry)

        # entry = np.where((res_df['close'].shift(1) <= res_df['upper_middle5']) &
        #                   (res_df['close'] > res_df['upper_middle5'])
        #                   , 1, entry)

        # --------------- bb level entry --------------- #
        # entry = np.where((res_df['close'].shift(1) <= res_df['bb_upper_1m']) &
        #                   (res_df['close'] > res_df['bb_upper_1m'])
        #                   , 1, entry)

        # entry = np.where((res_df['close'].shift(1) <= res_df['bb_upper_1m'].shift(1)) &
        #                   (res_df['close'] > res_df['bb_upper_1m'])
        #                   , 1, entry)

        # --------------- cbline rolling entry --------------- #
        # entry = np.where((res_df['close'].shift(1) <= res_df['cloud_bline1']) &
        #                   (res_df['close'] > res_df['cloud_bline1'])
        #                   , 1, entry)

        # --------------- ema rolling entry --------------- #
        # entry = np.where((res_df['close'].shift(1) <= res_df['ema1']) &
        #                   (res_df['close'] > res_df['ema1'])
        #                   , 1, entry)

        # --------------- mmh_st rolling entry --------------- #
        # entry = np.where((res_df['close'].shift(1) <= res_df['mmh_st2'].shift(1)) &
        #                 # (long_ep <= res_df['high']) &
        #                 (res_df['close'] > res_df['mmh_st2']) &
        #                 # (res_df['norm_st_trend'] == -1)
        #                 (res_df['norm_st_trend'] == 1)
        #                 , 1, entry)

        #       1-2. tp line = middle line 조금 이내         #

        # ------------------------------ out ------------------------------ #

        # --------------- st level consolidation out --------------- #
        short_out = res_df['st_upper2_5m']
        long_out = res_df['st_lower2_5m']

        # --------------- bb level consolidation out --------------- #
        # short_out = res_df['bb_upper_1m']
        # long_out = res_df['bb_lower_1m']

        # --------------- cloud_bline rolling out --------------- #
        # short_out = res_df['cloud_bline1']
        # long_out = res_df['cloud_bline1']

        # --------------- ema rolling out --------------- #
        # short_out = res_df['ema1_5']
        # long_out = res_df['ema1_5']

        # --------------- mmh_st rolling out --------------- #
        # short_out = res_df['mmh_st1']
        # long_out = res_df['mmh_st1']

        # --------------- st rolling out --------------- #

        # short_out = res_df['lower_middle%s' % basic_st_number]
        # long_out = res_df['upper_middle%s' % basic_st_number]

        # short_out = res_df['upper_middle%s' % basic_st_number]
        # long_out = res_df['lower_middle%s' % basic_st_number]

        # short_out = short_ep + res_df['st_gap%s' % basic_st_number] * out_gap
        # long_out = long_ep - res_df['st_gap%s' % basic_st_number] * out_gap

        # short_out2 = res_df['lower_middle]
        # long_out2 = res_df['upper_middle]

        # short_out = short_ep + res_df['st_gap']
        # long_out = long_ep - res_df['st_gap']

        # ------------------------------ tp ------------------------------ #
        # --------------- st level tp --------------- #
        short_tp = res_df['st_lower3_5m'] - res_df['st_gap_5m'] * tp_gap
        long_tp = res_df['st_upper3_5m'] + res_df['st_gap_5m'] * tp_gap

        # --------------- bb level tp --------------- #
        # short_tp = res_df['close'] - res_df['bb_gap_1m'] * tp_gap
        # long_tp = res_df['close'] + res_df['bb_gap_1m'] * tp_gap

        # --------------- mmh_st tp --------------- #

        # short_tp = res_df['close'] - tp_gap * abs(res_df['mmh_st1'] - res_df['close'])
        # long_tp = res_df['close'] + tp_gap * abs(res_df['mmh_st1'] - res_df['close'])

        # short_tp2 = res_df['close'] - tp_gap / 2 * abs(res_df['mmh_st1'] - res_df['close'])
        # long_tp2 = res_df['close'] + tp_gap / 2 * abs(res_df['mmh_st1'] - res_df['close'])

        # --------------- st rolling tp --------------- #
        # short_tp = res_df['min_lower%s' % basic_st_number]
        # long_tp = res_df['max_upper%s' % basic_st_number]

        # short_tp = short_ep - res_df['st_gap%s' % basic_st_number] * tp_gap
        # long_tp = long_ep + res_df['st_gap%s' % basic_st_number] * tp_gap

        if inversion:
            temp_short_out = short_out
            temp_long_out = long_out

            short_out = short_tp
            long_out = long_tp

            short_tp = temp_short_out
            long_tp = temp_long_out

        # --------------- set partial tp --------------- #

        short_tps = [short_tp]
        long_tps = [long_tp]

        # short_tps = [short_tp2]
        # long_tps = [long_tp2]

        # short_tps = [short_tp2, short_tp] # org
        # long_tps = [long_tp2, long_tp]

        # short_tps = [short_tp, short_tp2]
        # long_tps = [long_tp, long_tp2]

        #       trading : 여기도 체결 결과에 대해 묘사함       #
        trade_list = []
        h_trade_list = []
        lvrg_list = []
        open_list = []

        tp_ratio_list = []
        short_tp_ratio_list = []
        long_tp_ratio_list = []

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

            if entry[i] in short_entry:

                initial_i = i

                if static_out:
                    p_i = initial_i
                else:
                    p_i = i

                # -------------- ep scheduling -------------- #
                # ----- st_gap ----- #
                if res_df['close'].iloc[i] >= res_df['st_lower2_5m'].iloc[i]:
                    # if res_df['close'].iloc[i] <= res_df['lower_middle3'].iloc[i] and \
                    # abs(res_df['close'].iloc[i] - res_df['bb_lower'].iloc[i]) < ep_protect_gap * res_df['st_gap3'].iloc[i]:
                    # # abs(res_df['close'].iloc[i] - res_df['lower_middle3'].iloc[i]) < ep_protect_gap * res_df['st_gap3'].iloc[i]:
                    # # abs(res_df['close'].iloc[i] - short_ep.iloc[i]) < ep_protect_gap * res_df['st_gap'].iloc[i]:
                    # # if abs((res_df['close'].iloc[i] - upper_middle.iloc[i]) / upper_middle.iloc[i]) < ep_protect_gap:
                    # # if abs((res_df['close'].iloc[i] - res_df['middle_line'].iloc[i]) / res_df['middle_line'].iloc[i]) < ep_protect_gap:
                    # # if res_df['close'].shift(1).iloc[i] <= short_out.iloc[p_i] - ep_protect_gap * res_df['st_gap%s' % basic_st_number].iloc[p_i]:

                    # ----- bb_gap ----- #
                    # if abs(res_df['close'].iloc[i] - res_df['bb_lower_1m'].iloc[i]) < ep_protect_gap * bb_gap.iloc[i]:

                    pass
                else:
                    i += 1
                    if i >= len(res_df):
                        break
                    continue

                # -------------- tp scheduling -------------- #
                # if bb_gap.iloc[i] / res_df['close'].iloc[i] > fee:

                # ---- pgfr : profit_gap & fee ratio --- #
                # if (bb_gap.iloc[i] / res_df['close'].iloc[i] - fee) / abs(bb_gap.iloc[i] / res_df['close'].iloc[i] + fee) > pgfr:
                if (short_tp.iloc[i] / res_df['close'].iloc[i] - fee) / abs(
                        short_tp.iloc[i] / res_df['close'].iloc[i] + fee) > pgfr:

                    pass
                else:
                    i += 1
                    if i >= len(res_df):
                        break
                    continue

                # -------------- markup only minor -------------- #
                # ------- entry once ------- #
                #          1. find prev out         #
                #          2. count entry til prev out         #
                # prev_entry_cnt = 0
                # for back_i in range(i - 1, 0, -1):
                #   if entry[back_i] == 1:
                #     break

                #   elif entry[back_i] == -1:
                #     prev_entry_cnt += 1

                # # print("prev_entry_cnt :", prev_entry_cnt)
                # if prev_entry_cnt <= entry_incycle:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # ------- ema alignment ------- #
                # if res_df['ema5_1m'].iloc[i] < res_df['ema5_1m'].iloc[i - 1] < res_df['ema5_1m'].iloc[i - 2]:
                # # if res_df['ema5_1m'].iloc[i - 1] < res_df['ema5_1m'].iloc[i - 2]:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                #   # ------- sma alignment ------- #
                # if res_df['sma_1m'].iloc[i] < res_df['sma_1m'].iloc[i - 1]:
                # # if res_df['sma_1m'].iloc[i] < res_df['sma_1m'].iloc[i - 1] < res_df['sma_1m'].iloc[i - 2]:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                #   # ------- bb alignment ------- #
                # if res_df['bb_lower_1m'].iloc[i] < res_df['bb_lower_1m'].iloc[i - 1]:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # ------- bb_base alignment ------- #
                # if res_df['bb_base_1m'].iloc[i] < res_df['bb_base_1m'].iloc[i - 1]:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # ------------------- markup only major ------------------- #

                # -------------- ma const. -------------- #
                # ------- ema const. ------- #
                # if res_df['close'].shift(0).iloc[i] < res_df['ema5'].shift(0).iloc[i]: # and \
                # if res_df['close'].shift(1).iloc[i] < res_df['ema5'].shift(1).iloc[i]: # and \
                # if res_df['close'].shift(1).iloc[i] < res_df['ema5'].shift(0).iloc[i]: # and \
                # if short_ep.iloc[i] < res_df['ema5'].shift(0).iloc[i]: # and \
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # ------- sma const. ------- #
                # if res_df['close'].iloc[i] <= res_df['sma_1m'].iloc[i]: # and \
                # # if res_df['close'].shift(1).iloc[i] < res_df['sma_1m'].shift(1).iloc[i]: # and \
                # #   short_ep.iloc[i] <= res_df['sma_1m'].shift(sma_shift_size).iloc[i]:
                # # # under_sma = short_ep <= res_df['sma'].shift(sma_shift_size)
                # # # if np.sum(under_sma.iloc[i + 1 - sma_lookback:i + 1]) == sma_lookback:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # ------- bb zone const. ------- #
                # if res_df['bb_lower_1m'].iloc[i] < res_df['bb_lower_30m'].iloc[i]:
                # if res_df['close'].iloc[i] <= res_df['bb_lower_30m'].iloc[i]:
                # # if res_df['bb_lower_1m'].iloc[i] < res_df['bb_lower2_30m'].iloc[i]:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # ------- cloud bline ------- #
                # if res_df['close'].iloc[i] <= res_df['cloud_bline_5m'].iloc[i]:
                if res_df['close'].iloc[i] <= res_df['cloud_bline_3m'].iloc[i]:
                    pass
                else:
                    i += 1
                    if i >= len(res_df):
                        break
                    continue

                # -------------- distance protection -------------- #
                # tp_dist = (res_df['close'].iloc[i] - short_tp.iloc[i])
                # out_dist = (res_df['middle_line'].iloc[i] - res_df['close'].iloc[i])
                # if tp_dist / out_dist >= tp_out_ratio:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # -------------- feature dist const. -------------- #
                # if initial_i < input_size:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # entry_input_x = min_max_scale(res_df[selected_price_colname].iloc[initial_i - input_size:initial_i].values)

                # re_entry_input_x = expand_dims(entry_input_x)

                # entry_vector = model.predict(re_entry_input_x, verbose=0)
                # # print(test_result.shape)

                # f_dist = vector_dist(entry_vector, selected_vector)
                # print("f_dist :", f_dist)

                # if f_dist < fdist_thresh:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # -------------- htf data const. -------------- #
                # i_min = intmin(res_df.index[i]) # 2020-09-05 00:00:59.999000
                # if i_min >= 30:
                #   htf_ts = str(res_df.index[i])[:-12] + "59:59.999000"
                # else:
                #   htf_ts = str(res_df.index[i])[:-12] + "29:59.999000"

                #   # -------------- ema -------------- #
                # if fifth_df['close'].shift(1).loc[htf_ts] < fifth_df['ema'].shift(1).loc[htf_ts]:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # -------------- fisher const. -------------- #
                # if res_df['fisher30'].shift(1).iloc[i] < 0:
                # if res_df['fisher60'].shift(1).iloc[i] < 0:
                # if res_df['fisher120'].shift(1).iloc[i] < 0:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # -------------- htf st line const. -------------- #
                # ------------ 30m ------------ #
                # # if res_df['lower_middle5'].iloc[i] >= res_df['min_upper2'].iloc[i]:
                # # if res_df['lower_middle5'].iloc[i] >= res_df['close'].iloc[i]:
                # if res_df['bb_lower'].iloc[i] >= res_df['lower_middle5'].iloc[i] and \
                # res_df['bb_lower'].iloc[i] < res_df['ema1_5'].iloc[i]:

                #   # ------------ 4h ------------ #
                # # if res_df['lower_middle6'].iloc[i] >= res_df['min_upper2'].iloc[i]:

                #   # ------------ 5m & 30m ------------ #
                # # if res_df['lower_middle5'].iloc[i] >= res_df['middle_line3'].iloc[i]:
                # # if res_df['lower_middle5'].iloc[i] >= res_df['upper_middle3'].iloc[i]:

                #   # ------------ 5m & 30m & 4h ------------ #
                # # if res_df['lower_middle6'].iloc[i] >= res_df['lower_middle5'].iloc[i] >= res_df['lower_middle3'].iloc[i]:

                #     pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # -------------- 1d sma const. -------------- #
                # if res_df[sma].iloc[i] >= res_df['close'].iloc[i]:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # -------------- cloud lb const.-------------- #
                # if i < cloud_lookback:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # if np.sum(under_top.iloc[i + 1 - cloud_lookback:i + 1]) == cloud_lookback:
                # # if np.sum(under_bottom.iloc[i + 1 - cloud_lookback:i + 1]) == cloud_lookback:
                # # if np.sum(over_top.iloc[i + 1 - cloud_lookback:i + 1]) == cloud_lookback:
                # # if np.sum(under_top.iloc[i - cloud_lookback:i]) == cloud_lookback:
                #   pass

                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # -------------- cloud color const.-------------- #
                #               1. senkou_a1 < senkou_b1            #
                #               1-1. mutli clouds color 충분히 고려               #
                # if i < cloud_lookback:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # if np.sum(res_df[senkou_a].shift(cloud_shift_size).iloc[i + 1 - cloud_lookback:i + 1] <= res_df[senkou_b].shift(cloud_shift_size).iloc[i + 1 - cloud_lookback:i + 1]) == cloud_lookback and \
                #   np.sum(res_df["senkou_a2"].shift(0).iloc[i + 1 - cloud_lookback:i + 1] <= res_df["senkou_b2"].shift(0).iloc[i + 1 - cloud_lookback:i + 1]) == cloud_lookback: # and \
                #   # np.sum(res_df["senkou_a3"].shift(0).iloc[i + 1 - cloud_lookback:i + 1] <= res_df["senkou_b3"].shift(0).iloc[i + 1 - cloud_lookback:i + 1]) == cloud_lookback and \
                #   # np.sum(res_df["senkou_a4"].shift(0).iloc[i + 1 - cloud_lookback:i + 1] <= res_df["senkou_b4"].shift(0).iloc[i + 1 - cloud_lookback:i + 1]) == cloud_lookback and \
                #   # np.sum(res_df["senkou_a5"].shift(0).iloc[i + 1 - cloud_lookback:i + 1] <= res_df["senkou_b5"].shift(0).iloc[i + 1 - cloud_lookback:i + 1]) == cloud_lookback:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # -------------- st color const.-------------- #
                # if np.sum(res_df[['ST1_Trend%s' % basic_st_number, 'ST2_Trend%s' % basic_st_number, 'ST3_Trend%s' % basic_st_number]].iloc[i]) <= -1:
                # # if np.sum(res_df[['ST1_Trend%s' % basic_st_number, 'ST2_Trend%s' % basic_st_number, 'ST3_Trend%s' % basic_st_number]].iloc[i]) <= -3:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # -------------- htf st color const.-------------- #
                # if np.sum(res_df[['major_ST1_Trend', 'major_ST2_Trend', 'major_ST3_Trend']].iloc[i]) <= -1:
                # # if np.sum(res_df[['major_ST1_Trend', 'major_ST2_Trend', 'major_ST3_Trend']].iloc[i]) <= -3:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # -------------- 3rd st const. : st should have 2, 3 or more -------------- #
                # if np.sum(res_df[['minor_ST2_Trend', 'minor_ST3_Trend']].iloc[i]) <= -2:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # -------------- st gap const.-------------- #
                # if abs(res_df['ST2_Up%s' % basic_st_number].iloc[i] - res_df['ST1_Up%s' % basic_st_number].iloc[i]) <= approval_st_gap * res_df['st_gap%s' % basic_st_number].iloc[i]:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # -------------- sar const. -------------- #
                # if res_df['sar1'].iloc[i] > res_df['high'].iloc[i]:
                # # if res_df['sar2'].iloc[i] > res_df['high'].iloc[i] and res_df['sar3'].iloc[i] > res_df['high'].iloc[i]:
                # # if res_df['sar2'].iloc[i] > res_df['high'].iloc[i]: # and \
                # # if  res_df['sar3'].iloc[i] > res_df['high'].iloc[i]:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                ep_j = initial_i  # dynamic ep 를 위한 var.

                # -------------- limit waiting const. -------------- #

                # entry_done = False
                # prev_sar = None
                # # for e_j in range(i, len(res_df)):
                # for e_j in range(i + 1, len(res_df)): # entry signal이 close 기준 일 경우

                #   if not static_ep:
                #     ep_j = e_j

                #   #             Todo            #
                #   #             1. ep 설정

                #   # -------------- np.inf ep -------------- #
                #   # if short_ep.iloc[initial_i] == np.inf:
                #   #   break

                #   #             1-1. close 가 sar_change 이전 sar 을 cross 한 경우만 진입

                #   # -------------- ep scheduling -------------- #
                #   # if res_df['close'].shift(1).iloc[e_j] <= short_out.iloc[ep_j] - ep_protect_gap * res_df['st_gap%s' % basic_st_number].iloc[ep_j]:
                #   #   pass
                #   # else:
                #   #   continue

                #   #             1-0. ep touch, entry_done       #
                #   # if res_df['high'].iloc[e_j] >= short_ep.iloc[initial_i] and res_df['close'].shift(1).iloc[e_j] <= short_ep.iloc[initial_i]:
                #   # if res_df['high'].iloc[e_j] >= short_ep.iloc[initial_i] and res_df['close'].shift(1).iloc[e_j] >= short_out.iloc[initial_i] - res_df['st_gap'].iloc[initial_i] * ep_out_gap: # good tr
                #   # if res_df['high'].iloc[e_j] >= short_ep.iloc[initial_i] and res_df['close'].shift(1).iloc[e_j] <= short_out.iloc[initial_i] - res_df['st_gap'].iloc[initial_i] * ep_out_gap: # --> 이게 의도한거자나
                #   if res_df['high'].iloc[e_j] >= short_ep.iloc[ep_j]:
                #     entry_done = True
                #     # print("res_df['high'].iloc[e_j] :", res_df['high'].iloc[e_j])
                #     # print("e_j :", e_j)

                #     #     이미, e_j open 이 ep 보다 높은 경우, entry[ep_j] => -2 로 변경   #
                #     if res_df['open'].iloc[e_j] >= short_ep.iloc[ep_j]:
                #       entry[initial_i] = -2

                #     break

                #   #             2. limit 대기 시간 설정
                #   #             2-1. tp 조건이 성립되는 경우 limit 취소
                #   # -------------- st tp -------------- #
                #   if res_df['low'].iloc[e_j] <= short_tp.iloc[initial_i]: # 일단 tp 는 static 만 활용하는 걸로
                #     break

                #   # -------------- period fishing -------------- #
                #   # if intmin(res_df.index[e_j]) in [29, 59]:
                #   #   break

                #   # -------------- sar pbline -------------- #
                #   # if res_df['low'].iloc[e_j] <= short_tp.iloc[initial_i]:
                #   #   break

                #   #             2-2. out 조건이 성립되는 경우 limit 취소
                #   # -------------- st out -------------- #
                #   if res_df['close'].iloc[e_j] > short_out.iloc[e_j]:
                #     break

                #   # -------------- roc out -------------- #
                #   # if entry[e_j] == 1:
                #   #   break

                #   # -------------- stoch -------------- #
                #   # if res_df['stoch'].iloc[e_j - 2] >= res_df['stoch'].iloc[e_j - 1] and \
                #   #   res_df['stoch'].iloc[e_j - 1] < res_df['stoch'].iloc[e_j] and \
                #   #   res_df['stoch'].iloc[e_j - 1] <= stoch_lower:
                #   #   break

                #   # -------------- sar out -------------- #
                #   # if res_df['close'].iloc[e_j] > res_df['middle_line'].iloc[e_j]:
                #   # if res_df['close'].iloc[e_j] > short_out.iloc[initial_i]:
                #   #   break

                #   # -------------- sar prevsar out -------------- #
                #   # if res_df['sar2_uptrend'].iloc[e_j] == 1:

                #   #   if prev_sar is None:
                #   #     prev_sar = res_df['sar2'].iloc[e_j - 1]

                #   #   if res_df['close'].iloc[e_j] > prev_sar:
                #   #     break

                #   # else:
                #   #   if res_df['close'].iloc[e_j] > res_df['sar2'].iloc[e_j]:
                #   #     break

                #   # if res_df['close'].iloc[e_j] > res_df['middle_line'].iloc[e_j]:
                #   # if res_df['close'].iloc[e_j] > short_out.iloc[initial_i]: # or \
                #   #   # res_df['sar2_uptrend'].iloc[e_j] == 1: # or \

                #   # # if res_df['close'].iloc[e_j] > res_df['sar2'].iloc[e_j]:
                #   #   break

                # i = e_j
                # # print("i = e_j :", i)

                # if entry_done:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # ----------------- end wait ----------------- #

                # if e_j - initial_i >= 200:
                #   print("e_j, initial_i :", e_j, initial_i)
                # print("e_j - initial_i :", e_j - initial_i)
                # print()

                open_list.append(initial_i)

                if entry_type is 'market':
                    try:
                        ep_list = [res_df['close'].iloc[e_j]]
                    except Exception as e:
                        # print('error in ep_list (initial) :', e)
                        ep_list = [res_df['close'].iloc[ep_j]]
                else:
                    if entry[initial_i] == -1:
                        ep_list = [short_ep.iloc[ep_j]]
                    else:
                        #   e_j 가 있는 경우,
                        try:
                            ep_list = [res_df['open'].iloc[e_j]]
                        except Exception as e:
                            ep_list = [res_df['open'].iloc[ep_j]]

                if not static_lvrg:
                    # lvrg = target_pct / (res_df['high'].rolling(hl_lookback).max().iloc[initial_i] / res_df['close'].iloc[initial_i] - 1)

                    if entry_type is 'market':
                        # lvrg = target_pct / (short_out.iloc[ep_j] / res_df['close'].iloc[ep_j] - 1 - fee)
                        lvrg = target_pct / abs(res_df['close'].iloc[ep_j] / short_out.iloc[ep_j] - 1 - fee)
                    else:
                        lvrg = target_pct / (short_out.iloc[ep_j] / short_ep.iloc[ep_j] - 1 - fee)
                    lvrg = int(min(50, lvrg))
                    lvrg = max(lvrg, 1)
                    lvrg_list.append(lvrg)

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
                out_retouch = False

                if i == len(res_df) - 1:  # if j start from i + 1
                    open_list.pop()

                for j in range(i + 1, len(res_df)):
                    # for j in range(i, len(res_df)):

                    if static_tp:
                        tp_j = initial_i
                    else:
                        tp_j = j

                    if static_out:
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
                    if not non_tp:

                        #               1. by price line             #
                        if exit_type != 'market':

                            for s_i, short_tp_ in enumerate(short_tps):

                                if res_df['low'].iloc[j] <= short_tp_.iloc[
                                    tp_j] and partial_tp_cnt == s_i:  # we use static tp now
                                    # if res_df['low'].iloc[j] <= short_tp_.iloc[j]:
                                    # if res_df['low'].iloc[j] <= short_tp_.iloc[j] <= res_df['high'].iloc[j]: --> 이건 잘못되었음

                                    if s_i == len(short_tps) - 1:
                                        trade_done = True

                                    partial_tp_cnt += 1

                                    #         dynamic tp        #
                                    # if 0:
                                    if short_tp_.iloc[j] != short_tp_.iloc[j - 1] and not static_tp:

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

                            # ----------- st short inversion (long) ----------- #
                            if res_df['close'].iloc[j] >= short_tp.iloc[tp_j]:

                                # -------------- sar pb tp -------------- #
                                # if res_df['low'].iloc[j] <= short_tp.iloc[initial_i]:

                                # -------------- st tp -------------- #
                                # if res_df['close'].iloc[j] > res_df['middle_line'].iloc[j]:

                                # -------------- fisher tp -------------- #
                                # if entry[j] == 1:

                                # -------------- timestamp tp -------------- #
                                # if intmin(res_df.index[j]) in [29, 59]:

                                tp = res_df['close'].iloc[j]
                                # tp = res_df['open'].iloc[j]
                                trade_done = True

                                if trade_done:
                                    tp_state_list.append("long close tp")

                                tp_list.append(tp)
                                tp_idx_list.append(j)

                    # -------------- out -------------- #
                    if not trade_done and use_out and j != len(res_df) - 1:

                        # -------------- macd -------------- #
                        # if res_df['macd_hist3'].iloc[j] > 0:  #  macd out
                        # if res_df['macd_hist3'].iloc[i] < 0 and res_df['macd_hist3'].iloc[j] > 0:

                        # -------------- st retouch -------------- #
                        # out = True 상태면 동일 tick 에서 retouch 를 조사할 거기 때문에, 먼저 검사함
                        # 그리고, out 기준이 close 라 이게 맞음
                        # close 가 short_out 보다 올라가있는 상태일테니 low 를 조사하는게 맞음
                        # if out and res_df['low'].iloc[j] <= short_out.iloc[out_j]:
                        #   out_retouch = True

                        # ------- 일정시간 이상, dynamic_out 적용 ------ #
                        try:
                            if j - out_idx >= retouch_out_period:
                                static_short_out = short_out.iloc[j]

                        except Exception as e:
                            pass

                            # ------- static out ------ #
                        try:
                            if out and res_df['low'].iloc[j] <= static_short_out:
                                out_retouch = True
                        except Exception as e:
                            pass

                            # ------- retouch out ------ #
                        # if out and res_df['low'].iloc[j] <= short_out2.iloc[out_j]:
                        #   out_retouch = True

                        # -------------- st -------------- #
                        # if res_df['close'].iloc[j] > res_df['middle_line'].iloc[j]:
                        # if res_df['close'].iloc[j] > res_df['minor_ST3_Up'].iloc[j]:
                        # if res_df['close'].iloc[j] > upper_middle.iloc[j]:
                        # if res_df['close'].iloc[j] > res_df['minor_ST1_Up'].iloc[j]:
                        if out == 0:
                            if hl_out:
                                # if res_df['high'].iloc[j] >= short_out.iloc[out_j]: # check out only once
                                #   out = True
                                if res_df['low'].iloc[j] <= short_out.iloc[out_j]:  # short inversion out
                                    out = True

                            else:
                                # if res_df['close'].iloc[j] >= short_out.iloc[out_j]: # check out only once
                                #   out = True
                                if res_df['close'].iloc[j] <= short_out.iloc[out_j]:  # short inversion out
                                    out = True

                            # out_idx = j
                            # static_short_out = short_out.iloc[out_j]
                            # if second_out:
                            # static_short_out = short_out.iloc[out_j] + res_df['st_gap'].iloc[out_j] * second_out_gap

                        # if out == 0 and res_df['high'].iloc[j] >= short_out.iloc[out_j]: # check out only once
                        #   out = True

                        # -------------- sma -------------- #
                        # if res_df['close'].iloc[j] > res_df[sma].iloc[j]:

                        # -------------- sar -------------- #
                        # if res_df['close'].iloc[j] > res_df['minor_ST3_Up'].iloc[j] \
                        #   or res_df['sar2'].iloc[j] <= res_df['high'].iloc[j]:
                        # if res_df['close'].iloc[j] > short_out.iloc[initial_i]: # or \
                        #   out = True
                        # res_df['sar2_uptrend'].iloc[j] == 1: # or \

                        # if res_df['sar2_uptrend'].iloc[j] == 1:

                        #   if prev_sar is None:
                        #     prev_sar = res_df['sar2'].iloc[j - 1]

                        #   if res_df['close'].iloc[j] > prev_sar:
                        #     out = True

                        # else:
                        #   if res_df['close'].iloc[j] > res_df['sar2'].iloc[j]:
                        #     out = True

                        # -------------- hl -------------- #
                        # if res_df['close'].iloc[j] > short_out.iloc[tp_j]:

                        # -------------- stoch -------------- #
                        # if res_df['stoch'].iloc[j - 2] >= res_df['stoch'].iloc[j - 1] and \
                        #   res_df['stoch'].iloc[j - 1] < res_df['stoch'].iloc[j] and \
                        #   res_df['stoch'].iloc[j - 1] <= stoch_lower:
                        #   out = True

                        # retouch true 경우, out_retouch 조건도 있어야함
                        if out:
                            if retouch:
                                if out_retouch:
                                    pass
                                else:
                                    continue

                            else:
                                pass

                            if price_restoration:
                                tp = short_out.iloc[out_j]
                                if second_out:
                                    tp = short_out2.iloc[out_j]

                                # if res_df['close'].iloc[j] > tp: # 이 경우를 protect 하는건 insane 임
                                #   tp = res_df['close'].iloc[j]

                            else:

                                if not static_out:
                                    # if res_df['open'].iloc[j] >= short_out.iloc[out_j]: # close 기준이라 이런 조건을 못씀, 차라리 j 를 i 부터 시작
                                    if res_df['open'].iloc[j] <= short_out.iloc[out_j]:  # inversion
                                        tp = res_df['open'].iloc[j]
                                    else:
                                        tp = res_df['close'].iloc[j]

                                else:
                                    tp = res_df['close'].iloc[j]

                            if out_retouch:  # out 과 open 비교
                                if second_out:
                                    if res_df['open'].iloc[j] <= short_out2.iloc[out_j]:
                                        tp = res_df['open'].iloc[j]
                                else:
                                    if res_df['open'].iloc[j] <= short_out.iloc[out_j]:
                                        tp = res_df['open'].iloc[j]

                                try:  # static_short_out 인 경우, open 도 고려한 tp set
                                    if res_df['open'].iloc[j] <= static_short_out:
                                        tp = res_df['open'].iloc[j]
                                    else:
                                        tp = static_short_out
                                except Exception as e:
                                    pass

                            trade_done = True
                            tp_state_list.append("long close_out")

                            tp_list.append(tp)
                            tp_idx_list.append(j)

                    # -------------- non tp -------------- #
                    if j == len(res_df) - 1:
                        trade_done = True
                        tp = res_df['close'].iloc[j]
                        tp_list.append(tp)
                        tp_idx_list.append(j)

                    # -------------- append trade data -------------- #
                    if trade_done:

                        # --------------- tp_ratio info --------------- #
                        #         Todo        #
                        #          short_out 에 대한 정보는 존재함,
                        #          short_tp 에 대한 정보는 존재함,
                        #       => initial_i 기준으로 ,dynamic | static set 을 tp 와 out 에 각각 적용
                        #          lvrg 는 initial_i 기준으로 적용되니까
                        #          적용된 tp & out 으로 abs((tp - ep) / (ep - out)) 계산
                        try:
                            done_tp = short_tp.iloc[ep_j]
                            done_out = short_out.iloc[ep_j]

                            if done_out <= ep_list[0]:  # loss > 1
                                tp_ratio = np.nan
                            else:
                                tp_ratio = abs((ep_list[0] - done_tp) / (ep_list[0] - done_out))

                        except Exception as e:
                            # pass
                            tp_ratio = np.nan

                        tp_ratio_list.append(tp_ratio)
                        short_tp_ratio_list.append(tp_ratio)

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
                                    temp_qty = r_qty / p_qty_divider
                                else:
                                    temp_qty = r_qty

                            if inversion:
                                temp_pr = (tp_list[q_i] / ep_list[0] - fee - 1) * temp_qty * lvrg
                            else:
                                temp_pr = (ep_list[0] / tp_list[q_i] - fee - 1) * temp_qty * lvrg
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
                                sub_pr = (sub_ep_ / tp - fee - 1) * lvrg
                                p_ep_pr.append(sub_pr)

                            temp_pr = sum(p_ep_pr) + 1

                            print("temp_pr :", temp_pr)

                        # ------------ hedge + non_hedge pr summation ------------ #
                        #         hedge pr direction is opposite to the origin       #
                        hedge_pr = 1
                        if hedge_cnt == 0:

                            #       hedge tp      #
                            h_tp = res_df['close'].iloc[j]
                            if inversion:
                                hedge_pr = (h_ep / h_tp - fee - 1) * lvrg  # hedge short
                            else:
                                hedge_pr = (h_tp / h_ep - fee - 1) * lvrg  # hedge long
                            temp_pr += hedge_pr
                            h_j = j

                        hh = max(res_df['high'].iloc[i:j + 1])
                        short_liq = (ep_list[0] / hh - fee - 1) * lvrg + 1

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
                            short_list.append(temp_pr)

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

                            #         tp 미체결 survey        #
                            nontp_liqd_list.append(short_liq)
                            nontp_short_liqd_list.append(short_liq)
                            nontp_short_indexs.append(i)
                            nontp_short_ep_list.append(ep_list[0])

                            if inversion:
                                nontp_short_pr = (tp / ep_list[0] - fee - 1) * lvrg + 1
                            else:
                                nontp_short_pr = (ep_list[0] / tp - fee - 1) * lvrg + 1
                            nontp_pr_list.append(nontp_short_pr)
                            nontp_short_pr_list.append(nontp_short_pr)



            #                  long  phase                #
            elif entry[i] in long_entry:  # inversion

                initial_i = i

                if static_out:
                    p_i = initial_i
                else:
                    p_i = i

                # -------------- ep scheduling -------------- #
                # ----- st_gap ----- #
                if res_df['close'].iloc[i] <= res_df['st_upper2_5m'].iloc[i]:
                    # if res_df['close'].iloc[i] >= res_df['upper_middle3'].iloc[i] and \
                    # abs(res_df['close'].iloc[i] - res_df['bb_upper'].iloc[i]) < ep_protect_gap * res_df['st_gap3'].iloc[i]:
                    # # abs(res_df['close'].iloc[i] - res_df['upper_middle3'].iloc[i]) < ep_protect_gap * res_df['st_gap3'].iloc[i]:
                    # # if abs(res_df['close'].iloc[i] - long_ep.iloc[i]) < ep_protect_gap * res_df['st_gap'].iloc[i]:
                    # # if res_df['close'].iloc[i] < long_ep.iloc[i]:
                    # # # if abs((res_df['close'].iloc[i] - lower_middle.iloc[i]) / lower_middle.iloc[i]) < ep_protect_gap:
                    # # # if abs((res_df['close'].iloc[i] - res_df['middle_line'].iloc[i]) / res_df['middle_line'].iloc[i]) < ep_protect_gap:
                    # # if res_df['close'].shift(1).iloc[i] >= long_out.iloc[p_i] + ep_protect_gap * res_df['st_gap%s' % basic_st_number].iloc[p_i]: # out 기준임을 주목

                    # ----- bb_gap ----- #
                    # if abs(res_df['close'].iloc[i] - res_df['bb_upper_1m'].iloc[i]) < ep_protect_gap * bb_gap.iloc[i]:

                    pass
                else:
                    i += 1
                    if i >= len(res_df):
                        break
                    continue

                # -------------- tp scheduling -------------- #
                # if bb_gap.iloc[i] / res_df['close'].iloc[i] > fee:

                # ---- pgfr : profit_gap & fee ratio --- #
                # if (bb_gap.iloc[i] / res_df['close'].iloc[i] - fee) / abs(bb_gap.iloc[i] / res_df['close'].iloc[i] + fee) > pgfr:
                if (long_tp.iloc[i] / res_df['close'].iloc[i] - fee) / abs(
                        long_tp.iloc[i] / res_df['close'].iloc[i] + fee) > pgfr:

                    pass
                else:
                    i += 1
                    if i >= len(res_df):
                        break
                    continue

                # -------------- markup only -------------- #
                # ------- entry once ------- #
                #          1. find prev out         #
                #          2. count entry til prev out         #
                # prev_entry_cnt = 0
                # for back_i in range(i - 1, 0, -1):
                #   if entry[back_i] == -1:
                #     break

                #   elif entry[back_i] == 1:
                #     prev_entry_cnt += 1

                # if prev_entry_cnt <= entry_incycle:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # ------- ema alignment ------- #
                # if res_df['ema5_1m'].iloc[i] > res_df['ema5_1m'].iloc[i- 1] > res_df['ema5_1m'].iloc[i - 2]:
                # # if res_df['ema5_1m'].iloc[i- 1] > res_df['ema5_1m'].iloc[i - 2]:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                #   # ------- sma alignment ------- #
                # if res_df['sma_1m'].iloc[i] > res_df['sma_1m'].iloc[i- 1]:
                # # if res_df['sma_1m'].iloc[i] > res_df['sma_1m'].iloc[i- 1] > res_df['sma_1m'].iloc[i - 2]:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # # ------- bb alignment ------- #
                # if res_df['bb_upper_1m'].iloc[i] > res_df['bb_upper_1m'].iloc[i - 1]:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # ------- bb_base alignment ------- #
                # if res_df['bb_base_1m'].iloc[i] > res_df['bb_base_1m'].iloc[i - 1]:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # ------------------- markup only major ------------------- #

                # -------------- ma const. -------------- #
                # ------- ema const. ------- #
                # if res_df['close'].shift(0).iloc[i] > res_df['ema5'].shift(0).iloc[i]: # and \
                # if res_df['close'].shift(1).iloc[i] > res_df['ema5'].shift(1).iloc[i]: # and \
                # if res_df['close'].shift(1).iloc[i] > res_df['ema5'].shift(0).iloc[i]: # and \
                # if long_ep.iloc[i] > res_df['ema5'].shift(0).iloc[i]: # and \
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # ------- sma const. ------- #
                # if res_df['close'].iloc[i] >= res_df['sma_1m'].iloc[i]: # and \
                # # if res_df['close'].shift(1).iloc[i] > res_df['sma_1m'].shift(1).iloc[i]: # and \
                # #   long_ep.iloc[i] >= res_df['sma_1m'].shift(sma_shift_size).iloc[i]:
                # # # upper_sma = long_ep >= res_df['sma'].shift(sma_shift_size)
                # # # if np.sum(upper_sma.iloc[i + 1 - sma_lookback:i + 1]) == sma_lookback:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # ------- bb zone const. ------- #
                # if res_df['bb_upper_1m'].iloc[i] > res_df['bb_upper_30m'].iloc[i]:
                # if res_df['close'].iloc[i] >= res_df['bb_upper_30m'].iloc[i]:
                # # if res_df['bb_upper_1m'].iloc[i] > res_df['bb_upper2_30m'].iloc[i]:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # ------- cloud bline ------- #
                # if res_df['close'].iloc[i] >= res_df['cloud_bline_5m'].iloc[i]:
                if res_df['close'].iloc[i] >= res_df['cloud_bline_3m'].iloc[i]:
                    pass
                else:
                    i += 1
                    if i >= len(res_df):
                        break
                    continue

                # -------------- distance protection -------------- #
                # tp_dist = (long_tp.iloc[i] - res_df['close'].iloc[i])
                # out_dist = (res_df['close'].iloc[i] - res_df['middle_line'].iloc[i])
                # if tp_dist / out_dist >= tp_out_ratio:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # -------------- feature dist const. -------------- #
                # if initial_i < input_size:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # entry_input_x = min_max_scale(res_df[selected_price_colname].iloc[initial_i - input_size:initial_i].values)

                # re_entry_input_x = expand_dims(entry_input_x)

                # entry_vector = model.predict(re_entry_input_x, verbose=0)
                # # print(test_result.shape)

                # f_dist = vector_dist(entry_vector, selected_vector)
                # print("f_dist :", f_dist)

                # if f_dist < fdist_thresh:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # -------------- htf data const. -------------- #
                # i_min = intmin(res_df.index[i]) # 2020-09-05 00:00:59.999000
                # if i_min >= 30:
                #   htf_ts = str(res_df.index[i])[:-12] + "59:59.999000"
                # else:
                #   htf_ts = str(res_df.index[i])[:-12] + "29:59.999000"

                #   # -------------- ema -------------- #
                # if fifth_df['close'].shift(1).loc[htf_ts] > fifth_df['ema'].shift(1).loc[htf_ts]:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # -------------- fisher const. -------------- #
                # # if res_df['fisher30'].shift(1).iloc[i] > 0:
                # # if res_df['fisher60'].shift(1).iloc[i] > 0:
                # if res_df['fisher120'].shift(1).iloc[i] > 0:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # -------------- htf st line const. -------------- #

                # ------------ 30m ------------ #
                # # if res_df['upper_middle5'].iloc[i] <= res_df['max_lower2'].iloc[i]:
                # # if res_df['upper_middle5'].iloc[i] <= res_df['close'].iloc[i]:
                # if res_df['upper_middle5'].iloc[i] >= res_df['bb_upper'].iloc[i] and \
                # res_df['bb_upper'].iloc[i] > res_df['ema1_5'].iloc[i]:

                #   # ------------ 4h ------------ #
                # # if res_df['upper_middle6'].iloc[i] <= res_df['max_lower2'].iloc[i]:

                #  # ------------ 5m & 30m ------------ #
                # # if res_df['upper_middle5'].iloc[i] <= res_df['middle_line3'].iloc[i]:
                # # if res_df['upper_middle5'].iloc[i] <= res_df['lower_middle3'].iloc[i]:

                #  # ------------ 5m & 30m & 4h ------------ #
                # # if res_df['upper_middle6'].iloc[i] <= res_df['upper_middle5'].iloc[i] <= res_df['upper_middle3'].iloc[i]:

                #     pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # -------------- 1d sma const. -------------- #
                # if res_df[sma].iloc[i] <= res_df['close'].iloc[i]:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # -------------- cloud const. -------------- #
                # if i < cloud_lookback:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # # if np.sum(under_bottom.iloc[i + 1 - cloud_lookback:i + 1]) == cloud_lookback:
                # # if np.sum(under_top.iloc[i + 1 - cloud_lookback:i + 1]) == cloud_lookback:
                # if np.sum(over_bottom.iloc[i + 1 - cloud_lookback:i + 1]) == cloud_lookback:
                # # if np.sum(over_top.iloc[i + 1 - cloud_lookback:i + 1]) == cloud_lookback:
                #  pass

                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # -------------- cloud color const. -------------- #
                #               1. senkou_a1 >= senkou_b1            #
                #               1-1. mutli clouds color 충분히 고려               #
                # if i < cloud_lookback:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # if np.sum(res_df[senkou_a].shift(cloud_shift_size).iloc[i + 1 - cloud_lookback:i + 1] >= res_df[senkou_b].shift(cloud_shift_size).iloc[i + 1 - cloud_lookback:i + 1]) == cloud_lookback and \
                #   np.sum(res_df["senkou_a2"].shift(0).iloc[i + 1 - cloud_lookback:i + 1] >= res_df["senkou_b2"].shift(0).iloc[i + 1 - cloud_lookback:i + 1]) == cloud_lookback: # and \
                #   # np.sum(res_df["senkou_a3"].shift(0).iloc[i + 1 - cloud_lookback:i + 1] >= res_df["senkou_b3"].shift(0).iloc[i + 1 - cloud_lookback:i + 1]) == cloud_lookback and \
                #   # np.sum(res_df["senkou_a4"].shift(0).iloc[i + 1 - cloud_lookback:i + 1] >= res_df["senkou_b4"].shift(0).iloc[i + 1 - cloud_lookback:i + 1]) == cloud_lookback and \
                #   # np.sum(res_df["senkou_a5"].shift(0).iloc[i + 1 - cloud_lookback:i + 1] >= res_df["senkou_b5"].shift(0).iloc[i + 1 - cloud_lookback:i + 1]) == cloud_lookback:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # -------------- st color const. -------------- #
                # if np.sum(res_df[['ST1_Trend%s' % basic_st_number, 'ST2_Trend%s' % basic_st_number, 'ST3_Trend%s' % basic_st_number]].iloc[i]) >= 1:
                # # if np.sum(res_df[['ST1_Trend%s' % basic_st_number, 'ST2_Trend%s' % basic_st_number, 'ST3_Trend%s' % basic_st_number]].iloc[i]) >= 3:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # -------------- htf st color const. -------------- #
                # if np.sum(res_df[['major_ST1_Trend', 'major_ST2_Trend', 'major_ST3_Trend']].iloc[i]) >= 1:
                # # if np.sum(res_df[['major_ST1_Trend', 'major_ST2_Trend', 'major_ST3_Trend']].iloc[i]) >= 3:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # -------------- 3rd st const. : st should have 2, 3 or more -------------- #
                # if np.sum(res_df[['minor_ST2_Trend', 'minor_ST3_Trend']].iloc[i]) >= 2:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # -------------- st gap const.-------------- #
                # if abs(res_df['ST2_Down%s' % basic_st_number].iloc[i] - res_df['ST1_Down%s' % basic_st_number].iloc[i]) <= approval_st_gap * res_df['st_gap%s' % basic_st_number].iloc[i]:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # -------------- sar const. -------------- #
                # # if res_df['sar2'].iloc[i] < res_df['low'].iloc[i] and res_df['sar3'].iloc[i] < res_df['low'].iloc[i]:
                # if res_df['sar1'].iloc[i] < res_df['low'].iloc[i]:
                # # if res_df['sar2'].iloc[i] < res_df['low'].iloc[i]: # and \
                # # if  res_df['sar3'].iloc[i] < res_df['low'].iloc[i]:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                ep_j = initial_i

                # -------------- limit waiting const. -------------- #

                # entry_done = False
                # prev_sar = None
                # # for e_j in range(i, len(res_df)):
                # for e_j in range(i + 1, len(res_df)):

                #   if not static_ep:
                #     ep_j = e_j

                #   #             Todo            #
                #   #             1. ep 설정

                #   # -------------- np.inf ep -------------- #
                #   # if long_ep.iloc[initial_i] == np.inf:
                #   #   break

                #   #             1-1. close 가 sar_change 이전 sar 을 cross 한 경우만 진입

                #   # -------------- ep scheduling -------------- #
                #   # if res_df['close'].shift(1).iloc[e_j] >= long_out.iloc[ep_j] + ep_protect_gap * res_df['st_gap%s' % basic_st_number].iloc[ep_j]:
                #   #   pass
                #   # else:
                #   #   continue

                #   # if res_df['low'].iloc[e_j] <= long_ep.iloc[initial_i] and res_df['close'].shift(1).iloc[e_j] >= long_ep.iloc[initial_i]:
                #   # if res_df['low'].iloc[e_j] <= long_ep.iloc[initial_i] and res_df['close'].shift(1).iloc[e_j] <= long_out.iloc[initial_i] + res_df['st_gap'].iloc[initial_i] * ep_out_gap:
                #   # if res_df['low'].iloc[e_j] <= long_ep.iloc[initial_i] and res_df['close'].shift(1).iloc[e_j] >= long_out.iloc[initial_i] + res_df['st_gap'].iloc[initial_i] * ep_out_gap:
                #   if res_df['low'].iloc[e_j] <= long_ep.iloc[ep_j]:
                #     entry_done = True
                #     # print("e_j :", e_j)

                #     #     이미, e_j open 이 ep 보다 낮은 경우, entry[initial_i] => -2 로 변경   #
                #     if res_df['open'].iloc[e_j] <= long_ep.iloc[ep_j]:
                #       entry[initial_i] = -2

                #     break

                #   #             2. limit 대기 시간 설정
                #   #             2-1. tp 하거나, out 조건이 성립되는 경우 limit 취소
                #   #             2-1-1. timestamp = 29 or 59
                #   # -------------- st tp -------------- #
                #   if res_df['high'].iloc[e_j] >= long_tp.iloc[initial_i]:
                #     break

                #   # -------------- period fishing -------------- #
                #   # if intmin(res_df.index[e_j]) in [29, 59]:
                #   #   break

                #   # -------------- sar pbline -------------- #
                #   # if res_df['high'].iloc[e_j] >= long_tp.iloc[initial_i]:
                #     # break

                #   #             2-2. out 조건이 성립되는 경우 limit 취소
                #   # -------------- st out -------------- #
                #   if res_df['close'].iloc[e_j] < long_out.iloc[e_j]:
                #     break

                #   # -------------- roc out -------------- #
                #   # if entry[e_j] == -1:
                #   #   break

                #   # -------------- stoch -------------- #
                #   # if res_df['stoch'].iloc[e_j - 2] <= res_df['stoch'].iloc[e_j - 1] and \
                #   #   res_df['stoch'].iloc[e_j - 1] > res_df['stoch'].iloc[e_j] and \
                #   #   res_df['stoch'].iloc[e_j - 1] >= stoch_upper:
                #   #   break

                #   # -------------- sar out -------------- #
                #   # if res_df['close'].iloc[e_j] < long_out.iloc[initial_i]: # or \
                #   #   break

                #   # -------------- sar prevsar out -------------- #

                #   # if res_df['sar2_uptrend'].iloc[e_j] == 0:

                #   #     if prev_sar is None:
                #   #       prev_sar = res_df['sar2'].iloc[e_j - 1]

                #   #     if res_df['close'].iloc[j] < prev_sar:
                #   #       break

                #   # else:
                #   #   if res_df['close'].iloc[e_j] < res_df['sar2'].iloc[e_j]:
                #   #     break

                # i = e_j
                # # print("i = e_j :", i)

                # if entry_done:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # ---------------- end wait ---------------- #
                # if e_j - initial_i >= 200:
                #   print("e_j, initial_i :", e_j, initial_i)

                open_list.append(initial_i)

                if entry_type is 'market':
                    try:
                        ep_list = [res_df['close'].iloc[e_j]]
                    except Exception as e:
                        # print('error in ep_list (initial) :', e)
                        ep_list = [res_df['close'].iloc[ep_j]]
                else:
                    if entry[initial_i] == 1:
                        ep_list = [long_ep.iloc[ep_j]]
                    else:
                        try:
                            ep_list = [res_df['open'].iloc[e_j]]
                        except Exception as e:
                            ep_list = [res_df['open'].iloc[ep_j]]

                if not static_lvrg:
                    # lvrg = target_pct / (res_df['close'].iloc[initial_i] / res_df['low'].rolling(hl_lookback).min().iloc[initial_i] - 1)
                    if entry_type is 'market':
                        lvrg = target_pct / abs(long_out.iloc[ep_j] / res_df['close'].iloc[ep_j] - 1 - fee)
                    else:
                        lvrg = target_pct / (long_ep.iloc[ep_j] / long_out.iloc[ep_j] - 1 - fee)
                    lvrg = int(min(50, lvrg))
                    lvrg = max(1, lvrg)
                    lvrg_list.append(lvrg)

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
                out_retouch = False

                if i == len(res_df) - 1:  # if j start from i + 1
                    open_list.pop()

                for j in range(i + 1, len(res_df)):
                    # for j in range(i, len(res_df)):

                    if static_tp:
                        tp_j = initial_i
                    else:
                        tp_j = j

                    if static_out:
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
                    if not non_tp:
                        #            1. by price line            #
                        if exit_type != 'market':

                            for l_i, long_tp_ in enumerate(long_tps):

                                if res_df['high'].iloc[j] >= long_tp_.iloc[tp_j] and partial_tp_cnt == l_i:
                                    # if res_df['high'].iloc[j] >= long_tp.iloc[j]:

                                    if l_i == len(long_tps) - 1:
                                        trade_done = True

                                    partial_tp_cnt += 1

                                    #         dynamic tp        #
                                    # if 0:
                                    if long_tp_.iloc[j] != long_tp_.iloc[j - 1] and not static_tp:

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

                        #           2. by signal        #
                        else:

                            # -------------- sar tp -------------- #
                            # if (res_df['low'].iloc[j] <= res_df['sar2'].iloc[j]) & \
                            #   (res_df['low'].iloc[j - 1] > res_df['sar2'].iloc[j - 1]) & \
                            #   (res_df['low'].iloc[j - 2] > res_df['sar2'].iloc[j - 2]):

                            #       inversion     #
                            # if (res_df['high'].iloc[j] >= res_df['sar2'].iloc[j]) & \
                            #   (res_df['high'].iloc[j - 1] < res_df['sar2'].iloc[j - 1]) & \
                            #   (res_df['high'].iloc[j - 2] < res_df['sar2'].iloc[j - 2]):

                            # -------------- sar pb tp -------------- #
                            # if res_df['high'].iloc[j] >= long_tp.iloc[initial_i]:

                            # ----------- st short inversion (short) ----------- #
                            if res_df['close'].iloc[j] <= long_tp.iloc[tp_j]:

                                # -------------- st tp -------------- #
                                # if res_df['close'].iloc[j] < res_df['middle_line'].iloc[j]:

                                # -------------- fisher tp -------------- #
                                # if entry[j] == -1:

                                # -------------- timestamp tp -------------- #
                                # if intmin(res_df.index[j]) in [29, 59]:

                                tp = res_df['close'].iloc[j]
                                # tp = res_df['open'].iloc[j]
                                trade_done = True

                                if trade_done:
                                    tp_state_list.append("short close tp")

                                tp_list.append(tp)
                                tp_idx_list.append(j)

                    # -------------- out -------------- #
                    if not trade_done and use_out and j != len(res_df) - 1:

                        # -------------- macd -------------- #
                        # if res_df['macd_hist3'].iloc[j] < 0:
                        # # if res_df['macd_hist3'].iloc[i] > 0 and res_df['macd_hist3'].iloc[j] < 0:

                        # -------------- st retouch -------------- #
                        # out = True 상태면 동일 tick 에서 retouch 를 조사할 거기 때문에, 먼저 검사함
                        # 그리고, out 기준이 close 라 이게 맞음
                        # close 가 long_out 보다 내려가있는 상태일테니 high 를 조사하는게 맞음
                        # if out and res_df['high'].iloc[j] >= long_out.iloc[out_j]:
                        #   out_retouch = True

                        # ------- 일정시간 이상, dynamic_out 적용 ------ #
                        try:
                            if j - out_idx >= retouch_out_period:
                                static_long_out = long_out.iloc[j]

                        except Exception as e:
                            pass

                            # ------- static out ------ #
                        try:
                            if out and res_df['high'].iloc[j] >= static_long_out:
                                out_retouch = True
                        except Exception as e:
                            pass

                            # ------- retouch out ------ #
                        # if out and res_df['high'].iloc[j] >= long_out2.iloc[out_j]:
                        #   out_retouch = True

                        # -------------- st -------------- #
                        # if res_df['close'].iloc[j] < res_df['middle_line'].iloc[j]:
                        # if res_df['close'].iloc[j] < res_df['minor_ST3_Down'].iloc[j]:
                        # if res_df['close'].iloc[j] < lower_middle.iloc[j]:
                        # if res_df['close'].iloc[j] < res_df['minor_ST1_Down'].iloc[j]:
                        if out == 0:
                            if hl_out:
                                # if res_df['low'].iloc[j] <= long_out.iloc[out_j]: # check out only once
                                if res_df['high'].iloc[j] >= long_out.iloc[out_j]:  # long inversion
                                    out = True

                            else:
                                # if res_df['close'].iloc[j] <= long_out.iloc[out_j]: # check out only once
                                if res_df['close'].iloc[j] >= long_out.iloc[out_j]:  # long inversion
                                    out = True

                            # out_idx = j
                            # static_long_out = long_out.iloc[out_j]
                            # if second_out:
                            # static_long_out = long_out.iloc[out_j] - res_df['st_gap'].iloc[out_j] * second_out_gap

                        # if out == 0 and res_df['low'].iloc[j] <= long_out.iloc[out_j]: # check out only once
                        #   out = True

                        # -------------- sma -------------- #
                        # if res_df['close'].iloc[j] < res_df[sma].iloc[j]:

                        # -------------- sar -------------- #
                        # if res_df['close'].iloc[j] < res_df['minor_ST3_Down'].iloc[j] \
                        #   or res_df['sar2'].iloc[j] >= res_df['low'].iloc[j]:
                        # if res_df['close'].iloc[j] < long_out.iloc[initial_i]: # or \
                        #   #  res_df['close'].iloc[j] < res_df['sar2'].iloc[j]:
                        #   #  res_df['sar2_uptrend'].iloc[j] == 0 or \
                        #   out = True

                        # if res_df['sar2_uptrend'].iloc[j] == 0:

                        #     if prev_sar is None:
                        #       prev_sar = res_df['sar2'].iloc[j - 1]

                        #     if res_df['close'].iloc[j] < prev_sar:
                        #       out = True

                        # else:
                        #   if res_df['close'].iloc[j] < res_df['sar2'].iloc[j]:
                        #     out = True

                        # -------------- hl -------------- #
                        # if res_df['close'].iloc[j] < long_out.iloc[tp_j]:

                        # -------------- stoch -------------- #
                        # if res_df['stoch'].iloc[j - 2] <= res_df['stoch'].iloc[j - 1] and \
                        #   res_df['stoch'].iloc[j - 1] > res_df['stoch'].iloc[j] and \
                        #   res_df['stoch'].iloc[j - 1] >= stoch_upper:
                        #   out = True

                        # retouch true 경우, out_retouch 조건도 있어야함
                        if out:
                            if retouch:
                                if out_retouch:
                                    pass
                                else:
                                    continue

                            else:
                                pass

                            if price_restoration:
                                tp = long_out.iloc[out_j]
                                if second_out:
                                    tp = long_out2.iloc[out_j]

                                # if res_df['close'].iloc[j] < tp: # 이 경우를 protect 하는건 insane 임
                                # # if res_df['high'].iloc[j] < tp: # --> hl_out 사용시 이 조건은 valid 함
                                #   tp = res_df['close'].iloc[j]

                            else:

                                if not static_out:
                                    # if res_df['open'].iloc[j] <= long_out.iloc[out_j]: # dynamic close out 의 open 고려
                                    if res_df['open'].iloc[j] >= long_out.iloc[out_j]:  # inversion
                                        tp = res_df['open'].iloc[j]
                                    else:
                                        tp = res_df['close'].iloc[j]

                                else:
                                    tp = res_df['close'].iloc[j]

                            if out_retouch:  # out 과 open 비교
                                if second_out:  # long_out = sell
                                    # second_out 은 기본적으로 limit 이라 이 구조가 가능함
                                    if res_df['open'].iloc[j] >= long_out2.iloc[out_j]:  # dynamic_out 일 경우 고려해야함
                                        tp = res_df['open'].iloc[j]
                                else:
                                    if res_df['open'].iloc[j] >= long_out.iloc[out_j]:  # dynamic_out 일 경우 고려해야함
                                        tp = res_df['open'].iloc[j]

                                try:
                                    if res_df['open'].iloc[j] >= static_long_out:
                                        tp = res_df['open'].iloc[j]
                                    else:
                                        tp = static_long_out
                                except Exception as e:
                                    pass

                            # tp = res_df['open'].iloc[j]
                            tp_state_list.append("short close_out")
                            trade_done = True

                            tp_list.append(tp)
                            tp_idx_list.append(j)

                    # -------------- non tp -------------- #
                    if j == len(res_df) - 1:
                        trade_done = True
                        tp = res_df['close'].iloc[j]
                        tp_list.append(tp)
                        tp_idx_list.append(j)

                    if trade_done:

                        # --------------- tp_ratio info --------------- #
                        try:
                            done_tp = long_tp.iloc[ep_j]
                            done_out = long_out.iloc[ep_j]

                            if done_out >= ep_list[0]:  # loss >= 1
                                tp_ratio = np.nan
                            else:
                                tp_ratio = abs((ep_list[0] - done_tp) / (ep_list[0] - done_out))

                        except Exception as e:
                            # pass
                            tp_ratio = np.nan

                        tp_ratio_list.append(tp_ratio)
                        long_tp_ratio_list.append(tp_ratio)

                        qty_list = []
                        temp_pr_list = []
                        r_qty = 1
                        for q_i in range(len(tp_list) - 1, -1, -1):

                            if len(tp_list) == 1:
                                temp_qty = r_qty
                            else:
                                if q_i != 0:
                                    temp_qty = r_qty / p_qty_divider
                                else:
                                    temp_qty = r_qty

                            # temp_pr = (tp_list[q_i] / ep_list[0] - fee - 1) * temp_qty
                            if inversion:
                                temp_pr = (ep_list[0] / tp_list[q_i] - fee - 1) * temp_qty * lvrg
                            else:
                                temp_pr = (tp_list[q_i] / ep_list[0] - fee - 1) * temp_qty * lvrg
                            r_qty -= temp_qty

                            temp_pr_list.append(temp_pr)

                        temp_pr = sum(temp_pr_list) + 1

                        # -------------------- sub ep -> pr calc -------------------- #
                        if len(ep_list) > 1:

                            p_ep_pr = []
                            for sub_ep_ in ep_list:
                                sub_pr = (tp / sub_ep_ - fee - 1) * lvrg
                                p_ep_pr.append(sub_pr)

                            temp_pr = sum(p_ep_pr) + 1

                            print("temp_pr :", temp_pr)

                        # ------------ hedge + non_hedge pr summation ------------ #
                        #         hedge pr direction is opposite to the origin       #
                        hedge_pr = 1
                        if hedge_cnt == 0:

                            #       hedge tp      #
                            h_tp = res_df['close'].iloc[j]
                            if inversion:
                                hedge_pr = (h_tp / h_ep - fee - 1) * lvrg  # hedge short inversion
                            else:
                                hedge_pr = (h_ep / h_tp - fee - 1) * lvrg  # hedge short
                            temp_pr += hedge_pr
                            h_j = j

                        ll = min(res_df['low'].iloc[i:j + 1])
                        long_liq = (ll / ep_list[0] - fee - 1) * lvrg + 1

                        if j != len(res_df) - 1:

                            ep_tp_list.append((ep_list, tp_list))
                            trade_list.append((ep_idx_list, tp_idx_list))

                            liqd_list.append(long_liq)
                            long_liqd_list.append(long_liq)

                            h_ep_tp_list.append((h_ep, h_tp))
                            h_trade_list.append([initial_i, h_i, h_j])

                            pr_list.append(temp_pr)
                            long_list.append(temp_pr)

                            h_pr_list.append(hedge_pr)
                            h_long_list.append(hedge_pr)

                            i = j
                            break

                        else:

                            # ep_tp_list.append((ep_list, tp_list))
                            # trade_list.append((ep_idx_list, tp_idx_list))

                            # pr_list 를 넣지 않을거니까, open_list 에서 해당 idx 는 pop
                            open_list.pop()

                            #         tp 미체결 survey        #
                            nontp_liqd_list.append(long_liq)
                            nontp_long_liqd_list.append(long_liq)
                            nontp_long_indexs.append(i)
                            nontp_long_ep_list.append(ep_list[0])

                            if inversion:
                                nontp_long_pr = (ep_list[0] / tp - fee - 1) * lvrg + 1
                            else:
                                nontp_long_pr = (tp / ep_list[0] - fee - 1) * lvrg + 1
                            nontp_pr_list.append(nontp_long_pr)
                            nontp_long_pr_list.append(nontp_long_pr)

                        if len(open_list) > len(trade_list):
                            print('debug from index :', i)
                            print(len(open_list), len(trade_list))
                            print("len(res_df) :", len(res_df))
                            assert len(open_list) == len(trade_list), 'stopped'

            i += 1
            if i >= len(res_df):
                break

        # -------------------- result analysis -------------------- #
        try:
            # plt.style.use('default')
            # mpl.rcParams.update(mpl.rcParamsDefault)

            title_position = (0.5, 1.0 - 0.1)

            plt.figure(figsize=(16, 12))
            # plt.figure(figsize=(12, 8))
            # plt.figure(figsize=(10, 6))
            # plt.suptitle(key)

            np_pr = np.array(pr_list)
            # np_pr = (np.array(pr_list) - 1) * lvrg + 1

            # ----- fake_pr ----- #
            # np_pr = np.where(np_pr > 1, 1 + (np_pr - 1) * 2, np_pr)

            total_pr = np.cumprod(np_pr)
            wr = len(np_pr[np_pr > 1]) / len(np_pr[np_pr != 1])

            # mean_profit = np.mean(np_pr[np_pr > 1])
            # mean_loss = np.mean(np_pr[np_pr < 1])
            # cumprod_profit = np.cumprod(np_pr[np_pr > 1])[-1]
            # cumprod_loss = np.cumprod(np_pr[np_pr < 1])[-1]
            # pr_tr = cumprod_profit * cumprod_loss

            np_tp_ratio_list = np.array(tp_ratio_list)  # 초기에 tr 을 정하는거라 mean 사용하는게 맞음
            mean_tr = np.mean(np_tp_ratio_list[np.isnan(np_tp_ratio_list) == 0])

            pr_gap = (np_pr - 1) / lvrg + fee
            tp_gap_ = pr_gap[pr_gap > 0]
            # mean_tp_gap = np.mean(pr_gap[pr_gap > 0])
            mean_ls_gap = np.mean(pr_gap[pr_gap < 0])

            # ---- profit fee ratio ---- #
            mean_pgfr = np.mean((tp_gap_ - fee) / abs(tp_gap_ + fee))

            # plt.subplot(121)
            plt.subplot(231)
            plt.plot(total_pr)
            if len(nontp_liqd_list) != 0:
                plt.title(
                    "wr : %.2f\nlen(td) : %s\nmin_pr : %.2f\nacc_pr : %.2f\n lvrg %s\nliqd : %.2f\nmean_tr : %.2f\nmean_pgfr : %.2f\nnontp_liqd_cnt : %s\nnontp_liqd : %.2f\nontp_liqd_pr : %.2f"
                    % (
                    wr, len(np_pr[np_pr != 1]), np.min(np_pr), total_pr[-1], lvrg, min(liqd_list), mean_tr, mean_pgfr,
                    len(nontp_liqd_list), min(nontp_liqd_list), min(nontp_pr_list)),
                    position=title_position,
                    )
            else:
                plt.title(
                    "wr : %.2f\nlen(td) : %s\nmin_pr : %.2f\nacc_pr : %.2f\n lvrg %s\nliqd : %.2f\nmean_tr : %.2f\nmean_pgfr : %.2f\nnontp_liqd_cnt : %s"
                    % (
                    wr, len(np_pr[np_pr != 1]), np.min(np_pr), total_pr[-1], lvrg, min(liqd_list), mean_tr, mean_pgfr,
                    len(nontp_liqd_list)),
                    position=title_position,
                    )

            plt.tight_layout()
            # plt.show()

            print('supblot231 passed')

            #         short only      #
            short_np_pr = np.array(short_list)

            total_short_pr = np.cumprod(short_np_pr)
            short_wr = len(short_np_pr[short_np_pr > 1]) / len(short_np_pr[short_np_pr != 1])

            # short_cumprod_profit = np.cumprod(short_np_pr[short_np_pr > 1])[-1]
            # short_cumprod_loss = np.cumprod(short_np_pr[short_np_pr < 1])[-1]
            # short_pr_tr = short_cumprod_profit * short_cumprod_loss

            np_short_tp_ratio_list = np.array(short_tp_ratio_list)
            mean_short_tr = np.mean(np_short_tp_ratio_list[np.isnan(np_short_tp_ratio_list) == 0])

            short_pr_gap = (short_np_pr - 1) / lvrg + fee
            short_tp_gap = short_pr_gap[short_pr_gap > 0]
            # mean_short_tp_gap = np.mean(short_pr_gap[short_pr_gap > 0])
            # mean_short_ls_gap = np.mean(short_pr_gap[short_pr_gap < 0])

            mean_short_pgfr = np.mean((short_tp_gap - fee) / abs(short_tp_gap + fee))

            plt.subplot(232)
            plt.plot(total_short_pr)
            if len(nontp_short_liqd_list) != 0:

                max_nontp_short_term = len(res_df) - nontp_short_indexs[0]

                plt.title(
                    "wr : %.2f\nlen(short_td) : %s\nmin_pr : %.2f\nacc_pr : %.2f\n lvrg %s\nliqd : %.2f\nmean_tr : %.2f\nmean_short_pgfr : %.2f\nnontp_short_liqd_cnt : %s\nnontp_short_liqd : %.2f\nontp_short_liqd_pr : %.2f\nmax_nontp_short_term : %s"
                    % (short_wr, len(short_np_pr[short_np_pr != 1]), np.min(short_np_pr), total_short_pr[-1], lvrg,
                       min(short_liqd_list), mean_short_tr, mean_short_pgfr,
                       len(nontp_short_liqd_list), min(nontp_short_liqd_list), min(nontp_short_pr_list),
                       max_nontp_short_term),
                        position=title_position,
                        )
            else:
                plt.title(
                    "wr : %.2f\nlen(short_td) : %s\nmin_pr : %.2f\nacc_pr : %.2f\n lvrg %s\nliqd : %.2f\nmean_tr : %.2f\nmean_short_pgfr : %.2f\nnontp_short_liqd_cnt : %s"
                    % (short_wr, len(short_np_pr[short_np_pr != 1]), np.min(short_np_pr), total_short_pr[-1], lvrg,
                       min(short_liqd_list), mean_short_tr, mean_short_pgfr, len(nontp_short_liqd_list)),
                    position=title_position,
                    )

            print('supblot232 passed')

            #         long only      #
            long_np_pr = np.array(long_list)
            # long_np_pr = (np.array(long_list) - 1) * lvrg + 1

            total_long_pr = np.cumprod(long_np_pr)
            long_wr = len(long_np_pr[long_np_pr > 1]) / len(long_np_pr[long_np_pr != 1])

            # long_cumprod_profit = np.cumprod(long_np_pr[long_np_pr > 1])[-1]
            # long_cumprod_loss = np.cumprod(long_np_pr[long_np_pr < 1])[-1]
            # long_pr_tr = long_cumprod_profit * long_cumprod_loss

            np_long_tp_ratio_list = np.array(long_tp_ratio_list)
            mean_long_tr = np.mean(np_long_tp_ratio_list[np.isnan(np_long_tp_ratio_list) == 0])

            long_pr_gap = (long_np_pr - 1) / lvrg + fee
            long_tp_gap = long_pr_gap[long_pr_gap > 0]
            # mean_long_tp_gap = np.mean(long_pr_gap[long_pr_gap > 0])
            # mean_long_ls_gap = np.mean(long_pr_gap[long_pr_gap < 0])

            mean_long_pgfr = np.mean((long_tp_gap - fee) / abs(long_tp_gap + fee))

            plt.subplot(233)
            plt.plot(total_long_pr)
            if len(nontp_long_liqd_list) != 0:

                max_nontp_long_term = len(res_df) - nontp_long_indexs[0]

                plt.title(
                    "wr : %.2f\nlen(long_td) : %s\nmin_pr : %.2f\nacc_pr : %.2f\n lvrg %s\nliqd : %.2f\nmean_tr : %.2f\nmean_long_pgfr : %.2f\nnontp_long_liqd_cnt : %s\nnontp_long_liqd : %.2f\nontp_long_liqd_pr : %.2f\nmax_nontp_long_term : %s"
                    % (long_wr, len(long_np_pr[long_np_pr != 1]), np.min(long_np_pr), total_long_pr[-1], lvrg,
                       min(long_liqd_list), mean_long_tr, mean_long_pgfr,
                       len(nontp_long_liqd_list), min(nontp_long_liqd_list), min(nontp_long_pr_list),
                       max_nontp_long_term),
                    position=title_position,
                    )
            else:
                plt.title(
                    "wr : %.2f\nlen(long_td) : %s\nmin_pr : %.2f\nacc_pr : %.2f\n lvrg %s\nliqd : %.2f\nmean_tr : %.2f\nmean_long_pgfr : %.2f\nnontp_long_liqd_cnt : %s"
                    % (long_wr, len(long_np_pr[long_np_pr != 1]), np.min(long_np_pr), total_long_pr[-1], lvrg,
                       min(long_liqd_list), mean_long_tr, mean_long_pgfr, len(nontp_long_liqd_list)),
                    position=title_position,
                    )

            print('supblot233 passed')

            #     reversion adjustment      #
            # rev_np_pr = 1 / (np.array(pr_list) + fee) - fee
            rev_np_pr = (1 / ((np.array(pr_list) - 1) / lvrg + fee + 1) - fee - 1) * lvrg + 1
            # rev_np_pr = (1 / (np.array(pr_list) + fee) - fee - 1) * lvrg + 1

            rev_total_pr = np.cumprod(rev_np_pr)
            rev_wr = len(rev_np_pr[rev_np_pr > 1]) / len(rev_np_pr[rev_np_pr != 1])

            # plt.subplot(122)
            plt.subplot(234)
            plt.plot(rev_total_pr)
            plt.title("wr : %.2f\nmin_pr : %.2f\nacc_pr : %.2f\n lvrg %s" % (
            rev_wr, np.min(rev_np_pr), rev_total_pr[-1], lvrg))

            #         short       #
            # rev_short_np_pr = 1 / (np.array(short_list) + fee) - fee
            rev_short_np_pr = (1 / ((np.array(short_list) - 1) / lvrg + fee + 1) - fee - 1) * lvrg + 1
            # rev_short_np_pr = (1 / (np.array(short_list) + fee) - fee - 1) * lvrg + 1

            rev_total_short_pr = np.cumprod(rev_short_np_pr)
            rev_short_wr = len(rev_short_np_pr[rev_short_np_pr > 1]) / len(rev_short_np_pr[rev_short_np_pr != 1])

            # plt.subplot(122)
            plt.subplot(235)
            plt.plot(rev_total_short_pr)
            plt.title("wr : %.2f\nmin_pr : %.2f\nacc_pr : %.2f\n lvrg %s" % (
            rev_short_wr, np.min(rev_short_np_pr), rev_total_short_pr[-1], lvrg))

            #         long       #
            # rev_long_np_pr = 1 / (np.array(long_list) + fee) - fee
            rev_long_np_pr = (1 / ((np.array(long_list) - 1) / lvrg + fee + 1) - fee - 1) * lvrg + 1

            rev_total_long_pr = np.cumprod(rev_long_np_pr)
            rev_long_wr = len(rev_long_np_pr[rev_long_np_pr > 1]) / len(rev_long_np_pr[rev_long_np_pr != 1])

            # plt.subplot(122)
            plt.subplot(236)
            plt.plot(rev_total_long_pr)
            plt.title("wr : %.2f\nmin_pr : %.2f\nacc_pr : %.2f\n lvrg %s" % (
            rev_long_wr, np.min(rev_long_np_pr), rev_total_long_pr[-1], lvrg))

            # plt.show()

            h_np_pr = np.array(h_pr_list)
            # h_rev_np_pr = 1 / (np.array(h_pr_list) + fee) - fee    # define, for plot_check below cell
            h_rev_np_pr = (1 / ((np.array(h_pr_list) - 1) / lvrg + fee + 1) - fee - 1) * lvrg + 1

            # --------------------- h pr plot --------------------- #
            if len(h_np_pr[h_np_pr != 1]) != 0:
                plt.figure(figsize=(16, 12))
                # plt.suptitle(key + " hedge")

                h_total_pr = np.cumprod(h_np_pr)
                h_wr = len(h_np_pr[h_np_pr > 1]) / len(h_np_pr[h_np_pr != 1])

                # plt.subplot(121)
                plt.subplot(231)
                plt.plot(h_total_pr)
                plt.title(
                    "wr : %.2f\nmin_pr : %.2f\nacc_pr : %.2f\n lvrg %s" % (h_wr, np.min(h_np_pr), h_total_pr[-1], lvrg))
                # plt.show()

                #         short only      #
                h_short_np_pr = np.array(h_short_list)

                h_total_short_pr = np.cumprod(h_short_np_pr)
                h_short_wr = len(h_short_np_pr[h_short_np_pr > 1]) / len(h_short_np_pr[h_short_np_pr != 1])

                plt.subplot(232)
                plt.plot(h_total_short_pr)
                plt.title("wr : %.2f\nmin_pr : %.2f\nacc_pr : %.2f\n lvrg %s" % (
                h_short_wr, np.min(h_short_np_pr), h_total_short_pr[-1], lvrg))

                #         long only      #
                h_long_np_pr = np.array(h_long_list)

                h_total_long_pr = np.cumprod(h_long_np_pr)
                h_long_wr = len(h_long_np_pr[h_long_np_pr > 1]) / len(h_long_np_pr[h_long_np_pr != 1])

                plt.subplot(233)
                plt.plot(h_total_long_pr)
                plt.title("wr : %.2f\nmin_pr : %.2f\nacc_pr : %.2f\n lvrg %s" % (
                h_long_wr, np.min(h_long_np_pr), h_total_long_pr[-1], lvrg))

                #     reversion adjustment      #

                h_rev_total_pr = np.cumprod(h_rev_np_pr)
                h_rev_wr = len(h_rev_np_pr[h_rev_np_pr > 1]) / len(h_rev_np_pr[h_rev_np_pr != 1])

                # plt.subplot(122)
                plt.subplot(234)
                plt.plot(h_rev_total_pr)
                plt.title("wr : %.2f\nmin_pr : %.2f\nacc_pr : %.2f\n lvrg %s" % (
                h_rev_wr, np.min(h_rev_np_pr), h_rev_total_pr[-1], lvrg))

                #         short       #
                # h_rev_short_np_pr = 1 / (np.array(h_short_list) + fee) - fee
                h_rev_short_np_pr = (1 / ((np.array(h_short_list) - 1) / lvrg + fee + 1) - fee - 1) * lvrg + 1

                h_rev_total_short_pr = np.cumprod(h_rev_short_np_pr)
                h_rev_short_wr = len(h_rev_short_np_pr[h_rev_short_np_pr > 1]) / len(
                    h_rev_short_np_pr[h_rev_short_np_pr != 1])

                # plt.subplot(122)
                plt.subplot(235)
                plt.plot(h_rev_total_short_pr)
                plt.title("wr : %.2f\nmin_pr : %.2f\nacc_pr : %.2f\n lvrg %s" % (
                h_rev_short_wr, np.min(h_rev_short_np_pr), h_rev_total_short_pr[-1], lvrg))

                #         long       #
                # h_rev_long_np_pr = 1 / (np.array(h_long_list) + fee) - fee
                h_rev_long_np_pr = (1 / ((np.array(h_long_list) - 1) / lvrg + fee + 1) - fee - 1) * lvrg + 1

                h_rev_total_long_pr = np.cumprod(h_rev_long_np_pr)
                h_rev_long_wr = len(h_rev_long_np_pr[h_rev_long_np_pr > 1]) / len(
                    h_rev_long_np_pr[h_rev_long_np_pr != 1])

                # plt.subplot(122)
                plt.subplot(236)
                plt.plot(h_rev_total_long_pr)
                plt.title("wr : %.2f\nmin_pr : %.2f\nacc_pr : %.2f\n lvrg %s" % (
                h_rev_long_wr, np.min(h_rev_long_np_pr), h_rev_total_long_pr[-1], lvrg))

                # plt.show()

        except Exception as e:
            print('error in pr plot :', e)

        print()

        break  # indi. loop

    plt.savefig(dir_path + "/back_pr/back_pr.png")
    print("back_pr.png saved !")

    return ep_tp_list, trade_list
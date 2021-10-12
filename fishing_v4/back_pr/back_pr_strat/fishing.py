import matplotlib.pyplot as plt
import numpy as np
from funcs.funcs_for_trade import intmin

def min_max_scale(npy_x):
    return (npy_x - np.min(npy_x)) / (np.max(npy_x) - np.min(npy_x))


#       Todo        #
#        1. colab -> pycharm 시 수정해야할 사항     #
#          a. key 주석처리
#          b. intmin function import

fee = 0.0004
lvrg = 5
# lvrg = 2

# gap = 0.0002
# gap = 0.0001
gap = 0.00005

p_qty_divider = 1.5

# ------- ep set ------- #
entry_type = 'limit'
# entry_type = 'market'

# ep_gap = 0.0005
ep_gap = 0.0002
# ep_gap = 0.0001
# ep_gap = 0.00005

# ------- tp set ------- #
# exit_type = 'limit'
exit_type = 'market'
static_tp = 1

# ------- lvrg set ------- #
static_lvrg = 1
target_pct = 0.04
hl_lookback = 10

# ------- inversion set ------- #
inversion = 0

if inversion:
    short_entry = [1, 2]
    long_entry = [-1, -2]
else:
    short_entry = [-1, -2]
    long_entry = [1, 2]

tp_cut_ratio = 0.7

fdist_thresh = 1.25

# ----------------- indicator ----------------- #
# ------- shift_size ------- #
cloud_shift_size = 1
sma_shift_size = 1
close_shift_size = 1

# ------- lb ------- #
cloud_lookback = 30
cloud_lookback = 69
# cloud_lookback = 150
# cloud_lookback = 10

sma_lookback = 100
# sma_lookback = 100

sar_lookback = 5

# ------- indi. params ------- #
sma = 'sma1'

# sma_period = 250
sma_period = 100

fisher_upper = 1.5
fisher_lower = -1.5

stoch_upper = 63
stoch_lower = 37

cctbbo_upper = 80
cctbbo_lower = 20

senkoua_list = ['senkou_a1']
senkoub_list = ['senkou_b1']


def back_pr_check(res_df, dir_path):

    # print("cloud_lookback :", cloud_lookback)
    lvrg = 2

    for senkou_a, senkou_b in zip(senkoua_list, senkoub_list):
        # for sma4_period in range(5, 30, 2):
        # for sma4_period in range(13, 14, 2):
        # for cloud_lookback in np.arange(5, 100, 3):

        # print("sma4_period :", sma4_period)
        print("cloud_lookback :", cloud_lookback)

        # -------------------- additional indicators -------------------- #
        # senkou_a, senkou_b = 'senkou_a1', 'senkou_b1'

        cloud_top = np.max(res_df[[senkou_a, senkou_b]], axis=1)
        cloud_bottom = np.min(res_df[[senkou_a, senkou_b]], axis=1)

        under_top = res_df['close'].shift(cloud_shift_size) <= cloud_top.shift(cloud_shift_size)
        over_top = res_df['close'].shift(cloud_shift_size) >= cloud_top.shift(cloud_shift_size)

        over_bottom = res_df['close'].shift(cloud_shift_size) >= cloud_bottom.shift(cloud_shift_size)
        under_bottom = res_df['close'].shift(cloud_shift_size) >= cloud_bottom.shift(cloud_shift_size)

        # --------------- 2nd middle --------------- #
        upper_middle = (res_df['middle_line'] + res_df['min_upper']) / 2
        lower_middle = (res_df['middle_line'] + res_df['max_lower']) / 2

        # --------------- sma --------------- #
        res_df['sma1'] = res_df['close'].rolling(sma_period).mean()
        # # print(res_df['sma5'].tail())

        # --------------- htf sma --------------- #
        # fourth_df = pd.read_excel(date_path4 + key.replace("_4h1d_backi2", ""), index_col=0)

        # if "sma4" in res_df.columns:
        #   res_df.drop("sma4", axis=1, inplace=True)

        # fourth_df['sma'] = fourth_df['close'].rolling(sma4_period).mean()
        # res_df = res_df.join(pd.DataFrame(index=res_df.index, data=to_lower_tf(res_df, fourth_df, [-1]), columns=['sma4']))

        # --------------- open close ep --------------- #
        # short_ep = res_df['open'] * (1.037)
        # long_ep = res_df['open'] * (1 / 1.037)
        short_ep = res_df['close'].shift(1) * (1.037)
        long_ep = res_df['close'].shift(1) * (1 / 1.037)

        # -------------------- short = -1 -------------------- #
        # --------------- timestamp entry --------------- #
        # entry = np.where((intmin(res_df.index) in [0, 30])
        #                       , -1, 0)

        # --------------- st entry --------------- #
        # entry = np.where((res_df['close'].shift(1) <= short_ep) &
        #                       (res_df['high'] >= short_ep)
        #                       , -1, 0)
        # entry = np.where((res_df['high'].shift(1) <= upper_middle) &
        # entry = np.where((res_df['high'].shift(1) <= res_df['middle_line']) &
        #                     (res_df['high'] >= short_ep)
        #                     , -1, 0)

        # entry = np.where((res_df['close'].shift(1) > short_ep)
        #                   , -2, entry)

        # entry = np.where((res_df['close'].shift(1) >= short_ep) &
        #                 # (long_ep <= res_df['high']) &
        #                 (res_df['close'] <= short_ep)
        #                 , -1, 0)

        # --------------- sar entry --------------- #
        # # entry = np.where((res_df['close'] <= res_df['sar2']) &
        # #                   (res_df['close'].shift(1) > res_df['sar2'].shift(1))
        # #                   , -1, 0)
        # entry = np.where((res_df['close'] <= res_df['sar2']) &
        #                  (res_df['low'].shift(1) > res_df['sar2'].shift(1)) &
        #                  (res_df['low'].shift(2) > res_df['sar2'].shift(2))
        #                   , -1, 0)
        # entry = np.where((res_df['sar1'].shift(1) > res_df['low']) &
        #                  (res_df['sar1'].shift(2) <= res_df['low'].shift(1))
        #                   , -1, 0)

        # --------------- sar pb line : 정확한 진입시점은 아니지만 pb line 의 기준이 댐 --------------- #
        # entry = np.where((res_df['sar2_uptrend'].shift(1) == 1) &
        #                  (res_df['sar2_uptrend'] == 0)
        #                   , -1, 0)

        # --------------- fisher entry --------------- #
        # entry = np.where((res_df['fisher30'].shift(1) >= res_df['fisher30']) &
        #                   (res_df['fisher30'].shift(2) <= res_df['fisher30']).shift(1) &
        #                   (res_df['fisher30'].shift(1) >= fisher_upper)
        #                   , -1, 0)

        # --------------- cctbbo entry --------------- #
        # entry = np.where((res_df['cctbbo'].shift(1) >= res_df['cctbbo']) &
        #                   (res_df['cctbbo'].shift(2) <= res_df['cctbbo']).shift(1) &
        #                   (res_df['cctbbo'].shift(1) >= cctbbo_upper)
        #                   , -1, 0)

        # --------------- cloud entry --------------- #
        # cloud_bottom = np.min(res_df[[senkou_a, senkou_b]], axis=1)

        # entry = np.where((res_df['close'] < cloud_bottom) &
        #                   (res_df['close'].shift(1) >= cloud_bottom.shift(1))
        #                   , -1, 0)

        #       long = 1     #

        # --------------- timestamp entry --------------- #
        # entry = np.where((np.array(intmin(res_df.index)) in [0, 30])

        int_min_ts = np.array(list(map(lambda x: intmin(x), res_df.index)))
        # entry = np.where((intmin(res_df.index) == 0)
        entry = np.where((int_min_ts == 0) | (int_min_ts == 30)
                         , 1, 0)

        # print("int_min_ts :", int_min_ts)
        # print("entry :", entry)
        # break

        # --------------- st entry --------------- #
        # entry = np.where((res_df['close'].shift(1) >= long_ep) &
        #                   (res_df['low'] <= long_ep)
        #                   , 1, entry)
        # entry = np.where((res_df['low'].shift(1) >= lower_middle) &
        # entry = np.where((res_df['low'].shift(1) >= res_df['middle_line']) &
        #                   (res_df['low'] <= long_ep)
        #                   , 1, entry)

        # entry = np.where((res_df['close'].shift(1) < long_ep)
        #                   , 2, entry)

        # entry = np.where((res_df['close'].shift(1) <= long_ep) &
        #                   # (long_ep <= res_df['high']) &
        #                   (res_df['close'] >= long_ep)
        #                   , 1, entry)

        # --------------- sar entry --------------- #
        # # entry = np.where((res_df['close'] >= res_df['sar2']) &
        # #                   (res_df['close'].shift(1) < res_df['sar2'].shift(1))
        # #                   , 1, entry)
        # --------------- sar pb line : 정확한 진입시점은 아니지만 pb line 의 기준이 댐 --------------- #
        # entry = np.where((res_df['sar2_uptrend'].shift(1) == 0) &
        #                  (res_df['sar2_uptrend'] == 1)
        #                   , 1, entry)

        # #       lb sar 이 high 보다 커야함      #
        # entry = np.where((res_df['close'] >= res_df['sar2']) &
        #                  (res_df['high'].shift(1) < res_df['sar2'].shift(1)) &
        #                  (res_df['high'].shift(2) < res_df['sar2'].shift(2))
        #                   , 1, entry)
        # entry = np.where((res_df['sar1'].shift(1) < res_df['high']) &
        #                  (res_df['sar1'].shift(2) >= res_df['high'].shift(1))
        #                   , 1, entry)

        # --------------- fisher entry --------------- #
        # entry = np.where((res_df['fisher30'].shift(1) <= res_df['fisher30']) &
        #                   (res_df['fisher30'].shift(2) >= res_df['fisher30']).shift(1) &
        #                   (res_df['fisher30'].shift(1) <= fisher_lower)
        #                   , 1, entry)

        # --------------- cctbbo entry --------------- #
        # entry = np.where((res_df['cctbbo'].shift(1) <= res_df['cctbbo']) &
        #                   (res_df['cctbbo'].shift(2) >= res_df['cctbbo']).shift(1) &
        #                   (res_df['cctbbo'].shift(1) <= cctbbo_lower)
        #                   , 1, entry)

        # --------------- cloud entry --------------- #
        # cloud_top = np.max(res_df[[senkou_a, senkou_b]], axis=1)

        # entry = np.where((res_df['close'] > cloud_top) &
        #                   (res_df['close'].shift(1) <= cloud_top.shift(1))
        #                   , 1, entry)

        # print("len(entry) :", len(entry))
        # print("np.sum(entry == -1) :", np.sum(entry == -1))
        # print("np.sum(entry == 1) :", np.sum(entry == 1))
        # break

        #       1-2. tp line = middle line 조금 이내         #
        # --------------- gap range tp --------------- #
        # gap_range = 0.5
        # gap_range = 1

        # short_cut = res_df['high'].rolling(hl_lookback).max()
        # long_cut = res_df['low'].rolling(hl_lookback).min()

        # short_tp = res_df['close'] - gap_range * (short_cut - res_df['close'])
        # long_tp = res_df['close'] + gap_range * (res_df['close'] - long_cut)

        # --------------- st limit tp --------------- #

        # short_tp = res_df['middle_line'] * (1 + gap)
        # long_tp = res_df['middle_line'] * (1 - gap)

        # short_tp = res_df['middle_line']
        # long_tp = res_df['middle_line']

        # short_tp = lower_middle * (1 + gap)
        # long_tp = upper_middle * (1 - gap)

        short_tp = upper_middle
        long_tp = lower_middle

        # short_tp = (res_df['middle_line'] + res_df['min_upper']) / 2
        # long_tp = (res_df['middle_line'] + res_df['max_lower']) / 2

        # short_tp = res_df['close'] - (res_df['middle_line'] - res_df['close']) * tp_cut_ratio
        # long_tp = res_df['close'] + (res_df['close'] - res_df['middle_line']) * tp_cut_ratio

        # short_tp2 = res_df['middle_line'] * (1 + gap)
        # long_tp2 = res_df['middle_line'] * (1 - gap)

        # --------------- sar limit tp --------------- #
        # short_tp = res_df['sar2'].shift(1) - abs(res_df['sar2'] - res_df['sar2'].shift(1)) * 0.5
        # long_tp = res_df['sar2'].shift(1) + abs(res_df['sar2'] - res_df['sar2'].shift(1)) * 0.5

        # short_tp2 = res_df['sar2'].shift(1)
        # long_tp2 = res_df['sar2'].shift(1)

        # short_cut = res_df['sar2']
        # long_cut = res_df['sar2']

        # --------------- set partial tp --------------- #

        short_tps = [short_tp]
        long_tps = [long_tp]

        # short_tps = [short_tp2]
        # long_tps = [long_tp2]

        # short_tps = [short_tp2, short_tp]
        # long_tps = [long_tp2, long_tp]

        # short_tps = [short_tp, short_tp2]
        # long_tps = [long_tp, long_tp2]

        #       trading : 여기도 체결 결과에 대해 묘사함       #
        trade_list = []
        h_trade_list = []
        lvrg_list = []
        open_list = []

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

                # -------------- ep scheduling -------------- #
                # if  (res_df['close'].iloc[i] <= lower_middle.iloc[i]):
                # if abs((res_df['close'].iloc[i] - short_ep.iloc[i]) / short_ep.iloc[i]) < ep_gap:
                # if abs((res_df['close'].iloc[i] - upper_middle.iloc[i]) / upper_middle.iloc[i]) < ep_gap:
                # # if abs((res_df['close'].iloc[i] - res_df['middle_line'].iloc[i]) / res_df['middle_line'].iloc[i]) < ep_gap:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # -------------- distance protection -------------- #
                # tp_dist = (res_df['close'].iloc[i] - short_tp.iloc[i])
                # cut_dist = (res_df['middle_line'].iloc[i] - res_df['close'].iloc[i])
                # if tp_dist / cut_dist >= tp_cut_ratio:
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

                # -------------- sma const. -------------- #
                # if res_df['close'].iloc[i] < res_df[sma].iloc[i]: # and \
                # #   short_ep.iloc[i] <= res_df['sma1'].shift(sma_shift_size).iloc[i]:
                # # # under_sma = short_ep <= res_df['sma'].shift(sma_shift_size)
                # # # if np.sum(under_sma.iloc[i + 1 - sma_lookback:i + 1]) == sma_lookback:
                #   pass
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
                # if np.sum(res_df[['minor_ST1_Trend', 'minor_ST2_Trend', 'minor_ST3_Trend']].iloc[i]) <= -1:
                # if np.sum(res_df[['minor_ST1_Trend', 'minor_ST2_Trend', 'minor_ST3_Trend']].iloc[i]) <= -3:
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

                # -------------- sar const. -------------- #
                # if res_df['sar2'].iloc[i] > res_df['high'].iloc[i] and res_df['sar3'].iloc[i] > res_df['high'].iloc[i]:
                # if res_df['sar2'].iloc[i] > res_df['high'].iloc[i]: # and \
                # if  res_df['sar3'].iloc[i] > res_df['high'].iloc[i]:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # -------------- limit waiting const. -------------- #
                # print("initial_i :", initial_i)
                # print("res_df['sar2_uptrend'].iloc[initial_i - 1] :", res_df['sar2_uptrend'].iloc[initial_i - 1])
                # print("res_df['sar2_uptrend'].iloc[initial_i] :", res_df['sar2_uptrend'].iloc[initial_i])

                # print("res_df['sar2'].iloc[initial_i - 1] :", res_df['sar2'].iloc[initial_i - 5:initial_i])
                # print("res_df['sar2'].iloc[initial_i] :", res_df['sar2'].iloc[initial_i:initial_i + 5])
                # print("short_ep.iloc[initial_i] :", short_ep.iloc[initial_i])

                entry_done = False
                for e_j in range(i, len(res_df)):

                    #             Todo            #
                    #             1. ep 설정
                    #             1-1. close 가 sar_change 이전 sar 을 cross 한 경우만 진입
                    if res_df['high'].iloc[e_j] >= short_ep.iloc[initial_i]:
                        entry_done = True
                        # print("res_df['high'].iloc[e_j] :", res_df['high'].iloc[e_j])
                        # print("e_j :", e_j)
                        break

                    #             2. limit 대기 시간 설정
                    #             2-1. tp 하거나, cut 조건이 성립되는 경우 limit 취소
                    if intmin(res_df.index[e_j]) in [29, 59]:
                        break

                    # if res_df['low'].iloc[e_j] <= short_tp.iloc[initial_i]:
                    #   break

                    # if res_df['close'].iloc[e_j] > res_df['middle_line'].iloc[e_j]:
                    # if res_df['close'].iloc[e_j] > short_cut.iloc[initial_i]: # or \
                    #   # res_df['sar2_uptrend'].iloc[e_j] == 1: # or \

                    # # if res_df['close'].iloc[e_j] > res_df['sar2'].iloc[e_j]:
                    #   break

                i = e_j
                # print("i = e_j :", i)

                if entry_done:
                    pass
                else:
                    i += 1
                    if i >= len(res_df):
                        break
                    continue

                # print("initial_i :", initial_i)
                # print()

                open_list.append(initial_i)

                if entry_type is 'market':
                    ep_list = [res_df['close'].iloc[initial_i]]
                else:
                    if entry[initial_i] == -1:
                        ep_list = [short_ep.iloc[initial_i]]
                    else:
                        ep_list = [res_df['open'].iloc[initial_i]]

                if not static_lvrg:
                    # lvrg = target_pct / (res_df['high'].rolling(hl_lookback).max().iloc[initial_i] / res_df['close'].iloc[initial_i] - 1)
                    lvrg = target_pct / (short_cut.iloc[initial_i] / short_ep.iloc[initial_i] - 1)
                    lvrg = int(min(50, lvrg))
                    lvrg = max(lvrg, 1)
                    lvrg_list.append(lvrg)

                # ep_idx_list = [initial_i]
                ep_idx_list = [e_j]
                tp_list = []
                tp_idx_list = []

                partial_tp_cnt = 0
                hedge_cnt = 1

                h_ep, h_tp = None, None
                h_i, h_j = None, None

                trade_done = False

                # for j in range(i + 1, len(res_df)):
                for j in range(i, len(res_df)):

                    if static_tp:
                        tp_j = initial_i
                    else:
                        tp_j = j

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

                        # -------------- st tp -------------- #
                        # if res_df['close'].iloc[j] > res_df['middle_line'].iloc[j]:

                        # -------------- fisher tp -------------- #
                        # if entry[j] == 1:

                        # -------------- timestamp tp -------------- #
                        if intmin(res_df.index[j]) in [29, 59]:

                            tp = res_df['close'].iloc[j]
                            # tp = res_df['open'].iloc[j]
                            trade_done = True

                            if trade_done:
                                tp_state_list.append("short close tp")

                            tp_list.append(tp)
                            tp_idx_list.append(j)

                    # -------------- cut -------------- #
                    # if not trade_done:

                    #   # -------------- macd -------------- #
                    #   # if res_df['macd_hist3'].iloc[j] > 0:  #  macd cut
                    #   # if res_df['macd_hist3'].iloc[i] < 0 and res_df['macd_hist3'].iloc[j] > 0:

                    #   # -------------- st -------------- #
                    #   # if res_df['close'].iloc[j] > res_df['middle_line'].iloc[j]:
                    #   # if res_df['close'].iloc[j] > res_df['minor_ST3_Up'].iloc[j]:
                    #   # if res_df['close'].iloc[j] > upper_middle.iloc[j]:
                    #   # if res_df['close'].iloc[j] > res_df['minor_ST1_Up'].iloc[j]:

                    #   # -------------- sma -------------- #
                    #   # if res_df['close'].iloc[j] > res_df[sma].iloc[j]:

                    #   # -------------- sar -------------- #
                    #   # if res_df['close'].iloc[j] > res_df['minor_ST3_Up'].iloc[j] \
                    #   #   or res_df['sar2'].iloc[j] <= res_df['high'].iloc[j]:
                    #   if res_df['close'].iloc[j] > short_cut.iloc[initial_i]: # or \
                    #     # res_df['sar2_uptrend'].iloc[j] == 1: # or \
                    #   # if res_df['close'].iloc[j] > res_df['sar2'].iloc[j]:

                    #   # -------------- hl -------------- #
                    #   # if res_df['close'].iloc[j] > short_cut.iloc[tp_j]:

                    #     tp = res_df['close'].iloc[j]
                    #     # tp = res_df['open'].iloc[j]
                    #     trade_done = True
                    #     tp_state_list.append("short close_cut")

                    #     tp_list.append(tp)
                    #     tp_idx_list.append(j)

                    # -------------- non tp -------------- #
                    if j == len(res_df) - 1:
                        trade_done = True
                        tp = res_df['close'].iloc[j]
                        tp_list.append(tp)
                        tp_idx_list.append(j)

                    # -------------- append trade data -------------- #
                    if trade_done:

                        # if entry[initial_i] == -1:
                        #   # ep = res_df['close'].iloc[initial_i]
                        #   # ep = short_ep.iloc[initial_i]
                        #   # ep_list[0] = short_ep.iloc[initial_i]
                        #   pass
                        # else:
                        #   # ep = res_df['open'].iloc[initial_i]
                        #   ep_list[0] = res_df['open'].iloc[initial_i]

                        # ep = short_ep.iloc[initial_i]
                        # ep = res_df['close'].iloc[initial_i - 1]

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

                            h_ep_tp_list.append((h_ep, h_tp))
                            h_trade_list.append([initial_i, h_i, h_j])

                            pr_list.append(temp_pr)
                            short_list.append(temp_pr)

                            h_pr_list.append(hedge_pr)
                            h_short_list.append(hedge_pr)

                            i = j
                            break

                        else:

                            #         tp 미체결 survey        #
                            nontp_liqd_list.append(short_liq)
                            nontp_short_liqd_list.append(short_liq)
                            nontp_short_indexs.append(i)
                            nontp_short_ep_list.append(ep_list[0])

                            nontp_short_pr = (ep_list[0] / tp - fee - 1) * lvrg + 1
                            nontp_pr_list.append(nontp_short_pr)
                            nontp_short_pr_list.append(nontp_short_pr)



            #                  long  phase                #
            elif entry[i] in long_entry:  # inversion

                initial_i = i

                # -------------- ep scheduling -------------- #
                # if res_df['close'].iloc[i] >= upper_middle.iloc[i]:
                # if abs((res_df['close'].iloc[i] - long_ep.iloc[i]) / long_ep.iloc[i]) < ep_gap:
                # if abs((res_df['close'].iloc[i] - lower_middle.iloc[i]) / lower_middle.iloc[i]) < ep_gap:
                # # if abs((res_df['close'].iloc[i] - res_df['middle_line'].iloc[i]) / res_df['middle_line'].iloc[i]) < ep_gap:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # -------------- distance protection -------------- #
                # tp_dist = (long_tp.iloc[i] - res_df['close'].iloc[i])
                # cut_dist = (res_df['close'].iloc[i] - res_df['middle_line'].iloc[i])
                # if tp_dist / cut_dist >= tp_cut_ratio:
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

                # -------------- sma const. -------------- #
                # if res_df['close'].iloc[i] > res_df[sma].iloc[i]: # and \
                # #   long_ep.iloc[i] >= res_df['sma1'].shift(sma_shift_size).iloc[i]:
                # # # upper_sma = long_ep >= res_df['sma'].shift(sma_shift_size)
                # # # if np.sum(upper_sma.iloc[i + 1 - sma_lookback:i + 1]) == sma_lookback:
                #   pass
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
                # if np.sum(res_df[['minor_ST1_Trend', 'minor_ST2_Trend', 'minor_ST3_Trend']].iloc[i]) >= 1:
                # if np.sum(res_df[['minor_ST1_Trend', 'minor_ST2_Trend', 'minor_ST3_Trend']].iloc[i]) >= 3:
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

                # -------------- sar const. -------------- #
                # # if res_df['sar2'].iloc[i] < res_df['low'].iloc[i] and res_df['sar3'].iloc[i] < res_df['low'].iloc[i]:
                # if res_df['sar2'].iloc[i] < res_df['low'].iloc[i]: # and \
                # if  res_df['sar3'].iloc[i] < res_df['low'].iloc[i]:
                #   pass
                # else:
                #   i += 1
                #   if i >= len(res_df):
                #     break
                #   continue

                # -------------- limit waiting const. -------------- #
                entry_done = False
                for e_j in range(i, len(res_df)):

                    #             Todo            #
                    #             1. ep 설정
                    #             1-1. close 가 sar_change 이전 sar 을 cross 한 경우만 진입
                    if res_df['low'].iloc[e_j] <= long_ep.iloc[initial_i]:
                        entry_done = True
                        # print("e_j :", e_j)
                        break

                    #             2. limit 대기 시간 설정
                    #             2-1. tp 하거나, cut 조건이 성립되는 경우 limit 취소
                    #             2-1-1. timestamp = 29 or 59
                    if intmin(res_df.index[e_j]) in [29, 59]:
                        break

                    # if res_df['close'].iloc[e_j] < res_df['middle_line'].iloc[e_j]:
                    # if res_df['close'].iloc[e_j] < long_cut.iloc[initial_i]: # or \
                    #   # res_df['sar2_uptrend'].iloc[e_j] == 0 or \
                    #   # res_df['close'].iloc[e_j] < res_df['sar2'].iloc[e_j]:
                    #   break

                i = e_j
                # print("i = e_j :", i)

                if entry_done:
                    pass
                else:
                    i += 1
                    if i >= len(res_df):
                        break
                    continue

                open_list.append(initial_i)

                if entry_type is 'market':
                    ep_list = [res_df['close'].iloc[initial_i]]
                else:
                    if entry[initial_i] == 1:
                        ep_list = [long_ep.iloc[initial_i]]
                    else:
                        ep_list = [res_df['open'].iloc[initial_i]]

                if not static_lvrg:
                    # lvrg = target_pct / (res_df['close'].iloc[initial_i] / res_df['low'].rolling(hl_lookback).min().iloc[initial_i] - 1)
                    lvrg = target_pct / (long_ep.iloc[initial_i] / long_cut.iloc[initial_i] - 1)
                    lvrg = int(min(50, lvrg))
                    lvrg = max(1, lvrg)
                    lvrg_list.append(lvrg)

                # ep_idx_list = [initial_i]
                ep_idx_list = [e_j]
                tp_list = []
                tp_idx_list = []

                partial_tp_cnt = 0
                hedge_cnt = 1

                h_ep, h_tp = None, None
                h_i, h_j = None, None

                trade_done = False

                # for j in range(i + 1, len(res_df)):
                for j in range(i, len(res_df)):

                    if static_tp:
                        tp_j = initial_i
                    else:
                        tp_j = j

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

                        # -------------- st tp -------------- #
                        # if res_df['close'].iloc[j] < res_df['middle_line'].iloc[j]:

                        # -------------- fisher tp -------------- #
                        # if entry[j] == -1:

                        # -------------- timestamp tp -------------- #
                        if intmin(res_df.index[j]) in [29, 59]:

                            tp = res_df['close'].iloc[j]
                            # tp = res_df['open'].iloc[j]
                            trade_done = True

                            if trade_done:
                                tp_state_list.append("long close tp")

                            tp_list.append(tp)
                            tp_idx_list.append(j)

                    # -------------- cut -------------- #
                    # if not trade_done:

                    #   # -------------- macd -------------- #
                    #   # if res_df['macd_hist3'].iloc[j] < 0:
                    #   # # if res_df['macd_hist3'].iloc[i] > 0 and res_df['macd_hist3'].iloc[j] < 0:

                    #   # -------------- st -------------- #
                    #   # if res_df['close'].iloc[j] < res_df['middle_line'].iloc[j]:
                    #   # if res_df['close'].iloc[j] < res_df['minor_ST3_Down'].iloc[j]:
                    #   # if res_df['close'].iloc[j] < lower_middle.iloc[j]:
                    #   # if res_df['close'].iloc[j] < res_df['minor_ST1_Down'].iloc[j]:

                    #   # -------------- sma -------------- #
                    #   # if res_df['close'].iloc[j] < res_df[sma].iloc[j]:

                    #   # -------------- sar -------------- #
                    #   # if res_df['close'].iloc[j] < res_df['minor_ST3_Down'].iloc[j] \
                    #   #   or res_df['sar2'].iloc[j] >= res_df['low'].iloc[j]:
                    #   if res_df['close'].iloc[j] < long_cut.iloc[initial_i]: # or \
                    #     #  res_df['sar2_uptrend'].iloc[j] == 0 or \
                    #     #  res_df['close'].iloc[j] < res_df['sar2'].iloc[j]:

                    #   # -------------- hl -------------- #
                    #   # if res_df['close'].iloc[j] < long_cut.iloc[tp_j]:

                    #     tp = res_df['close'].iloc[j]
                    #     # tp = res_df['open'].iloc[j]
                    #     tp_state_list.append("long close_cut")
                    #     trade_done = True

                    #     tp_list.append(tp)
                    #     tp_idx_list.append(j)

                    # -------------- non tp -------------- #
                    if j == len(res_df) - 1:
                        trade_done = True
                        tp = res_df['close'].iloc[j]
                        tp_list.append(tp)
                        tp_idx_list.append(j)

                    if trade_done:

                        # if entry[initial_i] == 1:
                        #   # ep = res_df['close'].iloc[initial_i]
                        #   # ep_list[0] = long_ep.iloc[initial_i]
                        #   pass
                        # else:
                        #   # ep = long_ep.iloc[i]
                        #   ep_list[0] = res_df['open'].iloc[initial_i]
                        #   # ep = res_df['close'].iloc[initial_i - 1]

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

                            #         tp 미체결 survey        #
                            nontp_liqd_list.append(long_liq)
                            nontp_long_liqd_list.append(long_liq)
                            nontp_long_indexs.append(i)
                            nontp_long_ep_list.append(ep_list[0])

                            nontp_long_pr = (tp / ep_list[0] - fee - 1) * lvrg + 1
                            nontp_pr_list.append(nontp_long_pr)
                            nontp_long_pr_list.append(nontp_long_pr)

            i += 1
            if i >= len(res_df):
                break

        # -------------------- result analysis -------------------- #
        try:
            plt.figure(figsize=(16, 12))
            # plt.suptitle(key)

            np_pr = np.array(pr_list)
            # np_pr = (np.array(pr_list) - 1) * lvrg + 1

            total_pr = np.cumprod(np_pr)
            wr = len(np_pr[np_pr > 1]) / len(np_pr[np_pr != 1])

            # plt.subplot(121)
            plt.subplot(231)
            plt.plot(total_pr)
            if len(nontp_liqd_list) != 0:
                plt.title(
                    "wr : %.2f\nmin_pr : %.2f\nacc_pr : %.2f\n lvrg %s\nliqd : %.2f\nnontp_liqd_cnt : %s\nnontp_liqd : %.2f\nontp_liqd_pr : %.2f"
                    % (
                    wr, np.min(np_pr), total_pr[-1], lvrg, min(liqd_list), len(nontp_liqd_list), min(nontp_liqd_list),
                    min(nontp_pr_list)))
            else:
                plt.title("wr : %.2f\nmin_pr : %.2f\nacc_pr : %.2f\n lvrg %s\nliqd : %.2f\nnontp_liqd_cnt : %s"
                          % (wr, np.min(np_pr), total_pr[-1], lvrg, min(liqd_list), len(nontp_liqd_list)))
            # plt.show()

            #         short only      #
            np_short_pr = np.array(short_list)

            total_short_pr = np.cumprod(np_short_pr)
            short_wr = len(np_short_pr[np_short_pr > 1]) / len(np_short_pr[np_short_pr != 1])

            plt.subplot(232)
            plt.plot(total_short_pr)
            if len(nontp_short_liqd_list) != 0:

                max_nontp_short_term = len(res_df) - nontp_short_indexs[0]

                plt.title(
                    "wr : %.2f\nmin_pr : %.2f\nacc_pr : %.2f\n lvrg %s\nliqd : %.2f\nnontp_short_liqd_cnt : %s\nnontp_short_liqd : %.2f\nontp_short_liqd_pr : %.2f\nmax_nontp_short_term : %s"
                    % (short_wr, np.min(np_short_pr), total_short_pr[-1], lvrg, min(short_liqd_list),
                       len(nontp_short_liqd_list), min(nontp_short_liqd_list), min(nontp_short_pr_list),
                       max_nontp_short_term))
            else:
                plt.title("wr : %.2f\nmin_pr : %.2f\nacc_pr : %.2f\n lvrg %s\nliqd : %.2f\nnontp_short_liqd_cnt : %s"
                          % (short_wr, np.min(np_short_pr), total_short_pr[-1], lvrg, min(short_liqd_list),
                             len(nontp_short_liqd_list)))

            #         long only      #
            np_long_pr = np.array(long_list)
            # np_long_pr = (np.array(long_list) - 1) * lvrg + 1

            total_long_pr = np.cumprod(np_long_pr)
            long_wr = len(np_long_pr[np_long_pr > 1]) / len(np_long_pr[np_long_pr != 1])

            plt.subplot(233)
            plt.plot(total_long_pr)
            if len(nontp_long_liqd_list) != 0:

                max_nontp_long_term = len(res_df) - nontp_long_indexs[0]

                plt.title(
                    "wr : %.2f\nmin_pr : %.2f\nacc_pr : %.2f\n lvrg %s\nliqd : %.2f\nnontp_long_liqd_cnt : %s\nnontp_long_liqd : %.2f\nontp_long_liqd_pr : %.2f\nmax_nontp_long_term : %s"
                    % (long_wr, np.min(np_long_pr), total_long_pr[-1], lvrg, min(long_liqd_list),
                       len(nontp_long_liqd_list), min(nontp_long_liqd_list), min(nontp_long_pr_list),
                       max_nontp_long_term))
            else:
                plt.title("wr : %.2f\nmin_pr : %.2f\nacc_pr : %.2f\n lvrg %s\nliqd : %.2f\nnontp_long_liqd_cnt : %s"
                          % (long_wr, np.min(np_long_pr), total_long_pr[-1], lvrg, min(long_liqd_list),
                             len(nontp_long_liqd_list)))

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
            # rev_np_short_pr = 1 / (np.array(short_list) + fee) - fee
            rev_np_short_pr = (1 / ((np.array(short_list) - 1) / lvrg + fee + 1) - fee - 1) * lvrg + 1
            # rev_np_short_pr = (1 / (np.array(short_list) + fee) - fee - 1) * lvrg + 1

            rev_total_short_pr = np.cumprod(rev_np_short_pr)
            rev_short_wr = len(rev_np_short_pr[rev_np_short_pr > 1]) / len(rev_np_short_pr[rev_np_short_pr != 1])

            # plt.subplot(122)
            plt.subplot(235)
            plt.plot(rev_total_short_pr)
            plt.title("wr : %.2f\nmin_pr : %.2f\nacc_pr : %.2f\n lvrg %s" % (
            rev_short_wr, np.min(rev_np_short_pr), rev_total_short_pr[-1], lvrg))

            #         long       #
            # rev_np_long_pr = 1 / (np.array(long_list) + fee) - fee
            rev_np_long_pr = (1 / ((np.array(long_list) - 1) / lvrg + fee + 1) - fee - 1) * lvrg + 1

            rev_total_long_pr = np.cumprod(rev_np_long_pr)
            rev_long_wr = len(rev_np_long_pr[rev_np_long_pr > 1]) / len(rev_np_long_pr[rev_np_long_pr != 1])

            # plt.subplot(122)
            plt.subplot(236)
            plt.plot(rev_total_long_pr)
            plt.title("wr : %.2f\nmin_pr : %.2f\nacc_pr : %.2f\n lvrg %s" % (
            rev_long_wr, np.min(rev_np_long_pr), rev_total_long_pr[-1], lvrg))

            plt.show()

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
                h_np_short_pr = np.array(h_short_list)

                h_total_short_pr = np.cumprod(h_np_short_pr)
                h_short_wr = len(h_np_short_pr[h_np_short_pr > 1]) / len(h_np_short_pr[h_np_short_pr != 1])

                plt.subplot(232)
                plt.plot(h_total_short_pr)
                plt.title("wr : %.2f\nmin_pr : %.2f\nacc_pr : %.2f\n lvrg %s" % (
                h_short_wr, np.min(h_np_short_pr), h_total_short_pr[-1], lvrg))

                #         long only      #
                h_np_long_pr = np.array(h_long_list)

                h_total_long_pr = np.cumprod(h_np_long_pr)
                h_long_wr = len(h_np_long_pr[h_np_long_pr > 1]) / len(h_np_long_pr[h_np_long_pr != 1])

                plt.subplot(233)
                plt.plot(h_total_long_pr)
                plt.title("wr : %.2f\nmin_pr : %.2f\nacc_pr : %.2f\n lvrg %s" % (
                h_long_wr, np.min(h_np_long_pr), h_total_long_pr[-1], lvrg))

                #     reversion adjustment      #

                h_rev_total_pr = np.cumprod(h_rev_np_pr)
                h_rev_wr = len(h_rev_np_pr[h_rev_np_pr > 1]) / len(h_rev_np_pr[h_rev_np_pr != 1])

                # plt.subplot(122)
                plt.subplot(234)
                plt.plot(h_rev_total_pr)
                plt.title("wr : %.2f\nmin_pr : %.2f\nacc_pr : %.2f\n lvrg %s" % (
                h_rev_wr, np.min(h_rev_np_pr), h_rev_total_pr[-1], lvrg))

                #         short       #
                # h_rev_np_short_pr = 1 / (np.array(h_short_list) + fee) - fee
                h_rev_np_short_pr = (1 / ((np.array(h_short_list) - 1) / lvrg + fee + 1) - fee - 1) * lvrg + 1

                h_rev_total_short_pr = np.cumprod(h_rev_np_short_pr)
                h_rev_short_wr = len(h_rev_np_short_pr[h_rev_np_short_pr > 1]) / len(
                    h_rev_np_short_pr[h_rev_np_short_pr != 1])

                # plt.subplot(122)
                plt.subplot(235)
                plt.plot(h_rev_total_short_pr)
                plt.title("wr : %.2f\nmin_pr : %.2f\nacc_pr : %.2f\n lvrg %s" % (
                h_rev_short_wr, np.min(h_rev_np_short_pr), h_rev_total_short_pr[-1], lvrg))

                #         long       #
                # h_rev_np_long_pr = 1 / (np.array(h_long_list) + fee) - fee
                h_rev_np_long_pr = (1 / ((np.array(h_long_list) - 1) / lvrg + fee + 1) - fee - 1) * lvrg + 1

                h_rev_total_long_pr = np.cumprod(h_rev_np_long_pr)
                h_rev_long_wr = len(h_rev_np_long_pr[h_rev_np_long_pr > 1]) / len(
                    h_rev_np_long_pr[h_rev_np_long_pr != 1])

                # plt.subplot(122)
                plt.subplot(236)
                plt.plot(h_rev_total_long_pr)
                plt.title("wr : %.2f\nmin_pr : %.2f\nacc_pr : %.2f\n lvrg %s" % (
                h_rev_long_wr, np.min(h_rev_np_long_pr), h_rev_total_long_pr[-1], lvrg))

                plt.show()

        except Exception as e:
            print('error in pr plot :', e)

            # plt.show()
        # plt.savefig("basic_v1/back_pr/back_pr.png")
        plt.savefig(dir_path + "/back_pr/back_pr.png")
        print("back_pr.png saved !")

        # break

        return ep_tp_list, trade_list
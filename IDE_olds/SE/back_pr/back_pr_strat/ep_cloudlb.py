import matplotlib.pyplot as plt
import numpy as np

fee = 0.0004
lvrg = 5

cloud_shift_size = 1

cloud_lookback = 50
cloud_lookback = 3

gap = 0.00005

senkoua_list = ['senkou_a1']
senkoub_list = ['senkou_b1']


def back_pr_check(res_df):

    for senkou_a, senkou_b in zip(senkoua_list, senkoub_list):

        cloud_top = np.max(res_df[[senkou_a, senkou_b]], axis=1)
        cloud_bottom = np.min(res_df[[senkou_a, senkou_b]], axis=1)

        upper_ep = res_df['min_upper'] * (1 - gap)
        lower_ep = res_df['max_lower'] * (1 + gap)

        under_top = upper_ep <= cloud_top.shift(cloud_shift_size)
        over_bottom = lower_ep >= cloud_bottom.shift(cloud_shift_size)
        # print("under_top :", under_top)
        # break

        #       short = -1      #
        entry = np.where((res_df['close'].shift(1) <= upper_ep) &
                         # (lower_ep <= res_df['high']) &
                         (res_df['high'] >= upper_ep) &  # <-- 이것만 있어도 되지 않을까함
                         np.logical_not((res_df['minor_ST1_Trend'] == 1) &
                                        (res_df['minor_ST2_Trend'] == 1) &
                                        (res_df['minor_ST3_Trend'] == 1))
                         , -1, 0)

        #       short = -2      #
        entry = np.where((res_df['close'].shift(1) > upper_ep) &
                         # (lower_ep <= res_df['high']) &
                         # (res_df['high'] >= upper_ep) &   # <-- 이것만 있어도 되지 않을까함
                         np.logical_not((res_df['minor_ST1_Trend'] == 1) &
                                        (res_df['minor_ST2_Trend'] == 1) &
                                        (res_df['minor_ST3_Trend'] == 1))
                         , -2, entry)

        #       long = 1     #
        #       close.shift(1) > ep | 2nd_middle line       #

        entry = np.where((res_df['close'].shift(1) >= lower_ep) &
                         # (lower_ep <= res_df['high']) &
                         (res_df['low'] <= lower_ep) &  # <-- 이것만 있어도 되지 않을까함
                         np.logical_not((res_df['minor_ST1_Trend'] == -1) &
                                        (res_df['minor_ST2_Trend'] == -1) &
                                        (res_df['minor_ST3_Trend'] == -1))
                         , 1, entry)

        #       long = 2     #
        #       close.shift(1) > ep | 2nd_middle line       #

        entry = np.where((res_df['close'].shift(1) < lower_ep) &
                         # (lower_ep <= res_df['high']) &
                         # (res_df['low'] <= lower_ep) &    # <-- 이것만 있어도 되지 않을까함
                         np.logical_not((res_df['minor_ST1_Trend'] == -1) &
                                        (res_df['minor_ST2_Trend'] == -1) &
                                        (res_df['minor_ST3_Trend'] == -1))
                         , 1, entry)

        #       1-2. tp line = middle line 조금 이내         #
        short_tp = res_df['middle_line'] * (1 + gap)
        long_tp = res_df['middle_line'] * (1 - gap)

        # short_tp = (res_df['middle_line'] + res_df['min_upper']) / 2
        # long_tp = (res_df['middle_line'] + res_df['max_lower']) / 2

        #       trading : 여기도 체결 결과에 대해 묘사함       #
        trade_list = []
        pr_list = []
        ep_tp_list = []
        tp_state_list = []

        i = 0
        while 1:
            # for i in range(len(res_df)):

            if i < cloud_lookback:
                i += 1
                if i >= len(res_df):
                    break
                continue

            # if entry[i] == -1:
            if entry[i] in [-1, -2]:

                #     check cloud constraints   #
                if np.sum(under_top.iloc[i + 1 - cloud_lookback:i + 1]) != cloud_lookback:
                    # if np.sum(under_top.iloc[i - cloud_lookback:i]) != cloud_lookback:
                    i += 1
                    if i >= len(res_df):
                        break
                    continue
                    # else:
                #   print("np.sum(under_top.iloc[i + 1 - cloud_lookback:i + 1]) :", under_top.iloc[i + 1 - cloud_lookback:i + 1])

                for j in range(i + 1, len(res_df)):

                    if res_df['low'].iloc[j] <= short_tp.iloc[j]:
                        # if res_df['low'].iloc[j] <= short_tp.iloc[j] <= res_df['high'].iloc[j]: --> 이건 잘못되었음

                        #         dynamic tp        #
                        if short_tp.iloc[j] != short_tp.iloc[j - 1]:

                            #   tp limit 이 불가한 경우 = open 이 이미, tp 를 넘은 경우 #
                            if res_df['open'].iloc[j] < short_tp.iloc[j]:

                                # tp = short_tp.iloc[j]
                                tp = res_df['open'].iloc[j]
                                tp_state_list.append("d-open")

                            #   tp limit 이 가능한 경우 = open 이 아직, tp 를 넘지 않은 경우 #
                            else:

                                tp = short_tp.iloc[j]
                                # tp = res_df['open'].iloc[j]
                                tp_state_list.append("d-short_tp")

                        #         static tp         #
                        else:

                            #   tp limit 이 불가한 경우 = open 이 이미, tp 를 넘은 경우 #
                            #   non_inversion 의 경우, short_tp 가 가능함   #

                            if res_df['open'].iloc[j] < short_tp.iloc[j]:

                                tp = short_tp.iloc[j]
                                # tp = res_df['open'].iloc[j]
                                tp_state_list.append("s-short_tp")

                            #   tp limit 이 가능한 경우 = open 이 아직, tp 를 넘지 않은 경우 #
                            else:

                                tp = short_tp.iloc[j]
                                # tp = res_df['open'].iloc[j]
                                tp_state_list.append("s-short_tp")

                        if entry[i] == -1:
                            ep = upper_ep.iloc[i]
                        else:
                            ep = res_df['open'].iloc[i]
                            # ep = res_df['close'].iloc[i - 1]

                        temp_pr = ep / tp - fee
                        ep_tp_list.append((ep, tp))
                        trade_list.append([i, j])
                        pr_list.append(temp_pr)
                        i = j
                        break

            # elif entry[i] == 1:
            elif entry[i] in [1, 2]:

                if np.sum(over_bottom.iloc[i + 1 - cloud_lookback:i + 1]) != cloud_lookback:
                    # if np.sum(over_bottom.iloc[i - cloud_lookback:i]) != cloud_lookback:
                    i += 1
                    if i >= len(res_df):
                        break
                    continue

                for j in range(i + 1, len(res_df)):

                    #     1. 결과적으로는 tp 를 넘었는데    #
                    if res_df['high'].iloc[j] >= long_tp.iloc[j]:

                        #         dynamic tp        #
                        if long_tp.iloc[j] != long_tp.iloc[j - 1]:

                            #   tp limit 이 불가한 경우 = open 이 이미, tp 를 넘은 경우 #
                            if res_df['open'].iloc[j] >= long_tp.iloc[j]:

                                # tp = long_tp.iloc[j]
                                tp = res_df['open'].iloc[j]
                                tp_state_list.append("d-open")


                            #   tp limit 이 가능한 경우 = open 이 아직, tp 를 넘지 않은 경우 #
                            else:

                                tp = long_tp.iloc[j]
                                # tp = res_df['open'].iloc[j]
                                tp_state_list.append("d-long_tp")

                        #         static tp         #
                        else:

                            #   tp limit 이 불가한 경우 = open 이 이미, tp 를 넘은 경우 #
                            #   non_inversion 의 경우, short_tp 가 가능함   #

                            if res_df['open'].iloc[j] >= long_tp.iloc[j]:

                                tp = long_tp.iloc[j]
                                # tp = res_df['open'].iloc[j]
                                tp_state_list.append("s-long_tp")

                            #   tp limit 이 가능한 경우 = open 이 아직, tp 를 넘지 않은 경우 #
                            else:

                                tp = long_tp.iloc[j]
                                # tp = res_df['open'].iloc[j]
                                tp_state_list.append("s-long_tp")

                        if entry[i] == 1:
                            ep = lower_ep.iloc[i]
                        else:
                            ep = res_df['open'].iloc[i]
                            # ep = res_df['close'].iloc[i - 1]

                        temp_pr = tp / ep - fee
                        ep_tp_list.append((ep, tp))
                        trade_list.append([i, j])
                        pr_list.append(temp_pr)
                        i = j
                        break

            i += 1
            if i >= len(res_df):
                break

        # -------------------- result analysis -------------------- #
        plt.figure(figsize=(12, 8))

        np_pr = (np.array(pr_list) - 1) * lvrg + 1

        total_pr = np.cumprod(np_pr)
        wr = len(np_pr[np_pr > 1]) / len(np_pr[np_pr != 1])

        plt.subplot(121)
        plt.plot(total_pr)
        plt.title("wr : %.2f\nmin_pr : %.2f\nacc_pr : %.2f\n lvrg %s" % (wr, np.min(np_pr), total_pr[-1], lvrg))
        # plt.show()

        #     reversion adjustment      #
        rev_np_pr = (1 / (np.array(pr_list) + fee) - fee - 1) * lvrg + 1

        rev_total_pr = np.cumprod(rev_np_pr)
        rev_wr = len(rev_np_pr[rev_np_pr > 1]) / len(rev_np_pr[rev_np_pr != 1])

        plt.subplot(122)
        plt.plot(rev_total_pr)
        plt.title(
            "wr : %.2f\nmin_pr : %.2f\nacc_pr : %.2f\n lvrg %s" % (rev_wr, np.min(rev_np_pr), rev_total_pr[-1], lvrg))

        # plt.show()
        plt.savefig("basic_v1/back_pr/back_pr.png")
        print("back_pr.png saved !")

        # break

        return ep_tp_list, trade_list
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import time

import keras
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# print(tf.__version__)

# tf_config = tf.ConfigProto()
# tf_config.gpu_options.allow_growth = True
# # tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5
# tf.keras.backend.set_session(tf.Session(config=tf_config))
#
# #           GPU Set         #
# tf.device('/device:XLA_GPU:0')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
import keras
from sklearn.metrics import confusion_matrix
import pickle


def get_back_result(ohlcv, predictions, err_ranges, tp=0.04, sl=None, leverage=1, show_detail=False, show_plot=False,
                    return_pr=False, cumsum=False,
                    close_ver=False, reverse_short=False):
    high, low, test = np.split(ohlcv.values[-len(predictions):, [1, 2, 3]], 3, axis=1)

    if close_ver:
        predictions = ohlcv['close'].shift(1).values[-len(test):]

    fee = 0.0006
    long_profits = []
    short_profits = []
    liquidations = []
    win_cnt = 0
    for i in range(len(test)):

        # long_ep = predictions[i]
        long_ep = (predictions[i] - err_ranges[i]) * (1 / (1 + tp))
        if sl is not None:
            long_sl = long_ep * (1 / (sl + 1))

        # assert long_ep < long_exit, 'long_exit < long_ep !, %s, %s' % (long_exit, long_ep)

        short_ep = (predictions[i] + err_ranges[i]) * (1 + tp)
        # short_ep = (predictions[i] + err_ranges[i]) * (1 / (1 - tp))
        if sl is not None:
            short_sl = short_ep * (1 / (1 - sl))

        # print((low[i]))

        #    long 우선   # <-- long & short 둘다 체결된 상황에서는 long 체결을 우선으로 한다.
        if low[i] < long_ep:

            liquidation = low[i] / long_ep - fee
            l_liquidation = 1 + (liquidation - 1) * leverage
            liquidations.append(l_liquidation)

            if max(l_liquidation, 0) == 0:
                l_profit = 0
                # print('low[i], long_ep, l_liquidation :', low[i], long_ep, l_liquidation)
            else:

                if sl is not None:
                    if low[i] < long_sl:
                        profit = long_sl / long_ep - fee
                    else:
                        profit = test[i] / long_ep - fee

                else:
                    profit = test[i] / long_ep - fee

                l_profit = 1 + (profit - 1) * leverage
                l_profit = max(l_profit, 0)

                if profit >= 1:
                    win_cnt += 1

            long_profits.append(l_profit)
            short_profits.append(1.0)

            if show_detail:
                print(test[i], predictions[i], long_ep)

        # if high[i] > short_ep > low[i]: # 지정 대기가 아니라, 해당 price 가 지나면, long 한다.

        #   if not reverse_short:
        #     liquidation = short_ep / high[i]  - fee
        #   else:
        #     liquidation = low[i] / short_ep  - fee
        #   l_liquidation = 1 + (liquidation - 1) * leverage

        #   if max(l_liquidation, 0) == 0:
        #     l_profit = 0
        #   else:

        #     if sl is not None:
        #       if high[i] > short_sl:

        #         if not reverse_short:
        #           profit = short_ep / short_sl - fee
        #         else:
        #           profit = short_sl / short_ep - fee

        #       else:
        #         if not reverse_short:
        #           profit = short_ep / test[i] - fee
        #         else:
        #           profit = test[i] / short_ep - fee

        #     else:

        #       if not reverse_short:
        #         profit = short_ep / test[i] - fee
        #       else:
        #         profit = test[i] / short_ep - fee

        #     l_profit = 1 + (profit - 1) * leverage
        #     l_profit = max(l_profit, 0)

        #     if profit >= 1:
        #       win_cnt += 1

        #   short_profits.append(l_profit)
        #   long_profits.append(1.0)

        #   if show_detail:
        #     print(test[i], predictions[i], short_ep)

        else:
            long_profits.append(1.0)
            short_profits.append(1.0)
            liquidations.append(1.0)

    long_win_ratio = sum(np.array(long_profits) > 1.0) / sum(np.array(long_profits) != 1.0)
    short_win_ratio = sum(np.array(short_profits) > 1.0) / sum(np.array(short_profits) != 1.0)
    long_frequency = sum(np.array(long_profits) != 1.0) / len(test)
    short_frequency = sum(np.array(short_profits) != 1.0) / len(test)
    if not cumsum:
        long_accum_profit = np.array(long_profits).cumprod()
        short_accum_profit = np.array(short_profits).cumprod()
    else:
        long_accum_profit = (np.array(long_profits) - 1.0).cumsum()
        short_accum_profit = (np.array(short_profits) - 1.0).cumsum()

    # print(win_ratio)

    if show_plot:
        plt.figure(figsize=(10, 5))
        plt.suptitle('tp=%.4f, lvrg=%d' % (tp, leverage))

        plt.subplot(151)
        plt.plot(liquidations)
        plt.title('liquidations')

        plt.subplot(152)
        plt.plot(long_profits)
        plt.title('Win Ratio : %.2f %%\nrequency : %.2f %%' % (long_win_ratio * 100, long_frequency * 100),
                  color='black')
        # plt.show()

        # print()
        plt.subplot(153)
        plt.plot(long_accum_profit)
        plt.title('Accum_profit : %.2f' % long_accum_profit[-1], color='black')

        plt.subplot(154)
        plt.plot(short_profits)
        plt.title('Win Ratio : %.2f %%\nrequency : %.2f %%' % (short_win_ratio * 100, short_frequency * 100),
                  color='black')
        # plt.show()

        # print()
        plt.subplot(155)
        plt.plot(short_accum_profit)
        plt.title('Accum_profit : %.2f' % short_accum_profit[-1], color='black')
        plt.show()

    return [long_win_ratio, short_win_ratio], [long_frequency, short_frequency], [long_accum_profit[-1],
                                                                                  short_accum_profit[-1]], [
               long_profits, short_profits]


with open('C:/Users/Lenovo/PycharmProjects/Project_System_Trading/Rapid_Ascend/arima_result/pr_list/arima_arima_close_updown_profit_ls_only_long_result_30m.pickle', 'rb') as f:
    load_dict = pickle.load(f)

model_abs_path = os.path.abspath("test_set/model/classifier_45_ai_pr_0211_014026.h5")
# model_abs_path = os.path.abspath("test_set/model/classifier_45_close_updown_pr_all_pair_shuffle_final_300k.h5")
model_path = r'%s' % model_abs_path
model = keras.models.load_model(model_path)

long_index = 0
leverage = 5
thresh = 0.8

candis = list(load_dict.keys())
prev_x = None

from binance_futures_arima_modules_ma7_close_entry import ep_stacking

# df = pd.read_excel('./candlestick_concated/30m/ETHUSDT_ai_plus_test.xlsx', index_col=0)
df = pd.read_excel('./candlestick_concated/30m/ETHUSDT_ai_plus.xlsx', index_col=0)
complete_df = df.iloc[:-1]

# complete_df['ep'] = np.nan
# ep_list = ep_stacking(complete_df, tp=0, test_size=1000, use_rows=3000)
# complete_df['ep'].iloc[-len(ep_list):] = ep_list

#           close mode          #
# complete_df['ep'] = complete_df['close'].shift(1)

# complete_df.to_excel('./candlestick_concated/30m/ETHUSDT_ai_plus_ma7.xlsx')
# print(complete_df.tail())
input_df = complete_df.iloc[np.sum(np.isnan(complete_df['ep'].values)):]
# print(input_df)
# quit()

ohlcv = input_df.iloc[:, :-1]
print('len(ohlcv) :', len(ohlcv))
# ohlcv = ohlcv.iloc[-int(len(ohlcv) * 0.34):]
predictions = input_df.iloc[:, [-1]].values
# err_ranges = load_dict[key]['err_ranges']

# predictions = ohlcv['close'].shift(1).values
err_ranges = np.zeros_like(predictions)

# leverage_list = profit_result_dict[key]['leverage_list']
# temp_ap_list = list()
# temp_pr_list = list()

# try:
    # print('-------------- %s --------------' % key)
result = get_back_result(ohlcv, predictions, err_ranges, tp=0.012, leverage=leverage, show_plot=True,
                         reverse_short=False, show_detail=False)
# temp_ap_list.append(result[2])
# temp_pr_list.append(result[3])
# quit()
# if round(leverage) == 1:
#   temp_pr_list = result[3]
pr_list = result[3][long_index]

# except Exception as e:
#     print(e)
    # break
    # break

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#         clustering zone           #

#       set data features : ohlc, v, ep
ohlc = ohlcv.iloc[-len(predictions):, :4]
vol = ohlcv.iloc[-len(predictions):, [4]]
long_ep = np.array(predictions)
# long_ep = ohlc.iloc[:, [-1]].values <-- 이건 이전 종가가 아니라 그냥 종가임
long_ep = long_ep.reshape(-1, 1)

# ohlcv['u_wick'] = ohlcv['high'] / np.maximum(ohlcv['close'], ohlcv['open'])
# ohlcv['d_wick'] = np.minimum(ohlcv['close'], ohlcv['open']) / ohlcv['low']
# ohlcv['body'] = ohlcv['close'] / ohlcv['open']
#
# candle = ohlcv.iloc[-len(predictions):, -3:]

print('len(ohlc) :', len(ohlc))
print('long_ep.shape :', long_ep.shape)
print('len(pr_list) :', len(pr_list))

#       set params    #
period = 45
data_x, data_pr, data_updown = [], [], []
# key_i = i

for i in range(period, len(predictions)):

    #   pr_list != 1 인 데이터만 사용한다
    # if 1:
    if pr_list[i] != 1:

        #   prediction 을 제외한 이전 데이터를 사용해야한다
        temp_ohlc = ohlc.iloc[i - period: i].values
        temp_long_ep = long_ep[i - period: i]
        temp_vol = vol.iloc[i - period: i].values
        # temp_candle = candle.iloc[i - period: i].values

        # print(temp_ohlc.shape)
        # print(temp_long_ep.shape)
        # print(temp_vol.shape)
        # print(temp_candle.shape)
        # break

        #   stacking
        # temp_data = np.hstack((temp_ohlc, temp_long_ep, temp_vol, temp_candle))
        temp_data = np.hstack((temp_ohlc, temp_long_ep, temp_vol))
        # temp_data = np.hstack((temp_ohlc, temp_vol))

        # temp_data = np.hstack((temp_ohlc, temp_long_ep))
        # temp_data = temp_vol

        #   scaler 설정

        #   ohlc & ep -> max_abs
        # max_abs = MaxAbsScaler()
        # temp_data[:, :5] = max_abs.fit_transform(temp_data[:, :5])

        min_max = MinMaxScaler()
        temp_data[:, :5] = min_max.fit_transform(temp_data[:, :5])

        #   vol -> min_max
        min_max = MinMaxScaler()
        temp_data[:, [5]] = min_max.fit_transform(temp_data[:, [5]])

        #   candle -> max_abs
        # max_abs = MaxAbsScaler()
        # temp_data[:, -3:] = max_abs.fit_transform(temp_data[:, -3:])

        # min_max = MinMaxScaler()
        # temp_data[:, -3:] = min_max.fit_transform(temp_data[:, -3:])

        if np.isnan(np.sum(temp_data)):
            continue

        data_x.append(temp_data)
        #     append result data    #
        data_pr.append(pr_list[i])
        data_updown.append(ohlc['close'].iloc[i] / ohlc['open'].iloc[i])

print('np.array(data_x).shape :', np.array(data_x).shape)
# print(data_x[0])

#       Reshape data for image deep - learning     #
_, row, col = np.array(data_x).shape

input_x = np.array(data_x).reshape(-1, row, col, 1).astype(np.float32)

#     1c to 3c    #
input_x = input_x * np.ones(3, dtype=np.float32)[None, None, None, :]

input_pr = np.array(data_pr).reshape(-1, 1).astype(np.float32)
input_ud = np.array(data_updown).reshape(-1, 1).astype(np.float32)
print('input_x.shape :', input_x.shape)
print('input_x.dtype :', input_x.dtype)
print('input_pr.shape :', input_pr.shape)
print('input_ud.shape :', input_ud.shape)

#         reshape data     #
temp_x = list()
for d_i, data in enumerate(input_x):
    # resized_data = cv2.resize(data, (row * 2, col * 2)) --> input image 홰손된다
    resized_data = data.repeat(2, axis=0).repeat(2, axis=1)
    # cmapped = plt.cm.Set1(resized_data)[:, :, :3]  # Drop Alpha Channel

    # if d_i == 0:
    #   plt.imshow(data)
    #   plt.show()
    #   plt.imshow(resized_data)
    #   plt.show()
    # print('resized_data.shape :', resized_data.shape)
    # break
    temp_x.append(resized_data)

re_input_x = np.array(temp_x)
y_test = np.where(input_pr > 1, 1, 0)
# y_test = np.where(input_ud > 1, 1, 0)

#             ai tester phase              #
test_result = model.predict(re_input_x)

y_score = test_result[:, [1]]
y_pred = np.where(y_score[:, -1] > thresh, 1, 0)

#         plot result       #
test_size = len(y_test)
test_pr_list = input_pr
print('origin ac_pr :', np.cumprod(test_pr_list)[-1])

cmat = confusion_matrix(y_test, y_pred)

org_wr = np.sum(cmat, axis=1)[-1] / sum(np.sum(cmat, axis=1))
ml_wr = cmat[1][1] / np.sum(cmat, axis=0)[-1]
print('win ratio improvement %.2f --> %.2f' % (org_wr, ml_wr))

# print('pr_test.shape :', pr_test.shape)

# print(y_pred)
# print(test_pr_list)
pred_pr_list = np.where(y_pred == 1, test_pr_list.reshape(-1, ), 1.0)
# print('pred_pr_list.shape :', pred_pr_list.shape)

if np.cumprod(test_pr_list)[-1] < np.cumprod(pred_pr_list)[-1]:
    print('accum_pr increased ! : %.3f --> %.3f' % (np.cumprod(test_pr_list)[-1], np.cumprod(pred_pr_list)[-1]))
    print('thresh :', thresh)

# if len(threshold) == 1:
plt.figure(figsize=(10, 5))

# plt.suptitle(key)
plt.subplot(121)
plt.plot(np.cumprod(test_pr_list))
plt.title('%.3f' % (np.cumprod(test_pr_list)[-1]))
# plt.show()

plt.subplot(122)
plt.plot(np.cumprod(pred_pr_list))
plt.title('%.3f' % (np.cumprod(pred_pr_list)[-1]))
plt.show()

#     save acc_pr result for comparing pairs --> find best pair   #
#     save improved win_ratio, acc_pr   #
# temp_dict = load_dict[key]
# temp_dict['improved_wr'] = ml_wr
# temp_dict['improved_ap_list'] = np.cumprod(pred_pr_list)

#         save dict       #
# with open('./arima_result/arima_ma7_profit_ls_only_long_result_%s.pickle' % interval, 'wb') as f:
#     pickle.dump(load_dict, f)

#     do stacking   #
# if prev_x is None:
#   prev_x = input_x
#   prev_pr = input_pr
#   prev_ud = input_ud
# else:
#   total_x = np.vstack((prev_x, input_x))
#   total_pr = np.vstack((prev_pr, input_pr))
#   total_ud = np.vstack((prev_ud, input_ud))

#   prev_x = total_x
#   prev_pr = total_pr
#   prev_ud = total_ud

#   print('total_x.shape :', total_x.shape)
#   print('total_pr.shape :', total_pr.shape)
#   print('total_ud.shape :', total_ud.shape)

from rnn.utils import *

long_index = 0
leverage = 5

ohlcv = load_dict[key]['ohlcv']
print('len(ohlcv) :', len(ohlcv))

#       select timestamp range      #
# time_index = ohlcv.index
# total_stamp = list(map(lambda x: datetime.timestamp(x), time_index))

# rm_index_amt = np.sum(np.array(total_stamp) < start_stamp)

# ohlcv = ohlcv.iloc[rm_index_amt:]
# print(ohlcv.head())

# ohlcv = ohlcv.iloc[:-int(len(ohlcv) * 0.3)]  # exclude back_range
# predictions = load_dict[key]['predictions']
# err_ranges = load_dict[key]['err_ranges']
print("ohlcv.index[0] :", ohlcv.index[0])
print("ohlcv.index[-1] :", ohlcv.index[-1])

predictions = ohlcv['close'].shift(1).values
err_ranges = np.zeros_like(predictions)

# leverage_list = profit_result_dict[key]['leverage_list']
# temp_ap_list = list()
# temp_pr_list = list()

try:
    print('-------------- %s --------------' % key)
    result = get_back_result(ohlcv, predictions, err_ranges, tp=0, leverage=leverage, show_plot=True,
                             reverse_short=False, show_detail=False)
    # temp_ap_list.append(result[2])
    # temp_pr_list.append(result[3])

    # if round(leverage) == 1:
    #   temp_pr_list = result[3]
    pr_list = result[3][long_index]

except Exception as e:
    print(e)
    break
# break


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#         clustering zone           #

#       set data features : ohlc, v, ep
time_index = ohlcv.index[-len(predictions):]
ohlc = ohlcv.iloc[-len(predictions):, :4]
vol = ohlcv.iloc[-len(predictions):, [4]]
long_ep = np.array(predictions)
long_ep = long_ep.reshape(-1, 1)

ohlcv['u_wick'] = ohlcv['high'] / np.maximum(ohlcv['close'], ohlcv['open'])
ohlcv['d_wick'] = np.minimum(ohlcv['close'], ohlcv['open']) / ohlcv['low']
ohlcv['body'] = ohlcv['close'] / ohlcv['open']

candle = ohlcv.iloc[-len(predictions):, -3:]

print('len(ohlc) :', len(ohlc))
print('long_ep.shape :', long_ep.shape)
print('len(pr_list) :', len(pr_list))

#       set params    #
period = 45
data_x, data_pr, data_updown = [], [], []
data_index = []
key_i = i

for i in range(period, len(predictions)):

    #   pr_list != 1 인 데이터만 사용한다
    # if 1:
    if pr_list[i] != 1:

        #   prediction 을 제외한 이전 데이터를 사용해야한다
        temp_ohlc = ohlc.iloc[i - period: i].values
        temp_long_ep = long_ep[i - period: i]
        temp_vol = vol.iloc[i - period: i].values
        temp_candle = candle.iloc[i - period: i].values

        # print(temp_ohlc.shape)
        # print(temp_long_ep.shape)
        # print(temp_vol.shape)
        # print(temp_candle.shape)
        # break

        #   stacking
        # temp_data = np.hstack((temp_ohlc, temp_long_ep, temp_vol, temp_candle))
        # temp_data = np.hstack((temp_ohlc, temp_long_ep, temp_vol))
        # temp_data = np.hstack((temp_ohlc, temp_vol))

        #     only close    #
        temp_data = temp_ohlc[:, [-1]]

        # temp_data = np.hstack((temp_ohlc, temp_long_ep))
        # temp_data = temp_vol

        #   scaler 설정

        #   ohlc & ep -> max_abs
        # max_abs = MaxAbsScaler()
        # temp_data[:, :-1] = max_abs.fit_transform(temp_data[:, :-1])

        # min_max = MinMaxScaler()
        # temp_data[:, :-1] = min_max.fit_transform(temp_data[:, :-1])

        #   vol -> min_max
        min_max = MinMaxScaler()
        temp_data[:, [-1]] = min_max.fit_transform(temp_data[:, [-1]])

        #   candle -> max_abs
        # max_abs = MaxAbsScaler()
        # temp_data[:, -3:] = max_abs.fit_transform(temp_data[:, -3:])

        # min_max = MinMaxScaler()
        # temp_data[:, -3:] = min_max.fit_transform(temp_data[:, -3:])

        if np.isnan(np.sum(temp_data)):
            continue

        data_x.append(temp_data)
        data_pr.append(pr_list[i])
        data_index.append(time_index[i])
        data_updown.append(ohlc['close'].iloc[i] / ohlc['open'].iloc[i])

print('np.array(data_x).shape :', np.array(data_x).shape)
# print(data_x[0])


#       Reshape data for image deep - learning     #
_, row, col = np.array(data_x).shape

# input_x = np.array(data_x).reshape(-1, row, col, 1).astype(np.float32)
input_x = np.array(data_x).reshape(-1, row, col).astype(np.float32)

#     1c to 3c    #
# input_x = input_x * np.ones(3, dtype=np.float32)[None, None, None, :]
# input_x = np.array(resize_npy(input_x))


input_pr = np.array(data_pr).reshape(-1, 1).astype(np.float32)
input_ud = np.array(data_updown).reshape(-1, 1).astype(np.float32)
input_index = np.array(data_index).reshape(-1, 1)
print('input_x.shape :', input_x.shape)
print('input_x.dtype :', input_x.dtype)
print('input_pr.shape :', input_pr.shape)
print('input_ud.shape :', input_ud.shape)
print('input_index.shape :', input_index.shape)


#       we only train last row      #
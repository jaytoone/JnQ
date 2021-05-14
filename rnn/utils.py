from statsmodels.tsa.arima_model import ARIMA
import numpy as np
import pandas as pd
# from binance_futures_concat_candlestick import concat_candlestick
import matplotlib.pyplot as plt
from plotly import subplots
# import plotly.offline as offline
import plotly.graph_objs as go
# import pickle

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 2500)
pd.set_option('display.max_columns', 2500)


def get_back_result(ohlcv, predictions, err_ranges, tp=0.04, sl=None, leverage=1, show_detail=False, show_plot=False, return_pr=False, cumsum=False,
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

        long_ep = predictions[i]
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
        plt.title('Win Ratio : %.2f %%\nrequency : %.2f %%' % (long_win_ratio * 100, long_frequency * 100), color='black')
        # plt.show()

        # print()
        plt.subplot(153)
        plt.plot(long_accum_profit)
        plt.title('Accum_profit : %.2f' % long_accum_profit[-1], color='black')

        plt.subplot(154)
        plt.plot(short_profits)
        plt.title('Win Ratio : %.2f %%\nrequency : %.2f %%' % (short_win_ratio * 100, short_frequency * 100), color='black')
        # plt.show()

        # print()
        plt.subplot(155)
        plt.plot(short_accum_profit)
        plt.title('Accum_profit : %.2f' % short_accum_profit[-1], color='black')
        plt.show()

    return [long_win_ratio, short_win_ratio], [long_frequency, short_frequency], [long_accum_profit[-1], short_accum_profit[-1]], \
               [long_profits, short_profits]


def ep_stacking(df, order=(0, 2, 1), tp=0.04, test_size=None, use_rows=None):
    # print(high)

    X = df['close']

    if test_size is None:
        test_size = int(len(X) * 0.66)

    train, test = X[:-test_size].values, X[-test_size:]
    # test_shift = test.shift(1).values
    test = test.values

    # time_index = df.index[-len(test):]

    # print('test_size :', len(test))
    # break
    # high, low = np.split(df.values[-len(test):, [1, 2]], 2, axis=1)
    # print(np.hstack((high[:20], low[:20])))
    # quit()

    history = list(train)
    # print(train)
    # print(history)
    # break
    # pred = model_fit.predict()
    # print(pred)
    # print(model_fit.forecast())
    # print(len(close), len(pred))
    # test_pred = pred[-len(test):]
    predictions = []
    err_ranges = []
    ep_list = []
    for t in range(len(test)):

        if use_rows is not None:
            assert len(history) >= use_rows, "len(history) < use_rows !"
            history = history[-use_rows:]

        # print(len(history))
        model = ARIMA(history, order=order)
        model_fit = model.fit(trend='c', disp=0)
        output = model_fit.forecast()
        # print(output)
        # break

        predictions.append(output[0])
        err_ranges.append(output[1])
        obs = test[t]
        # print('obs :', obs)
        history.append(obs)

        # long_ep = (output[0] - output[1]) * (1 / (tp + 1))
        long_ep = output[0]
        ep_list.append(long_ep)

        # break
        print('\rep stacking %.2f%%' % (t / len(test) * 100), end='')

    print('\nlen(ep_list) :', len(ep_list))
    return ep_list


def arima_profit(df, order=(0, 2, 1), tp=0.04, leverage=1, tp2=None):
    close = df['close']
    model = ARIMA(close, order=order)
    model_fit = model.fit(trend='c', disp=0)
    output = model_fit.forecast()
    pred_close, err_range = output[:2]

    output_df = df.copy()
    output_df['trade_state'] = np.nan
    output_df['long_ep'] = np.nan
    output_df['long_tp_level'] = np.nan
    output_df['short_ep'] = np.nan
    output_df['short_tp_level'] = np.nan
    output_df['sl_level'] = np.nan
    output_df['tp_level'] = np.nan
    output_df['leverage'] = np.nan

    # long_ep = (pred_close - err_range) * (1 / (tp + 1))
    long_ep = pred_close
    short_ep = long_ep

    if tp2 is not None:
        long_tp_level = pred_close * (1 / (1 - tp2))
        short_tp_level = pred_close * (1 / (tp2 + 1))
        output_df['long_tp_level'].iloc[-1] = long_tp_level
        output_df['short_tp_level'].iloc[-1] = short_tp_level

    output_df['long_ep'].iloc[-1] = long_ep
    output_df['short_ep'].iloc[-1] = short_ep
    output_df['leverage'].iloc[-1] = leverage

    return output_df, pred_close, err_range


def arima_summary(df, order=(0, 2, 1)):
    close = df['close']
    model = ARIMA(close, order=order)
    model_fit = model.fit(trend='c')
    print(model_fit.summary())
    plt.show(model_fit.plot_predict())

    return


def arima_test(df, order=(0, 2, 1), tp=0.04, fee=0.0006, leverage=1, test_size=None, use_rows=None, show_detail=False):
    # print(high)

    X = df['close']

    if test_size is None:
        test_size = int(len(X) * 0.66)

    train, test = X[:-test_size].values, X[-test_size:]
    test_shift = test.shift(1).values
    test = test.values

    time_index = df.index[-len(test):]

    print('test_size :', len(test))
    # break
    high, low = np.split(df.values[-len(test):, [1, 2]], 2, axis=1)
    # print(np.hstack((high[:20], low[:20])))
    # quit()

    history = list(train)
    # print(train)
    # print(history)
    # break
    # pred = model_fit.predict()
    # print(pred)
    # print(model_fit.forecast())
    # print(len(close), len(pred))
    # test_pred = pred[-len(test):]
    predictions = list()
    err_ranges = list()
    for t in range(len(test)):

        if use_rows is not None:
            history = history[-use_rows:]

        # print(len(history))
        model = ARIMA(history, order=order)
        model_fit = model.fit(trend='c', disp=0)
        output = model_fit.forecast()
        # print(output)
        # break
        yhat = output[0]
        predictions.append(yhat)
        err_ranges.append(output[1])
        obs = test[t]
        # print('obs :', obs)
        history.append(obs)
        # break
        print('\r %.2f%%' % (t / len(test) * 100), end='')

    # print('predictions :', predictions)
    # quit()
    # print('err_ranges :', err_ranges)
    # predictions = test_shift

    profits = list()
    win_cnt = 0
    for i in range(len(test)):

        long_ep = (predictions[i] - err_ranges[i]) * (1 / (tp + 1))
        short_ep = (predictions[i] + err_ranges[i]) * (1 / (1 - tp))

        # print((low[i]))
        if low[i] <= long_ep:
            profit = test[i] / long_ep - fee
            profits.append(1 + (profit - 1) * leverage)
            if show_detail:
                print(time_index[i], test[i], predictions[i], long_ep)

            if profit >= 1:
                win_cnt += 1

        # elif high[i] >= short_ep:
        #     profit = short_ep / test[i] - fee
        #     profits.append(1 + (profit - 1) * leverage)
        #     if show_detail:
        #         print(time_index[i], test[i], predictions[i], short_ep)
        #
        #     if profit >= 1:
        #         win_cnt += 1

    win_ratio = win_cnt / len(profits)
    # print(win_ratio)

    plt.plot(profits)
    plt.title('Win Ratio : %.3f %% Frequency : %.3f %%' % (win_ratio * 100, len(profits) / len(test) * 100))
    plt.show()

    # print()
    accum_profit = np.array(profits).cumprod()
    plt.plot(accum_profit)
    plt.title('Accum_profit : %.3f' % accum_profit[-1])
    plt.show()

    # model_fit.plot_diagnostics()
    # print(model_fit.plot_predict(start=900, end=950))


def calc_train_days(interval, use_rows):
    if interval == '15m':
        data_amt = 96
    elif interval == '30m':
        data_amt = 48
    elif interval == '1h':
        data_amt = 24
    elif interval == '4h':
        data_amt = 6
    else:
        print('interval out of range')
        quit()

    days = int(use_rows / data_amt) + 1

    return days


def arima_plotting(key, ohlcv, predictions, err_ranges, pr_list):

    ohlc = ohlcv.iloc[-len(pr_list):, :-1]

    pr_list = np.array(pr_list, dtype=np.float64)
    predictions = np.array(predictions, dtype=np.float64)

    #       temp     #
    err_ranges = np.array(err_ranges, dtype=np.float64)
    ep = (predictions - err_ranges) * (1 / (0.021 + 1))

    ohlc['profit'] = pr_list
    ohlc['fluctuation'] = ohlc['high'] / ohlc['low']
    # print(pr_list[:5])
    # print(predictions[:5])
    # quit()
    ohlc['ep'] = ep
    ohlc['predictions'] = predictions

    ohlc_ply = go.Candlestick(x=ohlc.index, open=ohlc['open'], close=ohlc['close'], high=ohlc['high'], low=ohlc['low'], name='ohlc')
    pred_ply = go.Scatter(x=ohlc.index, y=ohlc['predictions'], name='predictions', line={'color': 'orange'})
    ep_ply = go.Scatter(x=ohlc.index, y=ohlc['ep'], name='predictions', line={'color': 'blue'})

    fluc_ply = go.Scatter(x=ohlc.index, y=ohlc['fluctuation'], name='fluctuation', line={'color': 'lime'})
    profit_ply = go.Scatter(x=ohlc.index, y=ohlc['profit'].cumprod(), name='profit', fill='tozeroy')

    row, col = 3, 1
    fig = subplots.make_subplots(row, col, shared_xaxes=True, row_width=[0.4, .3, .3])

    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, dragmode='pan')

    row, col = 1, 1
    fig.append_trace(ohlc_ply, row, col)
    fig.append_trace(pred_ply, row, col)
    fig.append_trace(ep_ply, row, col)
    fig.update_xaxes(showgrid=False)

    for trade_num in range(len(pr_list) - 1):

        profit = pr_list[trade_num]
        if profit != 1.:
            if profit > 1.:
                fillcolor = 'cyan'
            else:
                fillcolor = 'magenta'

            fig.add_vrect(x0=ohlc.index[trade_num], x1=ohlc.index[trade_num + 1], annotation_text='%.3f' % profit,
                          annotation_position='top left',
                          fillcolor=fillcolor, opacity=.5, line_width=0, row=row, col=col)
    row, col = 2, 1
    fig.append_trace(fluc_ply, row, col)

    row, col = 3, 1
    fig.append_trace(profit_ply, row, col)

    # if save_path is not None:
    #     fig.write_html('%s/%s.html' % (save_path, date), auto_open=True)
    # # else:
    fig.write_html('arima_result/temp/%s.html' % key, auto_open=True)


if __name__ == '__main__':
    days = 1
    symbol = 'ETHUSDT'
    interval = '30m'
    date = '2021-02-11'

    # print(calc_train_days('1h', 3000))
    df = pd.read_excel('./candlestick_concated/%s/%s %s.xlsx' % (interval, date, symbol), index_col=0)
    # print(df.head())
    print(len(df))

    arima_test(df, tp=0.012, leverage=3, test_size=1000, use_rows=2999, show_detail=True)
    _, a, b = arima_profit(df.iloc[:-2, :])     # [1630.07677303] [13.21675231]
    print(a, b)  # 1.254 --> 1.263
    # print(pickle.format_version)


    # with open('arima_result/pr_list/arima_candi_profit_ls_result_%s.pickle' % interval, 'rb') as f:
    #     load_pr_list_dict = pickle.load(f)
    #
    # print(list(load_pr_list_dict.keys()))
    # keys = list(load_pr_list_dict.keys())
    # keys = ['2021-02-11 BTCUSDT.xlsx']
    #
    # plot_size = 1000
    # for key in keys:
    #     try:
    #         load_pr_list_dict[key]['pr_list']
    #     except Exception as e:
    #         continue
    #     else:
    #         ohlcv = load_pr_list_dict[key]['ohlcv'][-plot_size:]
    #         predictions = load_pr_list_dict[key]['predictions'][-plot_size:]
    #         err_ranges = load_pr_list_dict[key]['err_ranges'][-plot_size:]
    #         pr_list = load_pr_list_dict[key]['pr_list'][-plot_size:]
    #         arima_plotting(key, ohlcv, predictions, err_ranges, pr_list)
    #
    #     break
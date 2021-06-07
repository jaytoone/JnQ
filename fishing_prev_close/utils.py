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


def tp_update(ohlcv, wr_threshold=0.65, lvrg=9, tp_list=np.arange(0.005, 0.05, 0.001), fee=1e-4, plotting=False, save_path=None):

    # print('len(ohlcv) :', len(ohlcv))

    #     find best lvrg with lq    #
    # for lvrg in lvrg_list:

    #       tp opt      #
    best_tp = None
    best_ap = 0
    best_pr = None

    #        find best tp       #
    for tp in tp_list:

        long_ep = ohlcv['close'].shift(1) * (1 / (tp + 1))

        #       long      #
        pr = (ohlcv['close'] / long_ep - fee - 1) * lvrg + 1
        #     set condition   #
        pr = np.where(ohlcv['low'] < long_ep, pr, 1)

        #     Todo set constraints     #
        #           ema           #
        ema = ohlcv['close'].ewm(span=190, min_periods=190 - 1, adjust=False).mean()
        # pr = np.where(ohlcv['close'].shift(1) > ema.shift(1), pr, 1)
        pr = np.where(ema.shift(1) > ema.shift(2), pr, 1)

        #         ma        #
        # ma = ohlcv['close'].rolling(120).mean()
        # pr = np.where(ma.shift(1) > ma.shift(2), pr, 1)

        #     Todo set liqudation     #
        lq = (ohlcv['low'] / long_ep - fee - 1) * lvrg + 1

        # plt.plot(np.cumprod(pr))
        # plt.show()
        # break

        #       fill na with 1.0      #
        pr = np.where(np.isnan(pr), 1, pr)
        # avoid_pr = np.where(np.isnan(avoid_pr), 1, avoid_pr)
        lq = np.where(np.isnan(lq), 1, lq)
        # s_pr = np.where(np.isnan(s_pr), 1, pr)

        #       set lq      #
        pr = np.where((pr != 1) & (lq <= 0), 0, pr)
        # avoid_pr = np.where((avoid_pr != 1) & (lq <= 0), 0, avoid_pr)

        # plt.plot(np.cumprod(pr))
        # plt.title("%.3f" % tp)
        # plt.show()
        # plt.close()

        wr = len(pr[pr > 1]) / len(pr[pr != 1])

        ap = np.cumprod(pr)[-1]
        # if ap >= best_ap:
        if ap >= best_ap and wr > wr_threshold:
            # best_ep = long_ep
            best_ap = ap
            best_tp = tp
            best_pr = pr

    best_pr = np.array(best_pr)
    best_wr = len(best_pr[best_pr > 1]) / len(best_pr[best_pr != 1])

    if plotting:
        plt.figure(figsize=(6, 4))
        # plt.subplot(121)
        plt.plot(np.cumprod(best_pr))
        plt.title("best_wr : %.3f\nacc_pr : %.3f\ntp : %.3f\nlvrg : %s" %
                  (best_wr, np.cumprod(best_pr)[-1], best_tp, lvrg))

        plt.show()

    if save_path is not None:
        plt.figure(figsize=(5, 7))
        # plt.subplot(121)
        plt.plot(np.cumprod(best_pr))
        plt.title("best_wr : %.3f\nacc_pr : %.3f\ntp : %.3f\nlvrg : %s" %
                  (best_wr, np.cumprod(best_pr)[-1], best_tp, lvrg))

        plt.savefig(save_path)

    return best_tp


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


def arima_profit(df, order=(0, 2, 1), tp=0.04, leverage=1):

    # close = df['close']
    prev_close = df['close']

    # model = ARIMA(close, order=order)
    # model_fit = model.fit(trend='c', disp=0)
    # output = model_fit.forecast()

    pred_close, err_range = prev_close.values[-1], 0

    output_df = df.copy()
    output_df['trade_state'] = np.nan
    output_df['long_ep'] = np.nan
    output_df['long_tp_level'] = np.nan
    output_df['short_ep'] = np.nan
    output_df['short_tp_level'] = np.nan
    output_df['sl_level'] = np.nan
    output_df['tp_level'] = np.nan
    output_df['leverage'] = np.nan

    long_ep = pred_close * (1 / (tp + 1))
    # long_ep = pred_close
    short_ep = long_ep

    # if tp2 is not None:
    #     long_tp_level = pred_close * (1 / (1 - tp2))
    #     short_tp_level = pred_close * (1 / (tp2 + 1))
    #     output_df['long_tp_level'].iloc[-1] = long_tp_level
    #     output_df['short_tp_level'].iloc[-1] = short_tp_level

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
    symbol = 'DOTUSDT'
    interval = '30m'
    date = '2021-05-17'

    # print(calc_train_days('1h', 3000))
    df = pd.read_excel('../candlestick_concated/%s/%s %s.xlsx' % (interval, date, symbol), index_col=0)
    # print(df.head())
    print(len(df))

    import time
    s_time = time.time()
    #       tp_update       #
    print(tp_update(df, plotting=False, save_path="test.png"))
    print("elapsed time :", time.time() - s_time)

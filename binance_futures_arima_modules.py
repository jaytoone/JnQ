from statsmodels.tsa.arima_model import ARIMA
import numpy as np
import pandas as pd
from Binance_Futures_concat_candlestick import concat_candlestick
import matplotlib.pyplot as plt


def arima_profit(df, order=(0, 2, 1), tp=0.04, leverage=1):

    close = df['close']
    model = ARIMA(close, order=order)
    model_fit = model.fit(trend='c', disp=0)
    output = model_fit.forecast()
    pred_close, err_range = output[:2]

    df['trade_state'] = np.nan
    df['long_ep'] = np.nan
    df['short_ep'] = np.nan
    df['sl_level'] = np.nan
    df['tp_level'] = np.nan
    df['leverage'] = np.nan

    long_ep = (pred_close - err_range) * (1 / (tp + 1))
    short_ep = (pred_close + err_range) * (1 / (1 - tp))

    df['long_ep'].iloc[-1] = long_ep
    df['short_ep'].iloc[-1] = short_ep
    df['leverage'].iloc[-1] = leverage

    return df, pred_close, err_range


def arima_summary(df, order=(0, 2, 1)):

    close = df['close']
    model = ARIMA(close, order=order)
    model_fit = model.fit(trend='c')
    print(model_fit.summary())
    plt.show(model_fit.plot_predict())

    return


def arima_test(df, order=(0, 2, 1), tp=0.04, fee=0.0006, leverage=1):

    # print(high)

    X = df['close']
    size = int(len(X) * 0.66)

    train, test = X[:size].values, X[size:len(X)]
    test_shift = test.shift(1).values
    test = test.values

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

    show_detail = False
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
                print(test[i], predictions[i], long_ep)

            if profit >= 1:
                win_cnt += 1

        elif high[i] >= short_ep:
            profit = short_ep / test[i] - fee
            profits.append(1 + (profit - 1) * leverage)
            if show_detail:
                print(test[i], predictions[i], short_ep)

            if profit >= 1:
                win_cnt += 1

    win_ratio = win_cnt / len(profits)
    # print(win_ratio)

    plt.plot(profits)
    plt.title('Win Ratio : %.2f %%' % (win_ratio * 100))
    plt.show()

    # print()
    accum_profit = np.array(profits).cumprod()
    plt.plot(accum_profit)
    plt.title('Accum_profit : %.2f' % accum_profit[-1])
    plt.show()

    # model_fit.plot_diagnostics()
    # print(model_fit.plot_predict(start=900, end=950))


if __name__=='__main__':

    days = 1
    end_date = '2021-02-07'
    # end_date = None
    symbol = 'ETH'
    interval = '4h'

    # df, _ = concat_candlestick(symbol + 'USDT', interval=interval, days=days, end_date=end_date, show_process=True)
    df = pd.read_excel('./candlestick_concated/%s/%s %s.xlsx' % (interval, end_date, symbol), index_col=0)
    # print(df.head())

    # arima_test(df)
    _, a, b = arima_profit(df.iloc[:-1, :])     # [1630.07677303] [13.21675231]
    print(a, b)
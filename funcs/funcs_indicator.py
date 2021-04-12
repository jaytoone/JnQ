import numpy as np
import pandas as pd


def nz(x, y=0):
    # print(x)
    if np.isnan(x):
        return y
    else:
        return x


def round(x):
    x = float(x)
    if x > 0.99:
        return 0.999
    else:
        if x < -0.99:
            return -0.999
        else:
            return x


def ema(data, period, adjust=True):
    return pd.Series(data.ewm(span=period, adjust=adjust).mean())


def heikinashi(df):
    ha_df = df.copy()
    ha_df['close'] = (df['open'] + df['close'] + df['high'] + df['low']) / 4
    ha_df['open'] = np.nan
    for i in range(1, len(df)):
        if pd.isna(ha_df['open'].iloc[i - 1]):
            ha_df['open'].iloc[i] = (df['open'].iloc[i] + df['close'].iloc[i]) / 2
        else:
            ha_df['open'].iloc[i] = (ha_df['open'].iloc[i - 1] + ha_df['close'].iloc[i - 1]) / 2
    ha_df['high'] = np.max(ha_df.iloc[:, [0, 1, 3]].values, axis=1)
    ha_df['low'] = np.min(ha_df.iloc[:, [0, 2, 3]].values, axis=1)

    return ha_df


def trix_hist(df, period, multiplier, signal_period):
    triple = ema(ema(ema(df['close'], period), period), period)

    roc = 100 * (triple.diff() / triple)
    trix = multiplier * roc
    signal = trix.rolling(signal_period).mean()

    hist = trix - signal

    return hist


def iszero(val, eps):
    return abs(val) <= eps


def tozero(fst, snd):
    eps = 1e-10
    result = fst + snd
    if iszero(result, eps):
        result = 0
    else:
        if not iszero(result, 1e-4):
            pass
        else:
            result = 1e-15

    return result


def stdev(df, period):
    avg = df['close'].rolling(period).mean()
    std = pd.Series(index=df.index)
    for i in range(len(df)):
        std.iloc[i] = 0.0
        for backing_i in range(period):
            sum = tozero(df['close'].iloc[i - backing_i], -avg.iloc[i])
            std.iloc[i] += sum ** 2
        std.iloc[i] = np.sqrt(std.iloc[i] / period)

    return std


def cct_bbo(df, period, smooth):
    avg_ = df['close'].rolling(period).mean()
    stdev_ = stdev(df, period)
    cctbbo = 100 * (df['close'] + stdev_.apply(lambda x: 2 * x) - avg_) / (stdev_.apply(lambda x: 4 * x))
    ema_cctbbo = ema(cctbbo, smooth)

    return cctbbo, ema_cctbbo


def bb_width(df, period, multiple):
    basis = df['close'].rolling(period).mean()
    # print(stdev(df, period))
    # quit()
    dev = multiple * stdev(df, period)
    upper = basis + dev
    lower = basis - dev
    bbw = 2 * dev / basis

    return upper, lower, bbw


def fisher(df, period):
    hl2 = (df['high'] + df['low']) / 2
    high_ = hl2.rolling(period).max()
    low_ = hl2.rolling(period).min()

    # print(type(hl2))
    # print(high_)
    value = pd.Series(index=hl2.index)
    fish = pd.Series(index=hl2.index)
    value.iloc[0] = 0.0
    fish.iloc[0] = 0.0

    # print(value)

    for i in range(1, len(df)):
        value.iloc[i] = (0.66 * ((hl2.iloc[i] - low_.iloc[i]) / max(high_.iloc[i] - low_.iloc[i], .001) - .5)
                         + .67 * nz(value.iloc[i - 1]))
        # print(value.iloc[i])
        value.iloc[i] = round(value.iloc[i])
        # print(value.iloc[i])
        # print()
        fish.iloc[i] = .5 * np.log((1 + value.iloc[i]) / max(1 - value.iloc[i], .001)) + .5 * nz(fish.iloc[i - 1])

    fish.iloc[0] = np.nan

    return fish


def fisher_trend(df, column, tc_upper, tc_lower):
    fisher_cross_series = np.where((df[column].shift(1) < tc_upper) & (tc_upper < df[column]), 'CO', np.nan)
    fisher_cross_series = np.where((df[column].shift(1) > tc_lower) & (tc_lower > df[column]), 'CU',
                                   fisher_cross_series)
    fisher_trend_df = pd.DataFrame([np.nan] * len(df))

    start_i = -1
    while True:
        start_i += 1
        if start_i >= len(df):
            break
        # print(start_i)
        if not pd.isna(fisher_cross_series[start_i]):
            if fisher_cross_series[start_i] == 'CO':
                for j in range(start_i + 1, len(df)):
                    if fisher_cross_series[j] == 'CU':
                        fisher_trend_df.iloc[start_i: j] = 'Long'
                        start_i = j - 1
                        break
                    elif j == len(df) - 1:
                        fisher_trend_df.iloc[start_i:] = 'Long'
                        start_i = j - 1
            elif fisher_cross_series[start_i] == 'CU':
                for j in range(start_i + 1, len(df)):
                    if fisher_cross_series[j] == 'CO':
                        fisher_trend_df.iloc[start_i: j] = 'Short'
                        start_i = j - 1
                        break
                    elif j == len(df) - 1:
                        fisher_trend_df.iloc[start_i:] = 'Short'
                        start_i = j - 1

    return fisher_trend_df.values


def lucid_sar(df, af_initial=0.02, af_increment=0.02, af_maximum=0.2):

    uptrend = pd.Series(True, index=df.index)
    new_trend = pd.Series(False, index=df.index)
    reversal_state = pd.Series(0, index=df.index)
    af = pd.Series(af_initial, index=df.index)

    ep = df['high'].copy()
    sar = df['low'].copy()

    for i in range(1, len(df)):
        # if not pd.isna(uptrend.iloc[i - 1]) and pd.isna(new_trend.iloc[i - 1]):
        if reversal_state.iloc[i] == 0:
            if uptrend.iloc[i - 1]:
                ep.iloc[i] = max(df['high'].iloc[i], ep.iloc[i - 1])
            else:
                ep.iloc[i] = min(df['low'].iloc[i], ep.iloc[i - 1])
            if new_trend.iloc[i - 1]:
                af.iloc[i] = af_initial
            else:
                if ep.iloc[i] != ep.iloc[i - 1]:
                    af.iloc[i] = min(af_maximum, af.iloc[i - 1] + af_increment)
                else:
                    af.iloc[i] = af.iloc[i - 1]
            sar.iloc[i] = sar.iloc[i - 1] + af.iloc[i] * (ep.iloc[i] - sar.iloc[i - 1])

            if uptrend.iloc[i - 1]:
                sar.iloc[i] = min(sar.iloc[i], df['low'].iloc[i - 1])
                # if not pd.isna(df['low'].iloc[i - 2]):
                if i >= 2:
                    sar.iloc[i] = min(sar.iloc[i], df['low'].iloc[i - 2])
                if sar.iloc[i] > df['low'].iloc[i]:
                    uptrend.iloc[i] = False
                    new_trend.iloc[i] = True
                    sar.iloc[i] = max(df['high'].iloc[i], ep.iloc[i - 1])
                    ep.iloc[i] = min(df['low'].iloc[i], df['low'].iloc[i - 1])
                    reversal_state.iloc[i] = 2
                else:
                    uptrend.iloc[i] = True
                    new_trend.iloc[i] = False

            else:
                sar.iloc[i] = max(sar.iloc[i], df['high'].iloc[i - 1])
                # if not pd.isna(df['high'].iloc[i - 2]):
                if i >= 2:
                    sar.iloc[i] = max(sar.iloc[i], df['high'].iloc[i - 2])
                if sar.iloc[i] < df['high'].iloc[i]:
                    uptrend.iloc[i] = True
                    new_trend.iloc[i] = True
                    sar.iloc[i] = min(df['low'].iloc[i],  ep.iloc[i - 1])
                    ep.iloc[i] = max(df['high'].iloc[i], df['high'].iloc[i - 1])
                    reversal_state.iloc[i] = 1
                else:
                    uptrend.iloc[i] = False
                    new_trend.iloc[i] = False

        else:
            if reversal_state.iloc[i] == 1:
                ep.iloc[i] = df['high'].iloc[i]
                if df['low'].iloc[i] < sar.iloc[i]:
                    sar.iloc[i] = ep.iloc[i]
                    ep.iloc[i] = df['low'].iloc[i]
                    reversal_state.iloc[i] = 2
                    uptrend.iloc[i] = False
            else:
                ep.iloc[i] = df['low'].iloc[i]
                if df['high'].iloc[i] > sar.iloc[i]:
                    sar.iloc[i] = ep.iloc[i]
                    ep.iloc[i] = df['high'].iloc[i]
                    reversal_state.iloc[i] = 1
                    uptrend.iloc[i] = True

    return sar


def cmo(df, period=9):
    df['closegap_cunsum'] = (df['close'] - df['close'].shift(1)).cumsum()
    df['closegap_abs_cumsum'] = abs(df['close'] - df['close'].shift(1)).cumsum()
    # print(df)

    df['CMO'] = (df['closegap_cunsum'] - df['closegap_cunsum'].shift(period)) / (
            df['closegap_abs_cumsum'] - df['closegap_abs_cumsum'].shift(period)) * 100

    del df['closegap_cunsum']
    del df['closegap_abs_cumsum']

    return df['CMO']


def rma(series, period):
    alpha = 1 / period
    rma = pd.Series(index=series.index)
    rma.iloc[0] = 0
    for i in range(1, len(series)):
        if np.isnan(rma.iloc[i - 1]):
            rma.iloc[i] = series.rolling(period).mean().iloc[i]
        else:
            rma.iloc[i] = series.iloc[i] * alpha + (1 - alpha) * nz(rma.iloc[i - 1])

    return rma


def rsi(ohlcv_df, period=14):
    ohlcv_df['up'] = np.where(ohlcv_df['close'].diff(1) > 0, ohlcv_df['close'].diff(1), 0)
    ohlcv_df['down'] = np.where(ohlcv_df['close'].diff(1) < 0, ohlcv_df['close'].diff(1) * (-1), 0)
    rs = rma(ohlcv_df['up'], period) / rma(ohlcv_df['down'], period)
    rsi_ = 100 - 100 / (1 + rs)

    del ohlcv_df['up']
    del ohlcv_df['down']

    return rsi_


def stochrsi(ohlcv_df, period_rsi=14, period_sto=14, k=3, d=3):
    rsi(ohlcv_df, period_rsi)
    ohlcv_df['H_RSI'] = ohlcv_df.RSI.rolling(period_sto).max()
    ohlcv_df['L_RSI'] = ohlcv_df.RSI.rolling(period_sto).min()
    ohlcv_df['StochRSI'] = (ohlcv_df.RSI - ohlcv_df.L_RSI) / (ohlcv_df.H_RSI - ohlcv_df.L_RSI) * 100

    ohlcv_df['StochRSI_K'] = ohlcv_df.StochRSI.rolling(k).mean()
    ohlcv_df['StochRSI_D'] = ohlcv_df.StochRSI_K.rolling(d).mean()

    del ohlcv_df['H_RSI']
    del ohlcv_df['L_RSI']

    return ohlcv_df['StochRSI_D']


def obv(df):
    obv = [0] * len(df)
    for m in range(1, len(df)):
        if df['close'].iloc[m] > df['close'].iloc[m - 1]:
            obv[m] = obv[m - 1] + df['volume'].iloc[m]
        elif df['close'].iloc[m] == df['close'].iloc[m - 1]:
            obv[m] = obv[m - 1]
        else:
            obv[m] = obv[m - 1] - df['volume'].iloc[m]

    return obv


def macd(df, short=9, long=19, signal=9):
    macd = df['close'].ewm(span=short, min_periods=short - 1, adjust=False).mean() - \
                 df['close'].ewm(span=long, min_periods=long - 1, adjust=False).mean()
    macd_signal = macd.ewm(span=signal, min_periods=signal - 1, adjust=False).mean()
    macd_hist = macd - macd_signal

    return macd_hist


def ema_ribbon(df, ema_1=5, ema_2=8, ema_3=13):
    df['EMA_1'] = df['close'].ewm(span=ema_1, min_periods=ema_1 - 1, adjust=False).mean()
    df['EMA_2'] = df['close'].ewm(span=ema_2, min_periods=ema_2 - 1, adjust=False).mean()
    df['EMA_3'] = df['close'].ewm(span=ema_3, min_periods=ema_3 - 1, adjust=False).mean()

    return


def ema_cross(df, ema_1=30, ema_2=60):
    df['EMA_1'] = df['close'].ewm(span=ema_1, min_periods=ema_1 - 1, adjust=False).mean()
    df['EMA_2'] = df['close'].ewm(span=ema_2, min_periods=ema_2 - 1, adjust=False).mean()

    return


def atr(df, period):
    tr = pd.Series(index=df.index)
    tr.iloc[0] = df['high'].iloc[0] - df['low'].iloc[0]
    for i in range(1, len(df)):
        if pd.isna(df['high'].iloc[i - 1]):
            tr.iloc[i] = df['high'].iloc[i] - df['low'].iloc[i]
        else:
            tr.iloc[i] = max(
                max(df['high'].iloc[i] - df['low'].iloc[i], abs(df['high'].iloc[i] - df['close'].iloc[i - 1])),
                abs(df['low'].iloc[i] - df['close'].iloc[i - 1]))
    # print(tr)
    # quit()
    atr = rma(tr, period)

    return atr


def supertrend(df, period, multiplier, cal_st=False):
    hl2 = (df['high'] + df['low']) / 2
    # print(hl2)
    # print(atr(df, period))
    # quit()
    atr_down = hl2 - (multiplier * atr(df, period))
    # atr_down = np.where(df['close'].shift(1) > atr_down.shift(1), max(atr_down, atr_down.shift(1)), atr_down)
    for i in range(len(df)):
        if df['close'].iloc[i - 1] > atr_down[i - 1]:
            atr_down[i] = max(atr_down[i], atr_down[i - 1])
    atr_up = hl2 + (multiplier * atr(df, period))
    # atr_up = np.where(df['close'].shift(1) < atr_up.shift(1), min(atr_up, atr_up.shift(1)), atr_up)
    for i in range(len(df)):
        if df['close'].iloc[i - 1] < atr_up[i - 1]:
            atr_up[i] = min(atr_up[i], atr_up[i - 1])

    atr_trend = pd.Series(index=df.index)
    atr_trend.iloc[0] = 0
    # atr_trend = np.where(atr_trend.shift(1) == np.nan, atr_trend, atr_trend)
    for i in range(1, len(df)):
        if df['close'].iloc[i] > atr_up[i - 1]:
            atr_trend.iloc[i] = 1
        else:
            if df['close'].iloc[i] < atr_down[i - 1]:
                atr_trend.iloc[i] = -1
            else:
                atr_trend.iloc[i] = atr_trend.iloc[i - 1]

    if not cal_st:
        return atr_up, atr_down, atr_trend
    else:
        st = np.where(atr_trend == -1, pd.Series(atr_up), np.nan)
        st = np.where(atr_trend == 1, pd.Series(atr_down), st)
        return st


def ichimoku(ohlc, tenkan_period=9, kijun_period=26, senkou_period=52, chikou_period=26):

    #       Conversion Line     #
    tenkan_sen = pd.Series(
        (
                ohlc["high"].rolling(window=tenkan_period).max()
                + ohlc["low"].rolling(window=tenkan_period).min()
        )
        / 2,
        name="TENKAN",
    )

    #       Base Line       #
    kijun_sen = pd.Series(
        (
                ohlc["high"].rolling(window=kijun_period).max()
                + ohlc["low"].rolling(window=kijun_period).min()
        )
        / 2,
        name="KIJUN",
    )

    #       Leading Span        #
    senkou_span_a = pd.Series(
        ((tenkan_sen + kijun_sen) / 2), name="senkou_span_a"
    )
    senkou_span_b = pd.Series(
        (
                (
                        ohlc["high"].rolling(window=senkou_period).max()
                        + ohlc["low"].rolling(window=senkou_period).min()
                )
                / 2
        ),
        name="SENKOU",
    )

    chikou_span = pd.Series(
        ohlc["close"].shift(chikou_period).rolling(window=chikou_period).mean(),
        name="CHIKOU",
    )

    return senkou_span_a.shift(chikou_period - 1), senkou_span_b.shift(chikou_period - 1)


def ad(df):
    ad_value = (2 * df['close'] - df['low'] - df['high']) / (df['high'] - df['low']) * df['volume']
    ad_ = np.where((df['close'] == df['high']) & (df['close'] == df['low']) | (df['high'] == df['low']), 0, ad_value)
    ad_ = ad_.cumsum()

    return ad_


def lt_trend(df, lambda_bear=0.06, lambda_bull=0.08):
    df['LT_Trend'] = np.NaN
    t_zero = 0
    trend_state = None
    for i in range(1, len(df)):
        pmax = df['close'].iloc[t_zero:i].max()
        pmin = df['close'].iloc[t_zero:i].min()
        delta_bear = (pmax - df['close'].iloc[i]) / pmax
        delta_bull = (df['close'].iloc[i] - pmin) / pmin
        # print(pmax, pmin, delta_bear, delta_bull)

        if delta_bear > lambda_bear and trend_state != 'Bear':
            t_peak = df['close'].iloc[t_zero:i].idxmax()
            t_peak = df.index.to_list().index(t_peak)
            df['LT_Trend'].iloc[t_zero + 1:t_peak + 1] = 'Bull'
            t_zero = t_peak
            trend_state = 'Bear'

        elif delta_bull > lambda_bull and trend_state != 'Bull':
            t_trough = df['close'].iloc[t_zero:i].idxmin()
            t_trough = df.index.to_list().index(t_trough)
            df['LT_Trend'].iloc[t_zero + 1:t_trough + 1] = 'Bear'
            t_zero = t_trough
            trend_state = 'Bull'

        if i == len(df) - 1:
            if pd.isnull(df['LT_Trend'].iloc[i]):
                back_i = i
                while True:
                    back_i -= 1
                    if not pd.isnull(df['LT_Trend'].iloc[back_i]):
                        if df['LT_Trend'].iloc[back_i] == 'Bull':
                            df['LT_Trend'].iloc[back_i + 1:] = 'Bear'
                        else:
                            df['LT_Trend'].iloc[back_i + 1:] = 'Bull'
                        break

            df['LT_Trend'].iloc[0] = df['LT_Trend'].iloc[1]


from scipy.ndimage.filters import gaussian_filter1d


def support_line(df, sigma=10):
    #           Gaussian Curve          #
    df['Gaussian_close'] = gaussian_filter1d(df['close'], sigma=sigma)
    df['Gaussian_close_Trend'] = np.where(df['Gaussian_close'] > df['Gaussian_close'].shift(1), 1, 0)

    df['Support_Line'] = np.NaN
    i = 0
    while i < len(df):
        # print('i :', i)
        #           Find Red           #
        if not df['Gaussian_close_Trend'].iloc[i]:
            #     Find end of the Red       #
            for j in range(i + 1, len(df)):
                if df['Gaussian_close_Trend'].iloc[j]:
                    min_price = df['low'].iloc[i:j].min()
                    # print('i, j, min_price :', i, j, min_price)
                    #       Find Red to Green          #
                    for k in range(j + 1, len(df)):
                        if df['Gaussian_close_Trend'].iloc[k] == 1 and df['Gaussian_close_Trend'].iloc[k - 1] == 0:
                            df['Support_Line'].iloc[j:k] = min_price
                            # print('One S.P Drawing Done!')
                            i = j
                            break
                        else:
                            if k == len(df) - 1:
                                df['Support_Line'].iloc[j:] = min_price
                                i = len(df)
                                break
                    break
        else:
            i += 1

    return


bind_script = '''
var graph = document.getElementsByClassName("plotly-graph-div js-plotly-plot")[0];
var update;

function pan(axis, dx, mode=1) {
    axis += 'axis';
    var [min, max] = graph._fullLayout[axis].range;
    dx *= max - min;
    update[axis+'.range'] = [min+dx, max+mode*dx];
}

function panX(dx) {pan('x', dx)}
function panY(dy) {pan('y', dy)}
function zoomX(dx) {pan('x', dx, -1)}
function zoomY(dy) {pan('y', dy, -1)}

document.addEventListener("keydown", function(e){
    var key = e.key;
    if (e.ctrlKey) key = 'Ctrl+' + key;
    console.log(e, key);
    var fac = 0.1;   // pan and zoom factor
    update = {};
    var extremes = graph._fullData[0]._extremes;  // only first data set
    switch (key) {
        case 'Ctrl+ArrowRight': zoomX(fac); break;
        case 'Ctrl+ArrowLeft': zoomX(-fac); break;
        case 'Ctrl+ArrowUp': zoomY(fac); break;
        case 'Ctrl+ArrowDown': zoomY(-fac); break;
        case '+': zoomX(fac); zoomY(fac); break;
        case '-': zoomX(-fac); zoomY(-fac); break;
        case 'X': case 'U':
             update['xaxis.range'] = [extremes.x.min[0].val, extremes.x.max[0].val];
        case 'Y': case 'U':
             update['yaxis.range'] = [extremes.y.min[0].val, extremes.y.max[0].val]; break;
        case 'x': case 'u':
             update['xaxis.autorange'] = true;
        case 'y': case 'u':
             update['yaxis.autorange'] = true; break;
        case '0': update['yaxis.range[0]'] = 0; break;
        case 'ArrowRight': panX(fac); break;
        case 'ArrowLeft': panX(-fac); break;
        case 'ArrowUp': panY(fac); break;
        case 'ArrowDown': panY(-fac); break;
        case 'Home': panX(-1.); break;
        case 'End': panX(1.); break;
        case 'PageUp': panY(1.); break;
        case 'PageDown': panY(-1.); break;
        default: return;
    }
    Plotly.relayout(graph, update);
});
'''


def script_decorator(func):
    # append binding script
    def function_wrapper(*x):
        return (func(*x) or '') + bind_script
    return function_wrapper

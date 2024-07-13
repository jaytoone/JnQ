import numpy as np
import pandas as pd
import talib


"""
this file exists for moving updateing indicator.py (version & algorithms)
"""

def round(x):
    x = float(x)
    if x > 0.99:
        return 0.999
    else:
        if x < -0.99:
            return -0.999
        else:
            return x

def nz(x, y=0):
    # print(x)
    if np.isnan(x):
        return y
    else:
        return x
            
            
         
def get_fisher(df, 
              period, 
              interval):
    
    high_ = df.high.to_numpy()
    low_ = df.low.to_numpy()
    hl2 = (high_ + low_) / 2

    high_ = talib.MAX(hl2, period)
    low_ = talib.MIN(hl2, period)

    len_df = len(df)
    value = np.zeros(len_df)
    fish = np.zeros(len_df)

    for i in range(1, len_df):
        value[i] = (0.66 * ((hl2[i] - low_[i]) / max(high_[i] - low_[i], .001) - .5) + .67 * nz(value[i - 1]))
        value[i] = round(value[i])
        fish[i] = .5 * np.log((1 + value[i]) / max(1 - value[i], .001)) + .5 * nz(fish[i - 1])

    df['fisher_{}{}'.format(interval, period)] = fish

    return df



def get_BB(df, 
           period, 
           multiple, 
           interval,
           level=2):

    upper, \
    base, \
    lower = talib.BBANDS(df.close, 
                          timeperiod=period, 
                          nbdevup=multiple,
                          nbdevdn=multiple, 
                          matype=0)

    upper_name = 'BB_{}{}_upper'.format(interval, period)
    lower_name = 'BB_{}{}_lower'.format(interval, period)
    base_name = 'BB_{}{}_base'.format(interval, period)

    # drop existing cols.
    try:
        df.drop([upper_name, lower_name, base_name], inplace=True, axis=1)
    except:
        pass

    df['BB_{}{}_upper'.format(interval, period)] = upper
    df['BB_{}{}_lower'.format(interval, period)] = lower
    df['BB_{}{}_base'.format(interval, period)] = base

    if level != 1:
        BB_gap = upper - base
        
        df['BB_{}{}_upper2'.format(interval, period)] = upper + BB_gap
        df['BB_{}{}_lower2'.format(interval, period)] = lower - BB_gap

    return df


def get_MA(df, 
           period, 
           interval):    

    close = df['close'].to_numpy()

    df['MA_{}{}'.format(interval, period)] = talib.MA(close, timeperiod=period)

    return df




def get_CCI(df, 
            period, 
            interval,
            smooth=None):

    """
    v2.1
        replace to interval.

    last confirmed at, 20240621 1700.
    """

    high, low, close = [df[col_].to_numpy() for col_ in ['high', 'low', 'close']]

    if smooth is None:
        df['CCI_{}{}'.format(interval, period)] = talib.CCI(high, low, close, timeperiod=period)
    else:
        df['CCI_{}{}'.format(interval, period)] = talib.MA(talib.CCI(high, low, close, timeperiod=period), timeperiod=smooth)

    return df
    


def get_DC_perentage(df, 
                     period, 
                     interval):

    """
    v1.0

    last confirmed at, 20240530 1919.
    """
    
    DC_upper = df['DC_{}{}_upper'.format(interval, period)].to_numpy()
    DC_lower = df['DC_{}{}_lower'.format(interval, period)].to_numpy()
    close = df.close.to_numpy()
    
    # DC_bandwidth = DC_upper - DC_lower
    DC_percentage = (close - DC_lower) / (DC_upper - DC_lower)
    df['DC_{}{}_percentage'.format(interval, period)] = DC_percentage

    return df



def get_DC(df, 
           period, 
           interval):

    """
    v1.0
    
    v1.1
        assert input value interval

    last confirmed at, 20240617 2234.
    """


    upper_name = 'DC_{}{}_upper'.format(interval, period)
    lower_name = 'DC_{}{}_lower'.format(interval, period)
    base_name = 'DC_{}{}_base'.format(interval, period)

    try:
        df.drop([upper_name, lower_name, base_name], inplace=True, axis=1)
    except:
        pass

    if period != 1:
        hh = talib.MAX(df.high, period).to_numpy()
        ll = talib.MIN(df.low, period).to_numpy()
    else:
        hh = df.high.to_numpy()
        ll = df.low.to_numpy()

    df[upper_name] = hh
    df[lower_name] = ll
    df[base_name] = (hh + ll) / 2

    return df


def get_id_cols(df_chunk):
    
    df_chunk['idVolume'] =  df_chunk.volume.cumsum()
    df_chunk['idRangeH'] =  df_chunk.high.cummax()
    df_chunk['idRangeL'] =  df_chunk.low.cummin()

    return df_chunk



def get_II(df, 
           period,
           interval):

    """
    v1.0
        add assertion type(df.index) == pd.core.indexes.datetimes.DatetimeIndex
    v2.0
        add interval as input.
        return II_{}{}.

    last confirmed at, 2024712 2257.
    """

    assert type(df.index) == pd.core.indexes.datetimes.DatetimeIndex
    idx_startDay = df.index.map(lambda x : '09:00:00' in str(x))

    idx_startDay_arr = np.argwhere(idx_startDay)
    idx_startDay_arr = np.insert(idx_startDay_arr, 0, 0)
    idx_startDay_arr = np.append(idx_startDay_arr, len(df))
    
    df_chunks = [get_id_cols(df.iloc[idx_startDay_arr[n]:idx_startDay_arr[n + 1]]) for n in range(len(idx_startDay_arr)-1)]
    df_res_id = pd.concat(df_chunks)
    df_res_id_arr = df_res_id.to_numpy()
    
    idx_col_close = df_res_id.columns.get_loc('close')
    idx_col_idRangeH = df_res_id.columns.get_loc('idRangeH')
    idx_col_idRangeL = df_res_id.columns.get_loc('idRangeL')
    idx_col_idVolume = df_res_id.columns.get_loc('idVolume')
    
    #  # 357 µs ± 45.8
    # %timeit -n1 -r100 df_res_id['iiiValue'] = (2 * df_res_id['close'] - df_res_id['idRangeH'] - df_res_id['idRangeL']) / (df_res_id['idRangeH'] - df_res_id['idRangeL']) * df_res_id['idVolume']
    # # 39.7 µs ± 13.8 µs per loop
    # %timeit -n1 -r100 df_res_id['iiiValue'] = (2 * df_res_id_arr[:, idx_col_close] - df_res_id_arr[:, idx_col_idRangeH] - df_res_id_arr[:, idx_col_idRangeL]) / (df_res_id_arr[:, idx_col_idRangeH] - df_res_id_arr[:, idx_col_idRangeL]) * df_res_id_arr[:, idx_col_idVolume]
    
    # # 381 µs ± 31.7 µs per loop
    # %timeit -n1 -r100 df_res_id['iiSource'] = df_res_id.iiiValue.rolling(21).sum().to_numpy() / df_res_id.idVolume.rolling(21).sum().to_numpy() * 100
    # # 471 µs ± 52.8 µs per loop
    # %timeit -n1 -r100 df_res_id['iiSource'] = df_res_id.iiiValue.rolling(21).sum() / df_res_id.idVolume.rolling(21).sum() * 100
    
    df_res_id['iiiValue'] = (2 * df_res_id_arr[:, idx_col_close] - df_res_id_arr[:, idx_col_idRangeH] - df_res_id_arr[:, idx_col_idRangeL]) / (df_res_id_arr[:, idx_col_idRangeH] - df_res_id_arr[:, idx_col_idRangeL]) * df_res_id_arr[:, idx_col_idVolume]
    # df_res_id['iiSource'] = df_res_id.iiiValue.rolling(period).sum().to_numpy() / df_res_id.idVolume.rolling(period).sum().to_numpy() * 100
    df_res_id[f'II_{interval}{period}'] = df_res_id.iiiValue.rolling(period).sum().to_numpy() / df_res_id.idVolume.rolling(period).sum().to_numpy() * 100

    return df_res_id
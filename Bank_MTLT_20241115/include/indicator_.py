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
    

def get_SROC(df, ema_length, sroc_roc_length, interval):
    
    """
    v0.1
        - update
            use interger for roc_length.
    
    last confirmed, 20241025 2143.
    """

    def rate_of_change(series, length):
        """Calculate Rate of Change (ROC)."""
        return series.pct_change(periods=length) * 100

    def smoothed_rate_of_change(series, ema_length, roc_length):
        """Calculate Smoothed Rate of Change (SROC)."""
        ema = talib.EMA(series, timeperiod=ema_length)  # Exponential Moving Average (EMA)
        sroc = rate_of_change(ema, roc_length)  # ROC applied to the EMA
        return sroc

    # Inputs (you can modify these)
    # roc_length = 21
    # ema_length = 13
    # sroc_roc_length = 21

    # Using the 'close' prices from your DataFrame `df`
    # df['ROC'] = rate_of_change(df['close'], roc_length)
    df[f'SROC_{interval}{ema_length}&{sroc_roc_length}'] = smoothed_rate_of_change(df['close'], ema_length, sroc_roc_length)
    
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


from fishing_prev_close.traders.only_long_renewal import *

if __name__=='__main__':

    arima_bot = ARIMA_Bot(symbol='ADAUSDT', interval='30m', tp=0.037, leverage_=2, initial_asset=1800, stacked_df_on=True)
    arima_bot.run()

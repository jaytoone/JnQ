from fishing_prev_close.long_close_updown import *

if __name__=='__main__':

    arima_bot = ARIMA_Bot(symbol='DOTUSDT', interval='30m', tp=0.023, leverage_=9, initial_asset=240, stacked_df_on=True)
    arima_bot.run()

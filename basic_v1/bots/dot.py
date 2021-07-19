from basic_v1.trader import *

if __name__=='__main__':

    arima_bot = Trader(symbol='ETHUSDT', interval='30m', tp=0.023, leverage_=9, initial_asset=250, stacked_df_on=True)
    arima_bot.run()

from ARIMA_Bot_long import *

if __name__=='__main__':

    arima_bot = ARIMA_Bot(symbol='BTCUSDT', interval='30m', tp=0.021, leverage_=5, initial_asset=300,
                          stacked_df_on=True)
    arima_bot.run()
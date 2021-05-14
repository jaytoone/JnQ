from arima_bot_long import *

if __name__=='__main__':

    arima_bot = ARIMA_Bot(symbol='ETHUSDT', interval='30m', tp=0.012, leverage_=4, initial_asset=1800,
                          stacked_df_on=True)
    arima_bot.run()
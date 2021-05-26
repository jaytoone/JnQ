from arima_bot_long import *

if __name__=='__main__':

    arima_bot = ARIMA_Bot(symbol='THETAUSDT', interval='30m', tp=0.035, leverage_=5, initial_asset=700,
                          stacked_df_on=True)
    arima_bot.run()
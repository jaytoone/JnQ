from arima_bot_long_ai_plus import *

if __name__=='__main__':

    arima_bot = ARIMA_Bot(symbol='DOTUSDT', interval='30m', tp=0.007, leverage_=5, initial_asset=250,
                          stacked_df_on=True)
    arima_bot.run()
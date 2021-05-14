from arima_bot_long_ai_plus import *

if __name__=='__main__':

    arima_bot = ARIMA_Bot(symbol='ETHUSDT', interval='30m', threshold=0.2614, tp=0.012, leverage_=4, initial_asset=2875,
                          stacked_df_on=True)
    arima_bot.run()
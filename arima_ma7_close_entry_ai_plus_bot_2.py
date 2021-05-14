from ARIMA_Bot_long_ma7_close_entry_ai_plus import *

if __name__=='__main__':

    arima_bot = ARIMA_Bot(symbol='NEOUSDT', interval='30m', tp=0, leverage_=5, initial_asset=220,
                          stacked_df_on=True)
    arima_bot.run()
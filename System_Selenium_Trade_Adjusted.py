import pybithumb
import Funcs_MACD_OSC
import time
from datetime import datetime
import random
import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
# from selenium_move_cursor.MouseActions import move_to_element
import os
import warnings
from fake_useragent import UserAgent
import System_TRIX_TRIX_Funcs

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#       KEY SETTING     #
# with open("Keys.txt") as f:
#     lines = f.readlines()
#     key = lines[0].strip()
#     secret = lines[1].strip()
#     bithumb = pybithumb.Bithumb(key, secret)

#       TRADE IFNO      #
interval = 'minute1'
interval_key1 = '1'
interval_key2 = '3'
fluc_limit = 1.03
CoinVolume = 3
buy_wait = 3  # minute
Profits = 1.0
init = 1
init_tab_close = 1
headless = False

#       CHROME HEADLESS SETTING     #
options = webdriver.ChromeOptions()
options.add_argument('headless')
options.add_argument('window-size=1920x1080')
# options.add_argument("disable-gpu")
ua = UserAgent()
options.add_argument("user-agent=%s" % ua.random)
options.add_argument("lang=ko_KR")

#   CHROME SETTING  #
path = "C:/Users/Lenovo/PycharmProjects/Project_System_Trading/Rapid_Ascend/chromedriver.exe"
class_type = "<class 'selenium.webdriver.remote.webelement.WebElement'>"
if headless == 1:
    driver = webdriver.Chrome(path, chrome_options=options)
else:
    driver = webdriver.Chrome(path)

System_TRIX_TRIX_Funcs.chart_init(driver)
System_TRIX_TRIX_Funcs.open_coin_list(driver)
System_TRIX_TRIX_Funcs.get_coin_data(driver, 'BTC', interval_key2)
while True:
    print(System_TRIX_TRIX_Funcs.get_previous_data(driver))


# for Coin in pybithumb.get_tickers():
#     web_chart = 'https://www.bithumb.com/trade/status/{}_KRW'.format(Coin)
#     driver.get(web_chart)
#         # time.sleep(1)
#     driver.implicitly_wait(3)
#     iFrames = driver.find_elements_by_tag_name('iframe')
#     driver.switch_to.frame(iFrames[1])
#     driver.implicitly_wait(3)
#     # time.sleep(2)
#
#     while True:
#         try:
#             canvas = driver.find_element_by_class_name('chart-page.unselectable.on-widget')
#             print(canvas)
#             break
#         except Exception as e:
#             print('Error in getting canvas :', e)
#
#     #          INTERVAL CHANGE           #
#     while True:
#         try:
#             canvas.click()
#             canvas.send_keys(1)
#             canvas.send_keys(Keys.RETURN)
#             print('sendkey activated')
#             time.sleep(1)
#             interval_time = driver.find_element_by_class_name('pane-legend-title__interval').text
#             if interval_time == ', 1':
#                 break
#
#         except Exception as e:
#             print('Error in interval change :', e)

#     cursor_start_point = driver.find_elements_by_class_name('bg-1kRv1Pf2-')[5]
#     # cursor_end_point = driver.find_element_by_class_name('price-axis')
#     # # print(len(cursor_start_point))
#     # # quit()
#     # loc = cursor_start_point.location
#     # print(loc)
#     # driver.quit()
#     # quit()
#     offset = 1100
#     previous_close = np.NaN
#     previous_offset = offset
#     # while True:
#
#     action = ActionChains(driver)
#     action.move_to_element_with_offset(cursor_start_point, offset, 0).perform()
#     time.sleep(2)
#     #           GET DATA FROM THE CHARTS            #
#     indicator_value = driver.find_elements_by_class_name('pane-legend-item-value-container')
#     print(indicator_value[0].text.split('\n'))
#     close = float(indicator_value[0].text.split('\n')[3].replace("종", ""))

        #       FISHER 틱마다 값이 달라지니까 그 점을 이용한다.
        # if previous_close != close:
        #     print(previous_offset - offset)
        #     previous_close = close
        #     previous_offset = offset
        # # action.click()
        # # action.perform()
        #
        # #       한 픽셀당 오프셋 거리      #
        # offset -= 6

quit()

# System_Selenium_Funcs.open_coin_list()

trade_cnt = 0
while True:

    #               Finding Buy Signal              #
    buy_signal = 0
    web_chart_list = list()
    start = time.time()

    while True:

        #       60분 지나면 web_chart_list 초기화      #
        if time.time() - start > 60 * 15:
            web_chart_list = list()
            start = time.time()

        #              GET TOP COIN LIST             #
        try:
            #           TOP COIN LIST TAB으로 가서      #
            driver.switch_to.window(driver.window_handles[1])
            driver.implicitly_wait(3)
            #           주기적으로 SORTING 작업을 한다.         #
            sort_btn = driver.find_elements_by_class_name('sorting')[2]
            sort_btn.click()
            sort_btn.click()
            driver.implicitly_wait(3)

            coin_list = driver.find_element_by_class_name('coin_list')
            coins_info = coin_list.find_elements_by_tag_name('tr')
            TopCoin = list()
            for coin_info in coins_info:
                TopCoin.append(coin_info.find_element_by_class_name('sort_coin').text.split('/')[0])
                if len(TopCoin) >= CoinVolume:
                    break
            print(TopCoin)

        except Exception as e:
            print('Error in getting TopCoin :', e)
            continue
        # TopCoin = ['QBZ']

        for Coin in TopCoin:

            #      후보 코인창이 7개 미만이면 창을 추가한다.      #
            if len(driver.window_handles) < 7:

                System_Selenium_Funcs.get_coinly_data(driver, Coin, interval_key1)
                System_Selenium_Funcs.attach_chart(driver, Coin, interval_key2)
                # print(to_recent_span)
                # quit()
                #           GET DATA FROM THE CHARTS            #
                previous_interval_key = interval_key1
                previous_data = [np.NaN] * 3
                data = [np.NaN] * 3
                while True:

                    current_datetime = datetime.now()
                    if current_datetime.second >= 59:
                        print(datetime.now())

                        get_data_time = time.time()
                        while True:

                            try:
                                # # #           MOVE CURSOR TO RECENT DATA          #
                                # # to_recent_span = driver.find_element_by_class_name('wrap-18oKCBRc-')
                                # action.move_to_element(to_recent_span).perform()
                                # # # print(to_recent_span)
                                # time.sleep(1)
                                # # print('cursor moved :', datetime.now())
                                # # canvas.send_keys(Keys.F5)
                                #
                                # #           DATA INFO               #
                                # data_info = driver.find_elements_by_class_name('pane-legend-title__description')
                                # coin_info = data_info[0].text
                                # indicator_info = data_info[1].text
                                # print('got data :', datetime.now(), interval_time, coin_info, indicator_info)
                                # action.move_to_element(to_recent_span).perform()
                                # time.sleep(1)

                                #           GET DATA FROM THE CHARTS            #
                                interval_time = driver.find_element_by_class_name('pane-legend-title__interval').text
                                indicator_value = driver.find_elements_by_class_name('pane-legend-item-value-container')
                                close = float(indicator_value[0].text.split('\n')[3].replace("종", ""))
                                fisher = float(indicator_value[3].text.split('\n')[0].replace("−", "-"))
                                # print(datetime.now(), close, fisher)
                                # print()
                                # print(close, indicator_info, Funcs_MACD_OSC.clearance(close))
                                time.sleep(1)

                                if interval_time == ', 1':
                                    data[0] = close
                                    data[1] = fisher
                                    print('previous data at 1min :', previous_data)
                                    print(data)
                                    print()
                                    driver.switch_to.window(driver.window_handles[2])
                                    iFrames = driver.find_elements_by_tag_name('iframe')
                                    driver.implicitly_wait(3)
                                    driver.switch_to.frame(iFrames[1])
                                    driver.implicitly_wait(3)
                                    break
                                    # quit()

                                else:  # interval time = 3 min
                                    data[2] = fisher
                                    print('previous data at 3min :', previous_data)
                                    driver.switch_to.window(driver.window_handles[0])
                                    iFrames = driver.find_elements_by_tag_name('iframe')
                                    driver.implicitly_wait(3)
                                    driver.switch_to.frame(iFrames[1])
                                    driver.implicitly_wait(3)

                                # break
                                # print()

                            except Exception as e:
                                print('Error in getting indicator value :', e)

                            print('get data time :', time.time() - get_data_time)

                        #       PREVIOUS DATA & DATA COMPARING      #

                        #       IF COMPARING DONE,     #
                        previous_data = list()
                        for value in data:
                            previous_data.append(value)
                            #   '=' 을 사용하면 안돼는 것 같다.    #
                            #   previoud_data가 코드 실행 도중 변경된다..

                    # break

            # try:
            #
            # except Exception as e:
            #     print("Error in getting trade_state information :", e)

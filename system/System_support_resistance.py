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
import pickle
# from keras.models import load_model
import os
from Make_X_low_high_gaussian_support_line import low_high
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#   Chrome Setting  #
path = "C:/Users/Lenovo/PycharmProjects/Project_System_Trading/Rapid_Ascend/chromedriver.exe"
driver = webdriver.Chrome(path)

#       KEY SETTING     #
# with open("Keys.txt") as f:
#     lines = f.readlines()
#     key = lines[0].strip()
#     secret = lines[1].strip()
#     bithumb = pybithumb.Bithumb(key, secret)

#       MODEL INFO      #
input_data_length = 54
model_num = '124_ohlc'

#       TRADE IFNO      #
interval = 'minute1'
interval_key1 = Keys.NUMPAD1
# interval_key2 = Keys.NUMPAD0
fluc_limit = 1.03
CoinVolume = 20
buy_wait = 3  # minute
Profits = 1.0

#       Model Fitting       #
# model = load_model('./model/rapid_ascending %s_%s.hdf5' % (input_data_length, model_num))

while True:

    #               Finding Buy Signal              #
    buy_signal = 0
    web_chart_list = list()
    start = time.time()

    while True:

        #       60분 지나면 web_chart_list 초기화      #
        if time.time() - start > 60 * 60:
            web_chart_list = list()
            start = time.time()

        #           Making TopCoin List         #
        try:
            TopCoin = pybithumb.get_top_coin(CoinVolume)
        except Exception as e:
            continue
        # TopCoin = ['LRC']

        for Coin in TopCoin:

            #      후보 코인창이 7개 미만이면 창을 추가한다.      #
            if len(driver.window_handles) < 7:

                try:
                    while True:
                        if datetime.now().second >= 5:
                            break

                    #   User-Agent 설정 & IP 변경으로 크롤링 우회한다.
                    print('Loading %s data...' % Coin)
                    #       이거 안하면 ip 제한 먹힘     #
                    time.sleep(1 / 130)

                    #               JUST FIND THE V SHAPE SUPPORT_LINE SIGNAL           #
                    #       unpacking configuration for prediction : X_test, buy_price, _, exit_price   #
                    ohlcv = pybithumb.get_candlestick(Coin, chart_instervals='1m')
                    # trade_state = low_high(Coin, input_data_length, interval, crop_size=input_data_length, fluc_limit=fluc_limit)

                    #       GET CMO     #
                    Funcs_MACD_OSC.cmo(ohlcv)

                    #       V Shape을 찾으면 / 창에 보여지지 않았으면      #
                    if (ohlcv.CMO.iloc[-1] >= -80) and (Coin not in web_chart_list):
                        # if trade_state[-1] != 2. and Coin not in web_chart_list:

                        print(Coin, datetime.now())
                        web_chart = 'https://www.bithumb.com/trade/status/{}_KRW'.format(Coin)
                        web_chart_list.append(Coin)

                        #       해당 Coin이 1개보다 많으면 새탭열고 보여주기 : 처음에 탭을 열 필요는 없으니까    #
                        if len(web_chart_list) > 1:
                            #           새 탭을 열기위해 존재하는 마지막 탭으로 이동한다.        #
                            driver.switch_to.window(driver.window_handles[-1])
                            driver.execute_script("window.open('');")
                            # driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.TAB)
                            # ActionChains(driver).key_down(Keys.CONTROL).send_keys('t').key_up(Keys.CONTROL).perform()
                            driver.switch_to.window(driver.window_handles[-1])
                        # print(driver.window_handles)

                        driver.get(web_chart)
                        driver.implicitly_wait(3)
                        iFrames = driver.find_elements_by_tag_name('iframe')
                        # for i, iframe in enumerate(iFrames):
                        #     try:
                        #         print('%d번째 iframe 입니다.' % i)
                        #
                        #         # i 번째 iframe으로 변경합니다.
                        #         driver.switch_to.frame(iFrames[i])
                        #
                        #         # 변경한 iframe 안의 소스를 확인합니다.
                        #         print(driver.page_source)
                        #
                        #         # 원래 frame으로 돌아옵니다.
                        #         driver.switch_to.default_content()
                        #     except:
                        #         # exception이 발생했다면 원래 frame으로 돌아옵니다.
                        #         driver.switch_to.default_content()
                        #
                        #         # 몇 번째 frame에서 에러가 났었는지 확인합니다.
                        #         print('pass by except : iFrames[%d]' % i)
                        #
                        #         # 다음 for문으로 넘어갑니다.
                        #         pass
                        driver.switch_to.frame(iFrames[1])
                        action = ActionChains(driver)

                        #           CHART INITIATION        #
                        if len(web_chart_list) == 1:
                            #           WAIT FOR WEBDATA LOADING        #
                            driver.implicitly_wait(3)
                            canvas = driver.find_element_by_class_name('chart-page.unselectable.on-widget')
                            #           OPEN INDICATOR CLOSING BUTTON           #
                            while True:
                                try:
                                    indicator_opener = driver.find_element_by_class_name('expand.closed')
                                    indicator_opener.click()
                                    break
                                except Exception as e:
                                    try:
                                        driver.find_element_by_class_name('expand')
                                    except:
                                        print('Error in indicator opener :', e)
                                    else:
                                        break

                            #       DELETE ALL INDICATORS       #
                            while True:
                                try:
                                    indicator_delete_btns = driver.find_elements_by_class_name(
                                        'pane-legend-icon.apply-common-tooltip.delete')
                                    if len(indicator_delete_btns) == 0:
                                        break
                                    indicator_delete_btns[0].click()
                                except Exception as e:
                                    print("Error in deleting indicator :", e)

                            #           ADD SOME INDICATOR          #
                            indicator_search_btn = driver.find_element_by_id('header-toolbar-indicators')
                            indicator_search_btn.click()
                            # Stochastic_RSI_btn = driver.find_element_by_xpath('//*[@title="스토캐스틱 RSI (Stochastic RSI)"]')

                            # while True:
                            #     try:
                            #         CMO_btn = driver.find_element_by_xpath('//*[@title="샹드 모멘텀 오실레이터"]')
                            #         CMO_btn.click()
                            #         break
                            #     except Exception as e:
                            #         pass
                            while True:
                                try:
                                    Fisher_btn = driver.find_element_by_xpath(
                                        '//*[@title="피셔 트랜스폼 (Fisher Transform)"]')
                                    Fisher_btn.click()
                                    break
                                except Exception as e:
                                    print('Error in find fisher :', e)

                            #           INDICATOR SEARCHING DONE        #
                            indicator_search_close_btn = driver.find_element_by_class_name(
                                "tv-dialog__close.js-dialog__close")
                            indicator_search_close_btn.click()

                            #       INDICATOR SETTING      #

                            while True:
                                try:
                                    Fisher_set_btn = driver.find_element_by_xpath('//*[@title="설정"]')
                                    Fisher_set_btn.click()
                                    break
                                except Exception as e:
                                    print('Error in fisher set :', e)

                            while True:
                                try:
                                    Period_line = driver.find_element_by_class_name('innerInput-29Ku0bwF-')
                                    Period_line.click()
                                    break
                                except Exception as e:
                                    print('Error in period line :', e)

                            Period_line.send_keys(Keys.BACKSPACE)
                            Period_line.send_keys(Keys.NUMPAD6)
                            Period_line.send_keys(Keys.NUMPAD0)
                            confirm_btn = driver.find_element_by_class_name('button-1iktpaT1-.size-m-2G7L7Qat-.intent-primary-1-IOYcbg-.appearance-default-dMjF_2Hu-')
                            confirm_btn.click()

                            #           CLICKING INTERVAL TIME           #
                            canvas.click()
                            canvas.send_keys(interval_key1)

                            #       IF INTERVAL KEY2 IS EXIST       #
                            try:
                                interval_key2
                                time.sleep(0.5)
                                canvas.send_keys(interval_key2)
                            except Exception as e:
                                print('Error in send interval keys:', e)

                            while True:
                                #           TO ADJUST INTERVAL CHANGE YOU HAVE TO DOUBLE ENTER      #
                                canvas.send_keys(Keys.RETURN)
                                #       CHECK THE INTERVAL      #
                                interval_time = driver.find_element_by_class_name('pane-legend-title__interval').text
                                if interval_time == ', 1':
                                    break

                        else:
                            #           OPEN INDICATOR OPENER           #
                            while True:
                                try:
                                    indicator_opener = driver.find_element_by_class_name('expand.closed')
                                    indicator_opener.click()
                                    break
                                except Exception as e:
                                    print('Error in indicator opener :', e)

                        #           MOVE CURSOR TO RECENT DATA          #
                        to_recent_span = driver.find_element_by_class_name('wrap-18oKCBRc-')
                        action.move_to_element(to_recent_span).perform()
                        print(to_recent_span)
                        time.sleep(1)

                        #           GET DATA FROM THE CHARTS            #
                        indicator_value = driver.find_elements_by_class_name('pane-legend-item-value-container')
                        # fisher = float(indicator_value[1].text.split('\n')[0][:])
                        fisher = float(indicator_value[1].text.split('\n')[0].replace("−", "-"))
                        print(fisher)
                        print()

                except Exception as e:
                    print("Error in getting trade_state information :", e)

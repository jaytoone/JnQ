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
import os
import warnings
from fake_useragent import UserAgent
import System_Selenium_Funcs

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
interval_key1 = Keys.NUMPAD1
interval_key2 = Keys.NUMPAD3
fluc_limit = 1.03
CoinVolume = 7
buy_wait = 3  # minute
Profits = 1.0
init = 1
init_tab_close = 1
headless = 0

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

System_Selenium_Funcs.chart_init(driver, interval_key1=interval_key1)
# System_Selenium_Funcs.open_coin_list()


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

        #           Making TopCoin List         #
        # try:
        #     TopCoin = pybithumb.get_top_coin(CoinVolume)
        # except Exception as e:
        #     continue
        TopCoin = ['QBZ']

        for Coin in TopCoin:

            #      후보 코인창이 7개 미만이면 창을 추가한다.      #
            if len(driver.window_handles) < 7:

                try:
                    while True:
                        if datetime.now().second >= 5:
                            break

                    #               JUST FIND THE V SHAPE SUPPORT_LINE SIGNAL           #
                    #       unpacking configuration for prediction : X_test, buy_price, _, exit_price   #
                    # ohlcv = pybithumb.get_candlestick(Coin, chart_instervals='1m')
                    # trade_state = low_high(Coin, input_data_length, interval, crop_size=input_data_length, fluc_limit=fluc_limit)

                    #       GET FISHER VALUE     #

                    print(Coin, datetime.now())
                    web_chart = 'https://www.bithumb.com/trade/status/{}_KRW'.format(Coin)

                    if Coin in web_chart_list:
                        continue

                    #       처음 이후로는 탭 추가       #
                    # if init == 0:
                    #     #           새 탭을 열기위해 존재하는 마지막 탭으로 이동한다.        #
                    driver.switch_to.window(driver.window_handles[-1])
                    driver.execute_script("window.open('');")
                    #     # driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.TAB)
                    #     # ActionChains(driver).key_down(Keys.CONTROL).send_keys('t').key_up(Keys.CONTROL).perform()
                    driver.switch_to.window(driver.window_handles[-1])

                    # driver.execute_script(
                    #     "Object.defineProperty(navigator, 'plugins', {get: function() {return[1, 2, 3, 4, 5]}})")
                    # driver.execute_script(
                    #     "Object.defineProperty(navigator, 'languages', {get: function() {return ['ko-KR', 'ko']}})")
                    # driver.execute_script(
                    #     "const getParameter = WebGLRenderingContext.getParameter;WebGLRenderingContext.prototype.getParameter"
                    #     " = function(parameter) {if (parameter === 37445) {return 'NVIDIA Corporation'} if (parameter === 37446)"
                    #     " {return 'NVIDIA GeForce GTX 980 Ti OpenGL Engine';}return getParameter(parameter);};")
                    # plugins_length = driver.find_element_by_css_selector('#plugins-length').text
                    # languages = driver.find_element_by_css_selector('#languages').text
                    # webgl_vendor = driver.find_element_by_css_selector('#webgl-vendor').text
                    # webgl_renderer = driver.find_element_by_css_selector('#webgl-renderer').text
                    #
                    # print('Plugin length: ', plugins_length)
                    # print('languages: ', languages)
                    # print('WebGL Vendor: ', webgl_vendor)
                    # print('WebGL Renderer: ', webgl_renderer)
                    # quit()

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
                    if init == 1:

                        init = 0
                        #           WAIT FOR WEBDATA LOADING        #
                        driver.implicitly_wait(3)
                        canvas = driver.find_element_by_class_name('chart-page.unselectable.on-widget')
                        #           OPEN INDICATOR CLOSING BUTTON           #
                        while True:
                            try:
                                time.sleep(1)
                                indicator_opener = driver.find_element_by_class_name('expand.closed')
                                indicator_opener.click()
                                if str(type(indicator_opener)) == class_type:
                                    break
                            except Exception as e:
                                try:    # 이미 열려져있는 경우 CHECK
                                    expanded = driver.find_element_by_class_name('expand')
                                    if str(type(expanded)) == class_type:
                                        break
                                except Exception as e:
                                    print('Error in indicator opener :', e)

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
                        # try:
                        #     interval_key2
                        #     time.sleep(0.5)
                        #     canvas.send_keys(interval_key2)
                        # except Exception as e:
                        #     print('Error in send interval keys:', e)

                        while True:
                            #           TO ADJUST INTERVAL CHANGE YOU HAVE TO DOUBLE ENTER      #
                            canvas.send_keys(Keys.RETURN)
                            #       CHECK THE INTERVAL      #
                            interval_time = driver.find_element_by_class_name('pane-legend-title__interval').text
                            if interval_time == ', 1':
                                break

                    else:
                        #          OPEN INDICATOR OPENER           #
                        while True:
                            try:
                                time.sleep(1)
                                indicator_opener = driver.find_element_by_class_name('expand.closed')
                                indicator_opener.click()
                                if str(type(indicator_opener)) == class_type:
                                    break
                            except Exception as e:
                                try:  # 이미 열려져있는 경우 CHECK
                                    expanded = driver.find_element_by_class_name('expand')
                                    if str(type(expanded)) == class_type:
                                        break
                                except Exception as e:
                                    print('Error in indicator opener :', e)

                    #           MOVE CURSOR TO RECENT DATA          #
                    to_recent_span = driver.find_element_by_class_name('wrap-18oKCBRc-')
                    action.move_to_element(to_recent_span).perform()
                    print(to_recent_span)
                    time.sleep(1)

                    # #           GET DATA FROM THE CHARTS            #
                    previous_interval_key = interval_key1
                    previous_data = [np.NaN] * 3
                    data = [np.NaN] * 3
                    while True:
                        # if datetime.now().second >= 57:
                            print(datetime.now())

                            get_data_time = time.time()
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
                                action.move_to_element(to_recent_span).perform()
                                time.sleep(1)

                                #           GET DATA FROM THE CHARTS            #
                                interval_time = driver.find_element_by_class_name('pane-legend-title__interval').text
                                indicator_value = driver.find_elements_by_class_name('pane-legend-item-value-container')
                                close = float(indicator_value[0].text.split('\n')[3].replace("종", ""))
                                fisher = float(indicator_value[1].text.split('\n')[0].replace("−", "-"))
                                print(datetime.now(), close, fisher)
                                print()
                                # print(close, indicator_info, Funcs_MACD_OSC.clearance(close))
                                time.sleep(1)

                                if interval_time == ', 1':
                                    data[0] = close
                                    data[1] = fisher
                                else:  # interval time = 3 min
                                    data[2] = fisher
                                    print('previous_data :', previous_data)
                                    print('data :', data)
                                    print()
                                # break
                                # print()

                            except Exception as e:
                                print('Error in getting indicator value :', e)
                            print('get data time :', time.time() - get_data_time)
                    #
                    #         #           CLICKING INTERVAL TIME           #
                    #         while True:
                    #             try:
                    #                 if previous_interval_key == interval_key1:
                    #                     canvas.send_keys(interval_key2)
                    #                 else:
                    #                     canvas.send_keys(interval_key1)
                    #                 break
                    #
                    #             except Exception as e:
                    #                 print('Error in clicking canvas :', e)
                    #
                    #         #       IF INTERVAL KEY2 IS EXIST       #
                    #         # try:
                    #         #     interval_key2
                    #         #     time.sleep(0.5)
                    #         #     canvas.send_keys(interval_key2)
                    #         # except Exception as e:
                    #         #     print('Error in send interval keys:', e)
                    #
                    #         interval_change_time = time.time()
                    #         while True:
                    #             try:
                    #                 #           TO ADJUST INTERVAL CHANGE YOU HAVE TO DOUBLE ENTER      #
                    #                 canvas.send_keys(Keys.RETURN)
                    #                 #       CHECK THE INTERVAL      #
                    #                 interval_time = driver.find_element_by_class_name(
                    #                     'pane-legend-title__interval').text
                    #
                    #                 if previous_interval_key == interval_key1 and interval_time == ', 3':
                    #                     previous_interval_key = interval_key2
                    #                     break
                    #                 if previous_interval_key == interval_key2 and interval_time == ', 1':
                    #                     previous_interval_key = interval_key1
                    #                     break
                    #
                    #             except Exception as e:
                    #                 print('Error in interval change :', e)
                    #         print('interval change time :', time.time() - interval_change_time)
                    #
                    # # if fisher > -1.5 and len(driver.window_handles) > 1:
                    # #     driver.close()
                    # #     if len(driver.window_handles) == 2 and init_tab_close == 1:
                    # #         driver.switch_to.window(driver.window_handles[0])
                    # #         driver.close()
                    # #         init_tab_close = 0
                    # #
                    # # elif fisher <= -1.5:
                    # #     web_chart_list.append(Coin)

                except Exception as e:
                    print("Error in getting trade_state information :", e)

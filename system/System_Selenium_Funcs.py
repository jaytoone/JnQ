from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import time

class_type = "<class 'selenium.webdriver.remote.webelement.WebElement'>"


def chart_init(driver):
    #                                        SCRAPING ENV CONFIG                                             #
    web_chart = 'https://www.bithumb.com/trade/status/{}_KRW'.format('BTC')
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
    driver.implicitly_wait(3)

    #                       CHART INITIATION                    #

    #           WAIT FOR WEBDATA LOADING        #

    #           OPEN INDICATOR EXPANDING BUTTON           #
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
    #         SMMA_btn = driver.find_element_by_xpath(
    #             '//*[@title="스무디드 무빙 애버리지 (Smoothed Moving Average)"]')
    #         SMMA_btn.click()
    #         break
    #     except Exception as e:
    #         print('Error in finding SMMA :', e)
    while True:
        try:
            indicator_btn1 = driver.find_element_by_xpath(
                '//*[@title="돈치안 채널 (Donchian Channels)"]')
            indicator_btn1.click()
            break
        except Exception as e:
            print('Error in finding DC :', e)
    while True:
        try:
            indicator_btn2 = driver.find_element_by_xpath(
                '//*[@title="트릭스 (TRIX)"]')
            indicator_btn2.click()
            break
        except Exception as e:
            print('Error in finding TRIX :', e)
    while True:
        try:
            indicator_btn3 = driver.find_element_by_xpath(
                '//*[@title="피셔 트랜스폼 (Fisher Transform)"]')
            indicator_btn3.click()
            break
        except Exception as e:
            print('Error in finding Fisher :', e)

    #           INDICATOR SEARCHING DONE        #
    indicator_search_close_btn = driver.find_element_by_class_name("tv-dialog__close.js-dialog__close")
    indicator_search_close_btn.click()

    #       INDICATOR SETTING      #
    while True:
        try:
            indicator_btn1 = driver.find_elements_by_xpath('//*[@title="설정"]')[0]
            indicator_btn1.click()
            break
        except Exception as e:
            print('Error in first indicator set :', e)
    while True:
        try:
            Period_line = driver.find_element_by_class_name('innerInput-29Ku0bwF-')
            Period_line.click()
            break
        except Exception as e:
            print('Error in period line :', e)
    Period_line.send_keys(Keys.BACKSPACE)
    Period_line.send_keys(Keys.BACKSPACE)
    Period_line.send_keys(80)
    confirm_btn = driver.find_element_by_class_name(
        'button-1iktpaT1-.size-m-2G7L7Qat-.intent-primary-1-IOYcbg-.appearance-default-dMjF_2Hu-')
    confirm_btn.click()
    #       DC SETTING      #
    while True:
        try:
            indicator_btn2 = driver.find_elements_by_xpath('//*[@title="설정"]')[1]
            indicator_btn2.click()
            break
        except Exception as e:
            print('Error in second indicator set :', e)
    while True:
        try:
            Period_line = driver.find_element_by_class_name('innerInput-29Ku0bwF-')
            Period_line.click()
            break
        except Exception as e:
            print('Error in period line :', e)
    Period_line.send_keys(Keys.BACKSPACE)
    Period_line.send_keys(Keys.BACKSPACE)
    Period_line.send_keys(20)
    confirm_btn = driver.find_element_by_class_name(
        'button-1iktpaT1-.size-m-2G7L7Qat-.intent-primary-1-IOYcbg-.appearance-default-dMjF_2Hu-')
    confirm_btn.click()
    #       FISHER SETTING      #
    while True:
        try:
            indicator_btn3 = driver.find_elements_by_xpath('//*[@title="설정"]')[2]
            indicator_btn3.click()
            break
        except Exception as e:
            print('Error in third indicator set :', e)
    while True:
        try:
            Period_line = driver.find_element_by_class_name('innerInput-29Ku0bwF-')
            Period_line.click()
            break
        except Exception as e:
            print('Error in period line :', e)
    Period_line.send_keys(Keys.BACKSPACE)
    Period_line.send_keys(Keys.BACKSPACE)
    Period_line.send_keys(60)
    confirm_btn = driver.find_element_by_class_name(
        'button-1iktpaT1-.size-m-2G7L7Qat-.intent-primary-1-IOYcbg-.appearance-default-dMjF_2Hu-')
    confirm_btn.click()

    return


def open_coin_list(driver):
    #           OPEN NEW TAB FOR FINDING SIGNAL     #
    driver.execute_script("window.open('');")
    driver.switch_to.window(driver.window_handles[1])

    #                     GET TOP FLUC COIN AT 12 HOUR PERIOD AT FIRST TAB               #
    web_chart = 'https://www.bithumb.com'
    driver.get(web_chart)
    driver.implicitly_wait(3)

    try:
        pop_close_btn = driver.find_element_by_class_name('pop_close.ico_close')
        pop_close_btn.click()
    except Exception as e:
        print('Error in pop closing :', e)

    period_btn = driver.find_element_by_id('selectRealTick')
    period_btn.click()
    period_btn.send_keys(Keys.DOWN)
    period_btn.send_keys(Keys.DOWN)
    period_btn.send_keys(Keys.RETURN)
    time.sleep(1)
    sort_btn = driver.find_elements_by_class_name('sorting')[2]
    sort_btn.click()
    driver.implicitly_wait(3)

    return


# def get_coinly_data(driver, Coin, interval_key1):
#
#     #       첫 탭에 출력       #
#     driver.switch_to.window(driver.window_handles[0])  # 첫 생성탭에 안하면 세팅 초기화된다.
#     # driver.execute_script("window.open('');")
#     # driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.TAB)
#     # ActionChains(driver).key_down(Keys.CONTROL).send_keys('t').key_up(Keys.CONTROL).perform()
#     # driver.switch_to.window(driver.window_handles[-1])
#     web_chart = 'https://www.bithumb.com/trade/status/{}_KRW'.format(Coin)
#     driver.get(web_chart)
#     driver.implicitly_wait(3)
#     iFrames = driver.find_elements_by_tag_name('iframe')
#     driver.switch_to.frame(iFrames[1])
#
#     #           OPEN INDICATOR EXPANDING BUTTON           #
#     while True:
#         try:
#             time.sleep(1)
#             indicator_opener = driver.find_element_by_class_name('expand.closed')
#             indicator_opener.click()
#             if str(type(indicator_opener)) == class_type:
#                 break
#         except Exception as e:
#             try:  # 이미 열려져있는 경우 CHECK
#                 expanded = driver.find_element_by_class_name('expand')
#                 if str(type(expanded)) == class_type:
#                     break
#             except Exception as e:
#                 print('Error in indicator opener :', e)
#
#     #           GET CANVAS          #
#     #       CANVAS CLICK이랑 같이 두면 INTERVAL 입력이 안돼는 수가 있다     #
#     while True:
#         try:
#             canvas = driver.find_element_by_class_name('chart-page.unselectable.on-widget')
#             break
#         except Exception as e:
#             print('Error in getting canvas :', e)
#             time.sleep(1)
#
#     #          INTERVAL CHANGE           #
#     while True:
#         try:
#             canvas.click()
#             canvas.send_keys(interval_key1)
#             canvas.send_keys(Keys.RETURN)
#             time.sleep(1)
#             interval_time = driver.find_element_by_class_name('pane-legend-title__interval').text
#             if interval_time == ', 1':
#                 break
#
#         except Exception as e:
#             print('Error in interval change :', e)
#             return
#
#     chart_error = False
#     opener_error = False
#     while True:
#         try:
#             #           MOVE CURSOR TO RECENT DATA          #
#             action = ActionChains(driver)
#             to_recent_span = driver.find_element_by_class_name('wrap-18oKCBRc-')
#             action.move_to_element(to_recent_span).perform()
#             time.sleep(1)
#
#             #           DATA INFO               #
#             interval_time = driver.find_element_by_class_name('pane-legend-title__interval').text
#             data_info = driver.find_elements_by_class_name('pane-legend-title__description')
#             coin_info = data_info[0].text
#             smma_info = data_info[1].text
#             dc_info = data_info[2].text
#             fisher_info = data_info[3].text
#
#             #           CHECK CHART SET           #
#             if interval_time != ', 1' or smma_info != 'SMMA (200, close)' \
#                     or dc_info != 'DC (80)' \
#                     or fisher_info != 'Fisher (60)':
#                 chart_error = True
#                 break
#
#             #           GET DATA FROM THE CHARTS            #
#             indicator_value = driver.find_elements_by_class_name('pane-legend-item-value-container')
#             close = float(indicator_value[0].text.split('\n')[3].replace("종", ""))
#             smma = float(indicator_value[1].text)
#             dc = float(indicator_value[2].text.split('\n')[0])
#             fisher = float(indicator_value[3].text.split('\n')[0].replace("−", "-"))
#             print("    " + coin_info + interval_time)
#             print("    " + smma_info, smma, end=' | ')
#             print(dc_info, dc, end=' | ')
#             print(fisher_info, fisher)
#             break
#             # print()
#
#         except Exception as e:
#             print('Error in getting indicator value :', e)
#             opener_error = True
#             break
#
#     if chart_error:
#         print('Chart Configuration miss')
#         quit()
#
#     elif opener_error:
#         print('Opener miss')
#         return
#
#     # elif fisher <= long_signal_value:
#     #     print('Fisher value at long position :', Coin, fisher)
#     #     buy_switch = 1
#     #     break   # GET OUT OF THE FOR LOOP
#
#     else:
#         time.sleep(1 / 130)
#
#     return close
import numpy as np


def get_coin_data(driver, Coin, interval_key1, period1=80, period2=20, period3=60):

    #       첫 탭에 출력       #
    driver.switch_to.window(driver.window_handles[0])  # 첫 생성탭에 안하면 세팅 초기화된다.
    # driver.execute_script("window.open('');")
    # driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.TAB)
    # ActionChains(driver).key_down(Keys.CONTROL).send_keys('t').key_up(Keys.CONTROL).perform()
    # driver.switch_to.window(driver.window_handles[-1])
    web_chart = 'https://www.bithumb.com/trade/status/{}_KRW'.format(Coin)
    driver.get(web_chart)
    driver.implicitly_wait(3)
    iFrames = driver.find_elements_by_tag_name('iframe')
    driver.switch_to.frame(iFrames[1])

    #           OPEN INDICATOR EXPANDING BUTTON           #
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
                return

    #           GET CANVAS          #
    #       CANVAS CLICK이랑 같이 두면 INTERVAL 입력이 안돼는 수가 있다     #
    while True:
        try:
            canvas = driver.find_element_by_class_name('chart-page.unselectable.on-widget')
            break
        except Exception as e:
            print('Error in getting canvas :', e)
            return

    #          INTERVAL CHANGE           #
    while True:
        try:
            canvas.click()
            canvas.send_keys(interval_key1)
            canvas.send_keys(Keys.RETURN)
            time.sleep(1)
            interval_time = driver.find_element_by_class_name('pane-legend-title__interval').text
            if interval_time == ', %s' % interval_key1:
                break

        except Exception as e:
            print('Error in interval change :', e)
            return

    chart_error = False
    opener_error = False
    cursor_start_point = driver.find_elements_by_class_name('bg-1kRv1Pf2-')[5]
    last_offset = 1074
    offset = last_offset
    candle_value = list()
    tick_change_cnt = 0
    return_list = list()
    while True:
        try:
            #           MOVE CURSOR TO BACK DATA          #
            action = ActionChains(driver)
            action.move_to_element_with_offset(cursor_start_point, offset, 0).perform()

            #           DATA INFO CHECK               #
            #               처음에만 검사              #
            if offset == last_offset:
                interval_time = driver.find_element_by_class_name('pane-legend-title__interval').text
                data_info = driver.find_elements_by_class_name('pane-legend-title__description')
                coin_info = data_info[0].text
                info1 = data_info[1].text
                info2 = data_info[2].text
                info3 = data_info[3].text

                #           CHECK CHART SET           #
                if interval_time != ', %s' % interval_key1 or info1 != 'DC (%s)' % period1 \
                        or info2 != 'TRIX (%s)' % period2 \
                        or info3 != 'Fisher (%s)' % period3:
                    chart_error = True
                    break

            #           GET DATA FROM THE CHARTS            #
            indicator_value = driver.find_elements_by_class_name('pane-legend-item-value-container')
            close = float(indicator_value[0].text.split('\n')[3].replace("종", ""))
            value1 = float(indicator_value[1].text)
            value2 = float(indicator_value[2].text.split('\n')[0])
            value3 = float(indicator_value[3].text.split('\n')[0].replace("−", "-"))

            # if offset == last_offset:
            #     print(coin_info + interval_time)
            #     print(info1, value1, end=' | ')
            #     print(info2, value2, end=' | ')
            #     print(info3, value3)
            # print(offset, end='\n')

            #           FIND LAST DATA          #
            #       COMPARE CANDLE DATA         #
            #       마지막 데이터를 찾으면 이후로는 CANDLE VALUE를 찾을 필요가 없다.      #
            if tick_change_cnt == 0:
                candle_value.append(indicator_value[0].text)

                #           MOVE CURSOR TO LAST DATA          #
                action = ActionChains(driver)
                action.move_to_element_with_offset(cursor_start_point, last_offset, 0).perform()
                #           GET CANDLE FROM THE CHARTS            #
                last_candle_value = driver.find_elements_by_class_name('pane-legend-item-value-container')[0].text
                #           시간차를 생각해서 한번 더 검사       #
                action = ActionChains(driver)
                action.move_to_element_with_offset(cursor_start_point, offset, 0).perform()
                indicator_value = driver.find_elements_by_class_name('pane-legend-item-value-container')
                candle_value.append(indicator_value[0].text)

                # print(last_candle_value)
                # print(candle_value)
                if last_candle_value not in candle_value:
                    tick_change_cnt = 1
                candle_value = list()

            #           MOVE CURSOR BACK        #
            offset -= 6
            if tick_change_cnt >= 1:
                tick_change_cnt += 1
                return_list.append(value2)
            #                   UNTIL GET LAST 4 DATA                  #
            #                  DUMP LAST UNFIXED DATA                   #
            if tick_change_cnt >= 4:
                break

        except Exception as e:
            print('Error in getting indicator value :', e)
            opener_error = True
            break

    if chart_error:
        print('Chart Configuration miss')
        return

    elif opener_error:
        print('Opener miss')
        return

    # elif value3 <= long_signal_value:
    #     print('value3 value at long position :', Coin, value3)
    #     buy_switch = 1
    #     break   # GET OUT OF THE FOR LOOP

    else:
        #           MOVE CURSOR TO LAST DATA          #
        action = ActionChains(driver)
        action.move_to_element_with_offset(cursor_start_point, last_offset, 0).perform()

    return list(reversed(return_list))


def attach_chart(driver, Coin, interval_key2):

    #   interval_time2 탭이 없을 시 첫탭 옆에 새탭 만든다.
    if len(driver.window_handles) < 3:
        driver.switch_to.window(driver.window_handles[0])
        driver.execute_script("window.open('');")
    driver.switch_to.window(driver.window_handles[2])   # 새 탭의 인덱스 번호는 생성순이니까 2번이다.
    web_chart = 'https://www.bithumb.com/trade/status/{}_KRW'.format(Coin)
    driver.get(web_chart)
    driver.implicitly_wait(3)
    iFrames = driver.find_elements_by_tag_name('iframe')
    driver.switch_to.frame(iFrames[1])
    driver.implicitly_wait(3)

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

    #           GET CANVAS          #
    # #       CANVAS CLICK이랑 같이 두면 INTERVAL 입력이 안돼는 수가 있다     #
    # while True:
    #     try:
    #         canvas = driver.find_element_by_class_name('chart-page.unselectable.on-widget')
    #         break
    #     except Exception as e:
    #         print('Error in getting canvas :', e)
    #         time.sleep(1)
    #
    # #          INTERVAL CHANGE           #
    # while True:
    #     try:
    #         canvas.click()
    #         canvas.send_keys(interval_key2)
    #         canvas.send_keys(Keys.RETURN)
    #         time.sleep(1)
    #         interval_time = driver.find_element_by_class_name('pane-legend-title__interval').text
    #         if interval_time == ', %s' % interval_key2:
    #             break
    #
    #     except Exception as e:
    #         print('Error in interval change :', e)

    #           MOVE CURSOR TO RECENT DATA          #
    action = ActionChains(driver)
    to_recent_span = driver.find_element_by_class_name('wrap-18oKCBRc-')
    action.move_to_element(to_recent_span).perform()
    time.sleep(1)

    return

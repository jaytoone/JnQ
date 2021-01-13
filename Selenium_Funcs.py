from selenium.webdriver.common.action_chains import ActionChains
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from fake_useragent import UserAgent

headless = True

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


def open_coin_list(coinvolume=7):
    #           OPEN NEW TAB FOR FINDING SIGNAL     #
    # driver.execute_script("window.open('');")
    # driver.switch_to.window(driver.window_handles[1])

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

    coin_list = driver.find_element_by_class_name('coin_list')
    coins_info = coin_list.find_elements_by_tag_name('tr')
    TopCoin = list()
    for coin_info in coins_info:
        TopCoin.append(coin_info.find_element_by_class_name('sort_coin').text.split('/')[0])
        if len(TopCoin) >= coinvolume:
            break
    print(TopCoin)

    return TopCoin
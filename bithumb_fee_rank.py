import selenium
from selenium import webdriver
import pybithumb
from bs4 import BeautifulSoup
import pandas as pd
import requests

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 2500)
pd.set_option('display.max_columns', 2500)

fee_url = 'https://www.bithumb.com/customer_support/info_fee'


path = "C:/Users/Lenovo/PycharmProjects/Project_System_Trading/Rapid_Ascend/chromedriver.exe"
# driver = webdriver.Chrome(path)

# driver.get(fee_url)
# driver.implicitly_wait(3)
page = requests.get(fee_url)
soup = BeautifulSoup(page.content, 'html.parser')

# fee_table = driver.find_element_by_class_name('g_tb_normal.fee_in_out')
fee_table = soup.find_all("table")[1]
# print(fee_table)
tab_data = [[cell.text for cell in row.find_all(["th","td"])]
                        for row in fee_table.find_all("tr")]
# coins = driver.find_elements_by_class_name('money_type.tx_c')
# # for coin in coins:
# #     print(coin.text)
# rights = driver.find_elements_by_class_name('right')
# rights_out_fee = driver.find_elements_by_class_name('right.out_fee')
# print(len(coins), len(rights), len(rights_out_fee))
fee_df = pd.DataFrame(tab_data).iloc[3:]
# fee_df = fee_df.set_index(0)
fee_df[0] = fee_df[0].apply(lambda x: x.split('(')[1].split(')')[0])
fee_df[2] = fee_df[2].apply(lambda x: float(x.replace('\n', '').replace('무료', '0').replace('-', '100')))

import numpy as np
fee_df = fee_df.iloc[:, [0, 2]].set_index(0)
fee_df.columns = ['percent_fee']
fee_df['current_fee'] = np.NaN
for i, coin in enumerate(fee_df.index):
    # print(coin)
    try:
        fee_df['current_fee'].iloc[i] = pybithumb.get_current_price(coin) * fee_df['percent_fee'].iloc[i]

    except Exception as e:
        print(e)

    print('\r %.2f %%' % (i / len(fee_df.index) * 100), end='')

print(fee_df.sort_values(by='current_fee'))
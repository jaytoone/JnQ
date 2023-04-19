from Bank.traders.shield_v2_7 import *
from multiprocessing import Process
from datetime import datetime
from pandas import to_datetime

# 0. input params.
paper_name = "wave_cci_wrr32_spread_wave_length"
main_name = os.path.basename(__file__).split('.')[0]
assert paper_name in main_name

#   a. id_zip = [utils_id_list, config_id_list]
#       i. selection_id = id_list id 와 일치시켜야될 것 (반드시 일치될 필요는 없으나 편의상 enlist_tr 에서 구별되어야하기 때문.)
id_zip = [[1], [2]]

# 1. shield time.
# shield_open = "09:00:00"      # "12:00:00"
# shield_close = "15:19:00"     # "15:19:00"
# shield_weekday = [5, 6]      # 토, 일


def shield_proc():
    shield = Shield(paper_name=paper_name, main_name=main_name, id_zip=id_zip, config_type="realtrade", shield_close=None)
    shield.run()


if __name__ == '__main__':

    while 1:
        #   a. 장 운영 시간 이외에는 프로그램을 일절 수행하지 않는다.
        #           i. login 조차 시도하지 않음. (서버 점검 시간을 고려하고 자원 낭비를 줄이기 위함임.)
        # now = datetime.now()
        # date = str(now).split(" ")[0]
        #
        # shield_open_timestamp = datetime.timestamp(to_datetime(" ".join([date, shield_open])))
        # shield_close_timestamp = datetime.timestamp(to_datetime(" ".join([date, shield_close])))
        # if now.timestamp() < shield_open_timestamp or now.timestamp() > shield_close_timestamp or now.weekday() in shield_weekday:
        #     #       ii. now 에 사용되는 자원을 줄이기 위한 term
        #     #               1. 30 min term.
        #     time.sleep(60)
        #     continue

        process = Process(target=shield_proc)
        process.start()
        process.join()
        print("connection to the server is broken.")
        # quit()

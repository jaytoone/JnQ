from funcs.binance.shield_module_v4_1 import ShieldModule
from funcs.public.broker import intmin_np
from funcs.public.constant import *

import telegram
from telegram.ext import Updater
from telegram.ext import MessageHandler, Filters

import numpy as np  # for np.nan, np.array() ...
from datetime import datetime
from pandas import to_datetime

import time
import shutil
import importlib
import os

import pickle
import logging.config
from easydict import EasyDict
import json
from ast import literal_eval


class Shield(ShieldModule):

    """
    v2_7 -> v2_7_2
        1. edit messenger phase.
        2. add severtime.
        3. add expire & open_exec validation.
        4. modify order_data -> self.over_balance (from available_balance)
        5. modify max_available_balance : * 0.98
    """

    def __init__(self, paper_name, main_name, id_zip, config_type, mode="Bank", shield_close=None):

        self.main_name = main_name
        self.utils_id_list, self.config_id_list = id_zip
        self.shield_close = shield_close

        # 1. queues for loop
        self.watch_data_dict = {}  # watch 성사 시, open_data_dict 로 전달
        self.open_data_dict = {}  # open_order 유효성 검증 후, open_remain_data_dict 로 전달
        self.open_remain_data_dict = {}  # ei_k 성사시 remove / exec 성사시 close_data_dict 로 전달
        self.close_data_dict = {}  # close_order 유효성 검증 후, close_remain_data_dict 로 전달
        self.close_remain_data_dict = {}  # out, exec 성사시 remove

        self.open_data_key = ["open_side_msg", "res_df_open", "res_df", "fee", "open_side", "limit_leverage", "trade_log_dict"]
        self.open_remain_data_key = ["open_side_msg", "res_df_open", "res_df", "fee", "open_side", "pos_side", "trade_log_dict",
                                     "tp", "ep", "out", "liqd_p", "leverage", "open_quantity", "post_order_res", "fake_order"]
        self.close_data_key = ["open_side_msg", "res_df", "fee", "open_side", "pos_side", "trade_log_dict",
                               "tp", "ep", "out", "liqd_p", "leverage", "open_exec_price_list", "open_exec_qty", "post_order_res_list", "fake_order",
                               "real_balance", "prev_exec_tp_len", "ex_dict"]
        self.close_remain_data_key = ["open_side_msg", "res_df", "fee", "open_side", "pos_side", "trade_log_dict",
                                      "tp", "ep", "out", "liqd_p", "leverage", "open_exec_price_list", "open_exec_qty", "post_order_res_list", "fake_order",
                                      "real_balance", "prev_exec_tp_len", "ex_dict",
                                      "partial_tps", "partial_qtys", "close_side", "quantity_precision"]

        # 2. set self.mode for save_data (IDEP 에서는 불가하도록.)
        self.mode = mode
        if self.mode == "Bank":
            self.pkg_path = os.path.dirname(os.getcwd())  # pkg_path 가 가리켜야할 곳은 "Bank" 임.
            os.chdir(self.pkg_path)  # log dir 저장위치를 고려해서, chdir() 실행 : 보통 JnQ -> Bank 로 전환됨.
        else:
            self.pkg_path = os.path.join(os.getcwd(), "Bank")  # IDEP 환경에서는 cwd 가 어디일지 몰라, Bank 를 기준함.
            # self.pkg_path =   # IDEP 환경에서는 cwd 가 어디일지 몰라, Bank 를 기준함.

        # 2. paper info
        #       a. public
        public_name = "Bank.papers.public.{}".format(paper_name)
        self.public = importlib.import_module(public_name)

        #       b. utils
        utils_name_list = ["Bank.papers.utils.{}_{}".format(paper_name, id_) for id_ in self.utils_id_list]
        self.utils_list = [importlib.import_module(utils_name) for utils_name in utils_name_list]

        #       c. configuration
        if config_type == "backtrade":
            self.config_name_list = ["{}_{}_backtrade.json".format(paper_name, id_) for id_ in self.config_id_list]
        elif config_type == "run_validation":
            self.config_name_list = ["{}_{}_run_validation.json".format(paper_name, id_) for id_ in self.config_id_list]
        else:
            self.config_name_list = ["{}_{}.json".format(paper_name, id_) for id_ in self.config_id_list]

        # 3. utils for individual ids
        self.config = None
        self.utils = None
        self.config_list = []

        # 4. set paper_name
        self.paper_name = paper_name

        # 5. sys_log
        self.sys_log = logging.getLogger()

        if mode == "Bank":
            #   x. define log_path
            log_path_list = ["sys_log", "trade_log", "df_log"]
            self.sys_log_path, self.trade_log_path, self.df_log_path = \
                [os.path.join(self.pkg_path, "logs", path_, self.paper_name) for path_ in log_path_list]

            for path_ in [self.sys_log_path, self.trade_log_path, self.df_log_path]:
                os.makedirs(path_, exist_ok=True)

            #   b. log_name for dict_wrapper_save_path & base_cfg_copy_path (+ sys_log file name)
            log_name = str(datetime.now().timestamp()).split(".")[0]

            #   c. dict_wrapper_save_path define
            self.dict_wrapper_save_path = os.path.join(self.trade_log_path, log_name + ".pkl")

            #   d. copy base_cfg.json -> {ticker}.json (log_cfg name)
            base_cfg_path = self.sys_log_path.replace(self.paper_name, "base_cfg.json")  # dir 와 같은 수준에 위치하는 baes_cfg.json 가져옴
            base_cfg_copy_path = os.path.join(self.sys_log_path, log_name + ".json")
            self.set_copy_base_log(log_name, base_cfg_path, base_cfg_copy_path)

        # 6. define config_path_list
        self.config_path_list = [os.path.join(self.pkg_path, "papers/config", name_) for name_ in self.config_name_list]

        # 7. we need first configuration set
        #       a. read config (refreshed by every trades)
        #               i. 기본 세팅을 위한 config 는 첫번째 config 사용
        self.read_write_config_list(mode='r')
        #       b. self.config_list loaded.
        for cfg_idx, cfg_ in enumerate(self.config_list):
            if cfg_idx == 0:
                self.config = cfg_  # 첫 running 을 위한, config 선언
            else:
                cfg_.trader_set = self.config.trader_set
        #       c. rewrite modified self.config_list
        self.read_write_config_list(mode='w', edited_config_list=self.config_list)

        # 8. Inherit + Set Messenger & Logger
        if "Bank" in mode:
            #   e. inherit
            #       i. websocket 에 self.config 의 code 를 기준으로 agg_trade 가 선언되기 때문에
            ShieldModule.__init__(self, self.config)

            #   j. add Telegram application.
            # token = "6198556911:AAFr0FUB4F0v9SyYy2LRpGzwuNQ9sHhz1iE"  # BinanceWatch
            # token = "5874553146:AAFbI49fpP4ELM0QX9qzQAAVjTRhjvfPz_4"  # StockWatch

            #       i. Telegram logger
            #               1. chat_id 는 env 동일.
            self.msg_bot = telegram.Bot(token=self.config.trader_set.token)
            self.chat_id = "5320962614"

            #       ii. Telegram messenger
            #           1. init.
            self.user_text = None

            self.updater = Updater(token=self.config.trader_set.token, use_context=True)
            dispatcher = self.updater.dispatcher

            echo_handler = MessageHandler(Filters.text & (~Filters.command), self.echo)
            dispatcher.add_handler(echo_handler)

            #           2. polling for messenger.
            self.updater.start_polling()

        # elif mode == "IDEP_Bank":
        #     #   e. inherit
        #     ShieldModule.__init__(self, self.config)
        #     CybosModule.__init__(self)

        # 9. balance
        self.available_balance = self.config.trader_set.initial_asset
        self.over_balance = None
        self.min_balance = 5.0  # USDT, not used in Stock.

        # 10. pnl
        self.income = 0.0
        self.accumulated_income = 0.0

        if self.config.trader_set.profit_mode == "SUM":
            self.accumulated_profit = 0.0
            self.ideal_accumulated_profit = 0.0
        else:
            self.accumulated_profit = 1.0  # if profit_mode == "SUM", will be changed to zero
            self.ideal_accumulated_profit = 1.0  # if profit_mode == "SUM", will be changed to zero

        # 11. posmode_margin_leverage deprecated in Stock.
        #       a. set fixed value for Stock
        # self.limit_leverage = 1
        # self.limit_leverage = self.posmode_margin_leverage()

        # 12. trade_log_dict 에 저장되는 데이터의 chunk_size 는 1 거래완료를 기준으로한다. (dict 로 운영할시 key 값 중복으로 인해 데이터 상쇄가 발생함.)
        self.trade_log_dict_wrapper = []

        # 13. set prev_w_i for index restoration after exit by loop_duration.
        self.prev_w_i = -1
        self.allow_watch = 1

    def kill_proc(self):
        self.updater.stop()
        quit()

    def echo(self, update, context):
        self.user_text = update.message.text

    def read_write_config_list(self, mode='r', edited_config_list=None):
        try:
            config_file_list = [open(cfg_path, mode) for cfg_path in self.config_path_list]

            if mode == 'r':
                self.config_list = [EasyDict(json.load(cfg_)) for cfg_ in config_file_list]
            elif mode == 'w':
                assert edited_config_list is not None, "assert edited_config_list is not None"
                _ = [json.dump(config_, config_file_, indent=2) for config_, config_file_ in zip(edited_config_list, config_file_list)]
            else:
                assert mode in ['r', 'w'], "assert mode in ['r', 'w']"

            # 1. opened files should be closed --> 닫지 않으면 reopen 시 error occurs#
            _ = [config_file_.close() for config_file_ in config_file_list]

        except Exception as e:
            msg = "error in read_write_config_list : {}".format(e)
            self.sys_log.error(msg)
            self.msg_bot.sendMessage(chat_id=self.chat_id, text=msg)
            self.kill_proc()

    def set_copy_base_log(self, log_name, base_cfg_path, base_cfg_copy_path):

        """
         set logger info. to {ticker}.json - offer realtime modification
         1. base_cfg_copy_path 기준으로 변경 내용 .log 로 적용
        """
        try:
            shutil.copy(base_cfg_path, base_cfg_copy_path)

            with open(base_cfg_copy_path, 'r') as sys_cfg:
                sys_log_cfg = EasyDict(json.load(sys_cfg))

            # 1. edit 내용
            sys_log_cfg.handlers.file_rot.filename = \
                os.path.join(self.sys_log_path, log_name + ".log")  # log_file name - log_name 에 종속
            logging.getLogger("apscheduler.executors.default").propagate = False

            with open(base_cfg_copy_path, 'w') as edited_cfg:
                json.dump(sys_log_cfg, edited_cfg, indent=3)

            logging.config.dictConfig(sys_log_cfg)  # 선언 순서 상관 없음.
            self.sys_log.info('# ----------- {} ----------- #'.format(self.paper_name))
            self.sys_log.info("pkg_path : {}".format(self.pkg_path))

        except Exception as e:
            print("error in load sys_log_cfg :", e)
            self.kill_proc()

    def save_data(self, re_id=""):

        if self.mode == "Bank":
            # 1. self.open / close dumping 때문에, for loop 로 하지 않음.
            data_dir_path = os.path.join(self.pkg_path, "data/{}".format(self.main_name))
            os.makedirs(data_dir_path, exist_ok=True)

            try:
                save_path = os.path.join(data_dir_path, "watch_data_dict{}.pkl".format(re_id))

                with open(save_path, 'wb') as f:
                    pickle.dump(self.watch_data_dict, f)
                    self.sys_log.warning("{} saved".format(save_path))
            except Exception as e:
                msg = "error in save_data : {}".format(e)
                self.sys_log.error(msg)
                self.msg_bot.sendMessage(chat_id=self.chat_id, text=msg)
            try:
                save_path = os.path.join(data_dir_path, "open_remain_data_dict{}.pkl".format(re_id))

                with open(save_path, 'wb') as f:
                    pickle.dump(self.open_remain_data_dict, f)
                    self.sys_log.warning("{} saved".format(save_path))
            except Exception as e:
                msg = "error in save_data : {}".format(e)
                self.sys_log.error(msg)
                self.msg_bot.sendMessage(chat_id=self.chat_id, text=msg)
            try:
                save_path = os.path.join(data_dir_path, "close_remain_data_dict{}.pkl".format(re_id))
                with open(save_path, 'wb') as f:
                    pickle.dump(self.close_remain_data_dict, f)
                    self.sys_log.warning("{} saved\n".format(save_path))
            except Exception as e:
                msg = "error in save_data : {}\n".format(e)
                self.sys_log.error(msg)
                self.msg_bot.sendMessage(chat_id=self.chat_id, text=msg)

    def load_data(self):

        #       x. make replication.
        self.save_data(re_id=str(datetime.now().timestamp()).split(".")[0])

        watch_code_dict = {}

        try:
            data_path = os.path.join(self.pkg_path, "data/{}/watch_data_dict.pkl".format(self.main_name))
            with open(data_path, 'rb') as f:
                self.watch_data_dict = pickle.load(f)
                for code, watch_data in self.watch_data_dict.copy().items():

                    #       a. websocket 과 watch_dict 에 추가되는 데이터가 watch_code_dict 를 기준하기 때문에 아래 logic 는 존재해야한다.
                    watch_code_dict[code] = watch_data['open_side_msg']
                    self.sys_log.warning("{} {} added to watch_code_dict. (watch_data_dict)".format(code, watch_data["open_side_msg"]))

        except Exception as e:
            msg = "error in load watch_data_dict : {}".format(e)
            self.sys_log.error(msg)
            self.msg_bot.sendMessage(chat_id=self.chat_id, text=msg)

        try:
            data_path = os.path.join(self.pkg_path, "data/{}/open_remain_data_dict.pkl".format(self.main_name))
            with open(data_path, 'rb') as f:
                self.open_remain_data_dict = pickle.load(f)
                for code, open_remain_data in self.open_remain_data_dict.copy().items():

                    watch_code_dict[code] = open_remain_data['open_side_msg']
                    self.sys_log.warning("{} {} added to watch_code_dict. (open_remain_data_dict)".format(code, open_remain_data["open_side_msg"]))

        except Exception as e:
            msg = "error in load open_remain_data_dict : {}".format(e)
            self.sys_log.error(msg)
            self.msg_bot.sendMessage(chat_id=self.chat_id, text=msg)

        try:
            data_path = os.path.join(self.pkg_path, "data/{}/close_remain_data_dict.pkl".format(self.main_name))
            with open(data_path, 'rb') as f:
                #   1. self.~remain_data_queue 를 정의하고.
                self.close_remain_data_dict = pickle.load(f)
                for code, close_remain_data in self.close_remain_data_dict.copy().items():

                    #       d. 존재하는 미체결내역에 대해 get_new_df 를 위한 add to watch_code_dict.
                    watch_code_dict[code] = close_remain_data['open_side_msg']
                    self.sys_log.warning("{} {} added to watch_code_dict (close_remain_data_dict).".format(code, close_remain_data["open_side_msg"]))

        except Exception as e:
            msg = "error in load close_remain_data_dict : {}".format(e)
            self.sys_log.error(msg)
            self.msg_bot.sendMessage(chat_id=self.chat_id, text=msg)

        return watch_code_dict

    def get_balance_info(self, min_balance, mode="PROD"):

        min_balance_bool = 0

        # 1. asset_change - 첫 거래여부와 무관하게 진행
        if self.config.trader_set.asset_changed:
            self.available_balance = self.config.trader_set.initial_asset
            #        1. 이런식으로, cfg 의 값을 변경(dump)하는 경우를 조심해야함 - config_list[0] 에 입력해야한다는 의미
            self.config_list[0].trader_set.asset_changed = 0
            with open(self.config_path_list[0], 'w') as cfg:
                json.dump(self.config_list[0], cfg, indent=2)

            self.sys_log.info("asset_changed 1 --> 0")

        # 2. get availableBalance
        if not self.config.trader_set.backtrade:
            max_available_balance = self.get_available_balance()
            self.sys_log.info('max_available_balance : {:.2f}'.format(max_available_balance))

            # a. over_balance 가 저장되어 있다면, 현재 사용하는 balance 가 원하는 것보다 작은 상태임.
            #       i. 최대 허용 가능 balance 인 max_balance 와의 지속적인 비교 진행
            if self.over_balance is not None:
                if self.over_balance <= max_available_balance * 0.98:  # 정상화 가능한 상태 (over_balanced 된 양을 허용할 수준의 max_balance)
                    self.available_balance = self.over_balance  # mode="SUM" 의 경우에도 self.over_balance 에는 initial_asset 만큼만 담길 수 있기 때문에 구분하지 않음.
                    self.over_balance = None
                else:  # 상태가 어찌되었든 일단은 지속해서 Bank 를 돌리려는 상황. (후조치 한다는 의미, 중단하는게 아니라)
                    self.available_balance = max_available_balance * 0.98  # max_available_balance 를 넘지 않는 선
                self.sys_log.info('available_balance (temp) : {:.2f}'.format(self.available_balance))

            # b. 예기치 못한 오류로 인해 over_balance 상태가 되었을때의 조치
            else:
                if self.available_balance >= max_available_balance:
                    self.over_balance = self.available_balance  # 복원 가능하도록 현재 상태를 over_balance 에 저장
                    self.available_balance = max_available_balance * 0.98
                self.sys_log.info('available_balance : {:.2f}'.format(self.available_balance))

            # c. min_balance check
            if self.available_balance < min_balance:
                self.sys_log.warning('available_balance {:.2f} < min_balance {:.2f}\n'.format(self.available_balance, min_balance))
                min_balance_bool = 1

            # d. API term
            time.sleep(self.config.trader_set.api_term)

        return min_balance_bool

    def get_income_info(self, real_balance, leverage, ideal_profit, real_profit, mode="PROD", currency="USDT"):  # 복수 pos 를 허용하기 위한 api 미사용

        """
        v2 -> _
            1. 단리 / 복리 mode 에 따라 출력값 변경하도록 구성함.
            2. Bank 내부 메소드는 version 을 따로 명명하지 않도록함. -> Bank version 에 귀속되도록하기 위해서.
        """

        ideal_profit_pct = ideal_profit - 1
        real_profit_pct = real_profit - 1
        self.income = real_balance * real_profit_pct
        self.accumulated_income += self.income

        if mode == "PROD":
            self.available_balance += self.income

        # 1. use precision 5 for checking detail profit calculation
        self.sys_log.info("ideal_profit : {:.5%}".format(ideal_profit_pct))
        self.sys_log.info("real_profit : {:.5%}".format(real_profit_pct))

        ideal_tmp_profit = ideal_profit_pct * leverage
        real_tmp_profit = real_profit_pct * leverage

        self.sys_log.info("income : {:.2f} {}".format(self.income, currency))
        self.sys_log.info("temporary profit : {:.2%} ({:.2%})".format(real_tmp_profit, ideal_tmp_profit))

        if mode == "PROD":
            self.accumulated_profit *= 1 + real_tmp_profit
            self.ideal_accumulated_profit *= 1 + ideal_tmp_profit
            self.sys_log.info("accumulated profit : {:.2%} ({:.2%})".format(self.accumulated_profit - 1, self.ideal_accumulated_profit - 1))
        else:
            self.accumulated_profit += real_tmp_profit
            self.ideal_accumulated_profit += ideal_tmp_profit
            self.sys_log.info("accumulated profit : {:.2%} ({:.2%})".format(self.accumulated_profit, self.ideal_accumulated_profit))

        msg = "accumulated income : {:.2f} {}\n".format(self.accumulated_income, currency)
        self.sys_log.info(msg)
        self.msg_bot.sendMessage(chat_id=self.chat_id, text=msg)

    def run(self):

        """
        ONE LOOP
        """

        #   x. 서버 연결 종료후 프로그램 재시작, dynamic_watch 의 상황에서 remain_loop 를 위한 code 추가.
        watch_code_dict = self.load_data()

        while 1:

            # 0. program error 인지를 위한 try ~ except 구문.
            #       a. error 발생시, remain_data 저장하기 위함.
            #       b. 장 마감시 프로그램 종료.
            now = datetime.now()
            date = str(now).split(" ")[0]
            # minute = int(str(now).split(":")[-2])

            if self.shield_close is not None:
                if now.timestamp() > datetime.timestamp(to_datetime(" ".join([date, self.shield_close]))):
                    self.sys_log.warning("Bank closed.")
                    self.kill_proc()

            """
            ADD WATCH
            """

            open_side_msg = None
            if not self.config.trader_set.backtrade:

                #   1. 특성상 code 를 맨 앞에 붙일 것.
                #       i. prev_w_i reset 해주어 앞의 code 부터 순회하게 됨.
                #       ii. save 에 대비해 먼저 이전 data load.

                #   2. get code (msg form : "code open_side_msg")
                if self.user_text is not None:

                    #   a. Telegram Messenger mode : on / off 도입.
                    #       i. default msg = invalid payload.
                    msg = "error in self.user_text : invalid payload."

                    if self.config.trader_set.messenger_on:
                        payload = self.user_text.upper().split(" ")

                        if len(payload) == 2:
                            code, open_side_msg = payload

                            with open(r"D:\Projects\System_Trading\JnQ\Bank\valid_code\ticker_in_futures.pkl", 'rb') as f:
                                valid_code_list = pickle.load(f)

                            #   i. code validation
                            if code in valid_code_list and open_side_msg in [OrderSide.BUY, OrderSide.SELL]:
                                watch_code_dict[code] = open_side_msg
                                msg = "{} {} added to watch_code_dict.".format(code, open_side_msg)

                            #   ii. remove code.
                            elif "rm " in self.user_text.lower():
                                code = self.user_text.split(" ")[1].upper()
                                if code not in self.open_remain_data_dict and code not in self.close_remain_data_dict:
                                    self.watch_data_dict.pop(code, None)
                                    msg = "watch : {}".format(list(self.watch_data_dict.keys()))
                                else:
                                    msg = "{} in remain_dict.".format(code)

                    #   b. watch
                    if "watch" in self.user_text.lower():
                        msg = "watch : {}".format(list(self.watch_data_dict.keys()))
                        for code, open_remain_data in self.open_remain_data_dict.items():
                            msg += "\nopen_remain : {}, tp : {}, ep : {}, out : {}".format(code, open_remain_data['tp'], open_remain_data['ep'], open_remain_data['out'])
                        for code, close_remain_data in self.close_remain_data_dict.items():
                            msg += "\nclose_remain : {}, tp : {}, ep : {}, out : {}".format(code, close_remain_data['tp'], close_remain_data['ep'], close_remain_data['out'])

                    self.msg_bot.sendMessage(chat_id=self.chat_id, text=msg)

                    #   d. 사용된 user_text -> None 으로 치환한다.
                    self.user_text = None

                #   3. Messenger off 시, default code = self.config.trader_set.symbol
                else:
                    if not self.config.trader_set.messenger_on:
                        if len(self.watch_data_dict) == 0:
                            code = self.config.trader_set.symbol
                            watch_code_dict[code] = open_side_msg

                #   4. add data to watch_data_dict.
                for code, open_side_msg in watch_code_dict.items():
                    #   i. add websocket.
                    self.websocket_client.agg_trade(symbol=code, id=1, callback=self.agg_trade_message_handler)
                    #   ii. set posmode_margin_leverage.
                    limit_leverage = self.posmode_margin_leverage(code)
                    self.watch_data_dict[code] = {"open_side_msg": open_side_msg, "limit_leverage": limit_leverage, "ep_loc_point2": 0, "streamer": None}

                #   5. reset watch_code_dict. --> watch_data_dict 에 data 지속적으로 추가하는 것을 방지하기 위함.
                watch_code_dict = {}

            else:
                if len(self.watch_data_dict) == 0:
                    #   d. backtrade just for a code (get code from back_data_path)
                    code = self.config.trader_set.symbol
                    #       i. bank_module 내부에 streamer 선언은 loop 이탈하면 초기화되서, Bank 의 streamer 로 유지함.
                    #       ii. default limit_leverage = 50.
                    self.watch_data_dict[code] = {"open_side_msg": None, "limit_leverage": 50, "ep_loc_point2": 0, "streamer": self.get_streamer()}

            """
            WATCH LOOP
            """

            start_time = time.time()

            # 0.  필요없는 API 사용을 지양하기 위해 time_term 도입. : watch_loop ban_term.
            #   a. 최소 시간동안 watch loop 에 들어오지 못하도록 해야한다.
            #       i. 최소시간 = 20 seconds --> 임의 설정.
            #       ii. 초 단위 < 2 로 고정. (Binance)
            if not self.config.trader_set.backtrade:
                try:
                    if time.time() - end_time_watch >= 20 and now.second < 2:
                        self.allow_watch = 1
                        #   ii. for save_data() term.
                        self.save_data()
                    else:
                        """
                        #   iii. time.sleep() 하지 않을시, 하드웨어 팬 소리 커진다.
                        #       1. 단순 while loop 의 자원 소모가 심하다는 이야기.
                        """
                        time.sleep(self.config.trader_set.realtime_term)
                except:
                    pass

            if self.allow_watch:

                for w_i, (code, watch_data) in enumerate(self.watch_data_dict.copy().items()):

                    if not self.config.trader_set.backtrade:
                        # y. set start w_i : loop_limit 으로 인해 이탈 후 복귀시의 start w_i 설정.
                        #       i. 시작 w_i 를 변경하는건, static_watch 만 해당됨. (dynamic 은 불필요함.) -> static 기준이 아니라, duplicate 기준임.
                        #       ii. dynamic update 안하는 경우 = static 과 동일.
                        if w_i <= self.prev_w_i:
                            continue
                        else:
                            #           2. w_i, last_index 도달시 reset.
                            if w_i == len(self.watch_data_dict) - 1:
                                self.prev_w_i = -1
                            else:
                                self.prev_w_i = w_i

                        # x. loop 최대 잔류 시간.
                        # if not self.config.trader_set.backtrade:  # 중복됨.
                        if time.time() - start_time > self.config.trader_set.loop_duration:
                            break

                    open_side_msg, limit_leverage, ep_loc_point2, streamer = watch_data.values()
                    self.sys_log.warning("WATCH : {} {}".format(code, open_side_msg))

                    # 0. param init.
                    # open_side = None
                    self.utils = None

                    # 1. get_new_df
                    if self.config.trader_set.backtrade:
                        res_df = self.get_new_df_onstream(streamer)
                    else:
                        res_df = self.get_new_df(code)
                    self.sys_log.info("WATCH : ~ get_new_df : %.2f" % (time.time() - start_time))

                    # 2. remote update res_df in open & close remain_loop (open / close skip 될 수 있기 때문에.)
                    duplicated = 0
                    for or_code, open_remain_data in self.open_remain_data_dict.copy().items():
                        if code == or_code:
                            #   i. update res_df
                            self.open_remain_data_dict[code]["res_df"] = res_df
                            duplicated = 1
                            break
                    for cr_code, close_remain_data in self.close_remain_data_dict.copy().items():
                        if code == cr_code:
                            #   ii. update res_df
                            self.close_remain_data_dict[code]["res_df"] = res_df
                            duplicated = 1
                            break
                    #       y. ban code duplication (temporarily) => 추후에 종목 중복 가능함.
                    if duplicated:
                        # if not self.config.trader_set.backtrade:
                        #     time.sleep(self.config.trader_set.api_term)
                        continue

                    # 2. paper works
                    try:
                        res_df = self.public.sync_check(res_df, self.config)  # function usage format maintenance
                        self.sys_log.info('WATCH : ~ sync_check : %.5f' % (time.time() - start_time))

                        np_timeidx = np.array([intmin_np(date_) for date_ in res_df.index.to_numpy()])

                        res_df = self.public.public_indi(res_df, self.config, np_timeidx)
                        self.sys_log.info('WATCH : ~ public_indi : %.5f' % (time.time() - start_time))

                        #   a. use_point2 사용시, 해당 ID 로만 enlist_ 진행
                        if self.utils is not None and ep_loc_point2:
                            res_df = self.utils.enlist_tr(res_df, self.config, np_timeidx, show_detail=self.config.trader_set.show_detail)
                        else:
                            for utils_, config_ in zip(self.utils_list, self.config_list):
                                res_df = utils_.enlist_tr(res_df, config_, np_timeidx, show_detail=config_.trader_set.show_detail)
                        self.sys_log.info('WATCH : ~ enlist_tr : %.5f' % (time.time() - start_time))

                        if not self.config.trader_set.backtrade:
                            self.sys_log.info('check res_df.index[-1] : {}'.format(res_df.index[-1]))
                        else:
                            self.sys_log.info('check res_df.index[-1] : {}'.format(res_df.index[-1]))

                    except Exception as e:
                        msg = "error in self.utils_ : {}".format(e)
                        self.sys_log.error(msg)
                        self.msg_bot.sendMessage(chat_id=self.chat_id, text=msg)
                        # time.sleep(self.config.trader_set.api_term)
                        continue

                    # 3. signal check
                    # if open_side is None:

                    #   a. ep_loc - get_open_side_v2
                    #       i. open_signal 포착한 utils & config type 으로 bank.utils & config 변경함.
                    try:
                        papers = (self.public, self.utils_list, self.config_list)
                        open_side, self.utils, self.config = self.get_open_side_v2(res_df, papers, np_timeidx, open_num=1)

                    except Exception as e:
                        msg = "error in ep_loc phase : {}".format(e)
                        self.sys_log.error(msg)
                        self.msg_bot.sendMessage(chat_id=self.chat_id, text=msg)
                        # time.sleep(self.config.trader_set.api_term)
                        continue

                    #   b. df logging
                    if self.config.trader_set.df_log:  # save res_df at ep_loc
                        excel_name = str(datetime.now()).replace(":", "").split(".")[0]
                        # res_df.reset_index().to_feather(os.path.join(self.df_log_path, "{}.ftr".format(excel_name)), compression='lz4')
                        res_df.reset_index().to_excel(os.path.join(self.df_log_path, "{}.xlsx".format(excel_name)))

                    #   c. open_signal 발생의 경우
                    if open_side is not None:

                        #       x. check open_side_msg.
                        if open_side_msg is not None:
                            if open_side != open_side_msg:
                                continue

                        #       i. res_df_open 정의는 이곳에서 해야함 - 아래서 할 경우 res_df_open 갱신되는 문제
                        res_df_open = res_df.copy()  # point2 에 사용됨.

                        #       ii. market_check_term for market_entry - deprecated
                        # #             1. 바로 진행할 경우, data loading delay 로 인한 오류 방지
                        # if self.config.ep_set.entry_type == "MARKET" and not self.config.trader_set.backtrade:
                        #     market_check_term = datetime.now().second
                        #     if market_check_term > self.config.trader_set.market_check_term:
                        #         open_side = None
                        #         self.sys_log.warning("market_check_term : {}".format(market_check_term))
                        #         continue  # ep_loc_check = False 라서 open_side None phase 로 감 -> 무슨 의미 ? 어쨌든 wait_zone 으로 회귀

                        #       iv. init fee
                        if self.config.ep_set.entry_type == "MARKET":
                            fee = self.config.trader_set.market_fee
                        else:
                            fee = self.config.trader_set.limit_fee

                        #       z. queue result
                        open_data_value = [open_side_msg, res_df_open, res_df, fee, open_side, limit_leverage, {}]
                        self.open_data_dict[code] = dict(zip(self.open_data_key, open_data_value))

                    # z. queue result
                    #       a. watch_queue 는 append 되는 것이 아님. => update 되는 것. (누적에 대해 염려할 필요가 없다는 이야기)
                    #       b. dynamic_watch 의 watch_queue update 이전은 static 상태와 동일함.
                    #               i. update 시에 remain_data 에 있는 code 가 유지될 수 있게만 설정하면 됨. => "dynamic 과 무관하게" 서버 다운에 상황에 대한 대응법이기도 함.

                    # 4. point2, ei_k
                    #        i. 매회 새로운 res_df 가 필요함 (ep_loc_point2 를 위해서), 동시에 ei_k 를 위해선 open 기준의 res_df 가 필요함.
                    # if self.config.loc_set.point2.use_point2:
                    #     ep_loc_point2 = 1
                    #     self.sys_log.warning("selection_id use_point2 : {}{}".format(selection_id, self.config.loc_set.point2.use_point2))
                    #
                    #     #        a. tp_j, res_df_open 으로 고정
                    #     c_i = self.config.trader_set.complete_index
                    #     expired = check_ei_k_onbarclose_v2(self, res_df_open, res_df, c_i, c_i, open_side)   # e_j, tp_j
                    #     #        b. e_j 에 관한 고찰 필요함, backtest 에는 i + 1 부터 시작하니 +1 하는게 맞을 것으로 봄
                    #     #          -> 좀더 명확하게, dc_lower_1m.iloc[i - 1] 에 last_index 가 할당되는게 맞아서
                    #     allow_ep_in, _ = self.public.ep_loc_point2_v2(res_df, self.config, c_i + 1, out_j=None, side=open_side)   # e_j
                    #     if allow_ep_in:
                    #         break
                    # else:   # point2 미사용시 바로 order phase 로 break
                    #     break
                    # break

                    self.sys_log.info('WATCH : ~ signal_check : %.5f\n' % (time.time() - start_time))

                    # 6. API term for consecutive request. (just for realtrade)
                    # if not self.config.trader_set.backtrade:
                    #     time.sleep(self.config.trader_set.api_term)

                if not self.config.trader_set.backtrade:
                    self.allow_watch = 0
                end_time_watch = time.time()

            """
            OPEN
            """
            start_time = time.time()

            for code, open_data in self.open_data_dict.copy().items():

                # y. data water_fall
                open_side_msg, res_df_open, res_df, fee, open_side, limit_leverage, trade_log_dict = open_data.values()
                self.sys_log.warning("OPEN : {} {}".format(code, open_side_msg))

                # 0. param init.
                fake_order = 0  # order 하는 것처럼 시간을 소비하기 위함임.

                # 1. get tr_set, leverage
                #       a. 일단은, ep1 default 로 설정
                tp, ep, out, open_side = self.get_tpepout(open_side, res_df_open, res_df)

                # 2. get balance
                #       a. Stock 특성상, min_balance = ep 로 설정함.
                #       b. open_data_dict.pop 은 watch_dict 의 일회성 / 영구성과 무관하다.
                try:
                    min_balance_bool = self.get_balance_info(self.min_balance, mode=self.config.trader_set.profit_mode)
                    if min_balance_bool:
                        #   i. static_watch 에서는 pop 하면, code 가 복원이 안됨.
                        # self.watch_data_dict.pop(code)
                        self.open_data_dict.pop(code)
                        continue
                except Exception as e:
                    msg = "error in get_balance_info : {}".format(e)
                    self.sys_log.error(msg)
                    self.msg_bot.sendMessage(chat_id=self.chat_id, text=msg)
                    self.open_data_dict.pop(code)
                    continue

                self.sys_log.info('OPEN : ~ get balance : %.5f' % (time.time() - start_time))

                #       a. set pos_side & fake_order & open_price comparison
                open_price = res_df['open'].to_numpy()[-1]  # open 은 latest_index 의 open 사용
                if open_side == OrderSide.BUY:
                    pos_side = PositionSide.LONG
                    if self.config.pos_set.long_fake:
                        fake_order = 1
                    ep = min(open_price, ep)
                else:
                    pos_side = PositionSide.SHORT
                    if self.config.pos_set.short_fake:
                        fake_order = 1
                    ep = max(open_price, ep)

                leverage, liqd_p = self.public.lvrg_liqd_set_v2(res_df, self.config, open_side, ep, out, fee, limit_leverage)

                #       b. get precision
                price_precision, quantity_precision = self.get_precision(code)
                tp, ep, out, liqd_p = [self.calc_with_precision(price_, price_precision) for price_ in [tp, ep, out, liqd_p]]  # includes half-dynamic tp

                self.sys_log.info('tp : {}'.format(tp))
                self.sys_log.info('ep : {}'.format(ep))
                self.sys_log.info('out : {}'.format(out))
                self.sys_log.info('liqd_p : {}'.format(liqd_p))
                self.sys_log.info('leverage : {}'.format(leverage))
                self.sys_log.info('available_balance : {}'.format(self.available_balance))
                self.sys_log.info('OPEN : ~ tp ep out leverage set : %.5f' % (time.time() - start_time))

                if leverage is None:
                    self.sys_log.info("leverage_rejection occured.")
                    continue

                if not self.config.trader_set.backtrade:
                    while 1:
                        try:
                            server_time = self.time()['serverTime']
                            self.change_leverage(symbol=code, leverage=leverage, recvWindow=6000, timestamp=server_time)
                        except Exception as e:
                            msg = 'error in change_initial_leverage : {}'.format(e)
                            self.sys_log.error(msg)
                            self.msg_bot.sendMessage(chat_id=self.chat_id, text=msg)
                            time.sleep(self.config.trader_set.api_term)
                            continue  # -->  ep market 인 경우에 조심해야함 - why ..?
                        else:
                            self.sys_log.info('leverage changed --> {}'.format(leverage))
                            break

                #       c. calc. open_quantity
                open_quantity = self.calc_with_precision(self.available_balance / ep * leverage, quantity_precision)
                self.sys_log.info("open_quantity : {}".format(open_quantity))

                # 3. open order
                error_code = 0
                if not self.config.trader_set.backtrade and not fake_order:
                    # [ API ]
                    order_data = (self.over_balance, leverage)
                    post_order_res, self.over_balance, error_code = self.limit_order_v2(code, self.config.ep_set.entry_type, open_side, pos_side, ep, open_quantity, order_data)

                    #   a. order deny exception, error_code = 0 : 일단은 error 발생시 continue.
                    if error_code:
                        time.sleep(self.config.trader_set.order_term)
                        # break  # 기존, trade_v1_5 에서는 break 해버림.
                else:
                    #   b. we don't use post_order_res in backtrade.
                    post_order_res = None

                if not error_code:
                    # z. queue result
                    #       i. add new order for preventing invalid close_order (popping old open_remain_data)
                    #       ii. 유효성이 검증된 open_data 는 지워줄 것.
                    open_remain_data_value = [open_side_msg, res_df_open, res_df, fee, open_side, pos_side, trade_log_dict, tp, ep, out, liqd_p, leverage, open_quantity, post_order_res, fake_order]
                    self.open_remain_data_dict[code] = dict(zip(self.open_remain_data_key, open_remain_data_value))
                    self.open_data_dict.pop(code)

                    self.msg_bot.sendMessage(chat_id=self.chat_id, text="{} {} open_order enlisted.".format(code, open_side))

                self.sys_log.info('OPEN : ~ open_order : %.5f\n' % (time.time() - start_time))

            """
            OPEN REMAIN LOOP
            """
            start_time = time.time()

            for code, open_remain_data in self.open_remain_data_dict.copy().items():

                # x. loop 최대 잔류 시간.
                #       i. open, close remain loop 는 내부에서 API 사용하지 않아 최대 잔류 시간이 의미 없어짐 (loop 내 잔류 시간이 적을 것으로 예상하기 때문임.)
                # if not self.config.trader_set.backtrade:
                #     if time.time() - start_time > self.config.trader_set.loop_duration:
                #         break

                # y. data water_fall
                open_side_msg, res_df_open, res_df, fee, open_side, pos_side, trade_log_dict, tp, ep, out, liqd_p, leverage, open_quantity, post_order_res, fake_order = open_remain_data.values()
                # self.sys_log.warning("OPEN_REMAIN : {} {}".format(code, open_side_msg))

                # 0. param init.
                #       a. expired 가 없는 경우를 위한 init.
                expired = 0

                # 1. OPEN REMAIN LOOP 를 위한 data.
                #       a. .loc 을 post_order_res 로 접근하면 order_info_valid type 이 Series 가 됨. => [post_order_res] 를 통해 DataFrame 으로 촐력한다.
                # if not self.config.trader_set.backtrade:
                #     #   b. open 후에 order_info 는 update 된 상태가 아님, 따라서 .loc 접근시 key_error 가 한번은 발생함.
                #     #   [ API ] : open_order_info_valid
                #     try:
                #         open_order_info_valid = order_info.loc[[post_order_res]]
                #     except Exception as e:
                #           msg = "error in order_info.loc[post_order_res_list], open_remain_loop : {}".format(e)
                #           self.sys_log.error(msg)
                #           self.msg_bot.sendMessage(chat_id=self.chat_id, text=msg)
                #         #       i. continue 는 retry 가 의미없을 경우 pop 함.
                #         continue

                # 1. market
                open_exec = 0
                if self.config.ep_set.entry_type == OrderType.MARKET:
                    open_exec = 1

                # 2. limit
                else:
                    if not self.config.trader_set.backtrade:
                        #   a. type validation => [0] 으로 접근하는 이유는 len(post_order_res) = 1, info_valid's type = DataFrame, return int.
                        #   [ API ] : order_info
                        realtime_price = self.get_market_price_v3(code)

                        #   a. ei_k
                        #       ii. expire_tp default
                        expired = self.check_ei_k_v2(res_df_open, res_df, realtime_price, open_side)

                        #   b. open_execution
                        if fake_order:
                            if open_side == OrderSide.BUY:
                                if realtime_price <= ep:
                                    open_exec = 1
                            else:
                                if realtime_price >= ep:
                                    open_exec = 1
                        else:
                            open_exec = self.check_open_exec_qty_v2(post_order_res, open_quantity)
                    else:
                        c_i = self.config.trader_set.complete_index  # 단지, 길이 줄이기 위해 c_i 로 재선언하는 것뿐, phase 가 많지 않아 수정하지 않음.

                        #   a. ei_k
                        #       i. expire_tp default
                        expired = self.check_ei_k_onbarclose_v2(res_df_open, res_df, c_i, c_i, open_side)  # e_j, tp_j

                        # b. open_execution
                        if open_side == OrderSide.BUY:
                            if res_df['low'].to_numpy()[c_i] <= ep:
                                open_exec = 1
                        else:
                            if res_df['high'].to_numpy()[c_i] >= ep:
                                open_exec = 1

                # self.sys_log.info('OPEN_REMAIN : ~ ei_k : %.5f' % (time.time() - start_time))

                if expired or open_exec:

                    #   c. when open order time expired or executed, regardless to position exist, cancel open orders       #
                    if not self.config.trader_set.backtrade and not fake_order:
                        # [ API ]
                        while 1:
                            open_exec_price_list, open_exec_qty = self.cancel_order_list([post_order_res], self.config.ep_set.entry_type)
                            if expired or open_exec_qty != 0:
                                break
                            else:
                                time.sleep(self.config.trader_set.api_term)
                    else:
                        open_exec_price_list = [ep]
                        open_exec_qty = open_quantity if open_exec else 0

                    #       1. expire case 의 open_exec_qty != 0 인 경우, close_order 실행함.
                    if open_exec or open_exec_qty != 0:
                        msg = "{} open_order executed.".format(code)
                        self.sys_log.info(msg)
                        self.msg_bot.sendMessage(chat_id=self.chat_id, text=msg)
                        self.sys_log.info("open_exec_price_list : {}".format(open_exec_price_list))

                        # d. just for calculating profit.
                        real_balance = open_exec_price_list[0] * open_exec_qty
                        self.sys_log.info("real_balance : {}".format(real_balance))

                        # e. add trade_log_dict open info.
                        #       i. backtrade's entry timestamp 의 경우, lastest_index 가 complete_index 로. (위에서 c_i 를 사용함)
                        open_idx_str = str(res_df_open.index[self.config.trader_set.complete_index])
                        trade_log_dict["open"] = [open_idx_str, open_side]

                        if not self.config.trader_set.backtrade:
                            date_str = str(datetime.now())
                            trade_log_dict["entry"] = [date_str.replace(date_str.split(':')[-1], open_idx_str.split(':')[-1]), open_side, ep]  # 초 단위 맞춰주기.
                        else:
                            #   ii. 불가피하게 backtrade 의 경우, 첫 execution check 에 res_df_open, res_df 가 같이 내려오기 때문에 market 으로 체결되는 경우 timestamp 이 open 과 동일해짐.
                            trade_log_dict["entry"] = [str(res_df.index[self.config.trader_set.complete_index]), open_side, ep]

                        # f. keep order 를 위해 이곳에서 선언함.
                        post_order_res_list = [None] * len(literal_eval(self.config.tp_set.partial_ranges))

                        # z. queue result
                        #       i. executed open_order 가 old_order (매칭되는 close_order 가 존재하는 경우.) 가 아니라면, close_order.
                        #       ii. new / old order 상관없이 미체결 open 의 체결 이후는 close 하는게 맞음.
                        #       iii. add 0, {} for 누적형 parameter : prev_exec_tp_len, ex_dict
                        close_data_value = [open_side_msg, res_df, fee, open_side, pos_side, trade_log_dict,
                                            tp, ep, out, liqd_p, leverage, open_exec_price_list, open_exec_qty, post_order_res_list, fake_order,
                                            real_balance, 0, {}]
                        self.close_data_dict[code] = dict(zip(self.close_data_key, close_data_value))

                    # z. expired, open_exec 모두 pop 을 수행함.
                    if expired:
                        msg = "{} open_order expired.".format(code)
                        self.sys_log.info(msg)
                        self.msg_bot.sendMessage(chat_id=self.chat_id, text=msg)

                        if self.config.trader_set.messenger_on:
                            self.watch_data_dict.pop(code)
                            self.websocket_client.stop_socket("{}@aggTrade".format(code.lower()))

                    self.open_remain_data_dict.pop(code)

                # self.sys_log.info("OPEN_REMAIN : ~ execution : %.5f\n" % (time.time() - start_time))

            """
            CLOSE
            """
            start_time = time.time()

            for code, close_data in self.close_data_dict.copy().items():

                # y. data water_fall
                open_side_msg, res_df, fee, open_side, pos_side, trade_log_dict,\
                tp, ep, out, liqd_p, leverage, open_exec_price_list, open_exec_qty, post_order_res_list, fake_order,\
                real_balance, prev_exec_tp_len, ex_dict = close_data.values()
                self.sys_log.warning("CLOSE : {} {}".format(code, open_side_msg))

                # 0. get close_side
                if open_side == OrderSide.BUY:
                    close_side = OrderSide.SELL
                else:
                    close_side = OrderSide.BUY

                # 2. tr_set : 추후 dynamic 사용시 사용.
                # tp, out, tp_series, out_series = self.get_dynamic_tpout(res_df, open_side, tp, out)

                # 3. limit_tp_order (close loop 의 목적이라, switch 기능 invalid.)
                price_precision, quantity_precision = self.get_precision(code)
                partial_tps, partial_qtys = self.get_partial_tp_qty(ep, tp, open_exec_qty, price_precision, quantity_precision, close_side)

                if not self.config.trader_set.backtrade and not fake_order:
                    #   a. tp_exectuedQty 감산했던 이유 : dynamic_tp 의 경우 체결된 qty 제외
                    #   b. reduceOnly = False for multi_position

                    # [ API ]
                    post_order_res_list = self.partial_limit_order_v5(code, post_order_res_list, partial_tps, partial_qtys, close_side, pos_side, open_exec_qty, quantity_precision)

                    #   c. 주문 단가가 상한선인 경우, 주문 가능할 때까지 keep.
                    #           i. out check 는 지속적으로 수행함.
                    #           ii. 한번만 append 해야할 것, close_data 에서는 pop 하지 않음.
                    #                   1. 이것도 누적인가 ?
                    # if error_code:
                    #     time.sleep(self.config.trader_set.order_term)
                    #     continue
                else:
                    post_order_res_list = []

                # if not error_code:
                # z. queue result
                #       i. keep order 에 대한 처리를 해주어야함.
                #               1. 지속되는 post_order_res_list 를 고려해, close_data_dict, close_remain_data_dict 의 값을 수정함.
                self.close_data_dict[code]["post_order_res_list"] = post_order_res_list
                #               2. code 별로 1개의 data 만 허용하도록. => 새로운 post_order_res_list 로 변경하는건 필요한 작업임. (기존 remain_data overwrite.)
                close_remain_data_value = [open_side_msg, res_df, fee, open_side, pos_side, trade_log_dict,
                                           tp, ep, out, liqd_p, leverage, open_exec_price_list, open_exec_qty, post_order_res_list, fake_order,
                                           real_balance, prev_exec_tp_len, ex_dict,
                                           partial_tps, partial_qtys, close_side, quantity_precision]
                self.close_remain_data_dict[code] = dict(zip(self.close_remain_data_key, close_remain_data_value))

                #       iii. invalid order 가 없는 경우만 close 에서 pop 함. -> keep order. / backtrade 의 [] 인 경우도 수용함.
                if None not in post_order_res_list:
                    self.close_data_dict.pop(code)

                self.sys_log.info('CLOSE : ~ partial_limit : %.5f\n' % (time.time() - start_time))

            """
            CLOSE REMAIN LOOP
            """
            start_time = time.time()

            for code, close_remain_data in self.close_remain_data_dict.copy().items():

                # # x. loop 최대 잔류 시간.
                #       i. open, close execution loop 는 내부에서 API 사용하지 않아 최대 잔류 시간이 의미 없어짐 (loop 내 잔류 시간이 적을 것으로 예상하기 때문임.)
                # if not self.config.trader_set.backtrade:
                #     if time.time() - start_time > self.config.trader_set.loop_duration:
                #         break

                # y. data water_fall
                open_side_msg, res_df, fee, open_side, pos_side, trade_log_dict,\
                tp, ep, out, liqd_p, leverage, open_exec_price_list, open_exec_qty, post_order_res_list, fake_order,\
                real_balance, prev_exec_tp_len, ex_dict,\
                partial_tps, partial_qtys, close_side, quantity_precision = close_remain_data.values()
                # self.sys_log.warning("CLOSE_REMAIN : {} {}".format(code, open_side_msg))

                # 0. param init.
                #       a. tp / out 이 없는 경우를 위한 init.
                tp_exec_price_list, out_exec_price_list = [], []
                exec_tp_len = 0
                all_executed = 0
                market_close_on = 0

                # post_order_res_list_valid = [post_order_res_ for post_order_res_ in post_order_res_list if post_order_res_ != ""]
                # post_order_res_validation = len(post_order_res_list_valid) != 0

                # 1. CLOSE REMAIN LOOP 를 위한 data.
                #       a. open remain 와는 다르게, post_order_res_list 로 접근하기 때문에 내부에서 수행함.
                #               i. 한 종목에 대한 close post_order_res_list 를 의미함.
                # if not self.config.trader_set.backtrade:
                #     if post_order_res_validation:
                #         #   b. close 후에 order_info 는 update 된 상태가 아님, 따라서 .loc 접근시 key_error 가 한번은 발생함.
                #         try:
                #             close_order_info_valid = order_info.loc[post_order_res_list_valid]
                #         except Exception as e:
                #               msg = "error in order_info.loc[post_order_res_list_valid], close_remain_loop : {}".format(e)
                #               self.sys_log.error(msg)
                #               self.msg_bot.sendMessage(chat_id=self.chat_id, text=msg)
                #             continue
                #         close_exec_series = (close_order_info_valid['주문수량'] - close_order_info_valid['체결량']) < 1 / (10 ** quantity_precision)
                #         self.sys_log.warning("close_exec_series : {}".format(close_exec_series))

                # 2. check tp execution
                if not self.config.trader_set.backtrade and not fake_order:
                    # [ API ] : close_exec_series, close_order_info_valid
                    #       a. 손매매로 인한 order_info.status == EXPIRED 는 고려 대상이 아님. (손매매 이왕이면 하지 말라는 소리.)
                    # if post_order_res_validation:
                    #     exec_tp_len = np.sum(close_exec_series)
                    #     if len(close_exec_series) == exec_tp_len:
                    #         all_executed = 1
                    #
                    #     tp_exec_price_list = close_order_info_valid[close_exec_series == 1]["체결가"].tolist()

                    all_executed, tp_exec_price_list = self.check_limit_tp_exec_v2(post_order_res_list, quantity_precision)
                    exec_tp_len = len(tp_exec_price_list)
                else:
                    if not self.config.tp_set.non_tp:
                        if open_side == OrderSide.BUY:
                            exec_tp_len = np.sum(res_df['high'].to_numpy()[self.config.trader_set.complete_index] >= np.array(partial_tps))
                        else:
                            exec_tp_len = np.sum(res_df['low'].to_numpy()[self.config.trader_set.complete_index] <= np.array(partial_tps))

                        tp_exec_price_list = partial_tps[:exec_tp_len]  # 지속적 갱신
                        all_executed = 1 if exec_tp_len == len(partial_tps) else 0

                #       a. tp execution logging (real, back 모두 포함해서.)
                if not self.config.tp_set.non_tp:
                    if prev_exec_tp_len != exec_tp_len:  # logging 기준, 체결이 되면 prev_exec_tp_len != exec_tp_len
                        ex_dict[str(res_df.index[self.config.trader_set.complete_index])] = partial_tps[prev_exec_tp_len:exec_tp_len]

                        #       i. replace 누적형 data : prev_exec_tp_len
                        self.close_remain_data_dict[code]["prev_exec_tp_len"] = exec_tp_len
                        self.sys_log.info("ex_dict : {}".format(ex_dict))

                # 2. check out
                if not self.config.trader_set.backtrade:
                    #       a. invalid post_order_res_list 에 대응하기 위해, order_info 에 종목명으로 접근함.
                    # [ API ] : order_info
                    realtime_price = self.get_market_price_v3(code)

                    market_close_on, log_out = self.check_hl_out_v3(res_df, market_close_on, out, liqd_p, realtime_price, open_side)
                else:
                    market_close_on, log_out = self.check_hl_out_onbarclose_v2(res_df, market_close_on, out, liqd_p, open_side)

                if not market_close_on:  # log_out 갱신 방지
                    market_close_on, log_out = self.check_signal_out_v3(res_df, market_close_on, open_side)

                # 3. check all_execution
                if all_executed:
                    fee += self.config.trader_set.limit_fee
                    msg = "{} all_executed.".format(code)
                    self.sys_log.info(msg)
                    self.msg_bot.sendMessage(chat_id=self.chat_id, text=msg)

                # 4. check market_close_on
                if market_close_on:
                    fee += self.config.trader_set.market_fee
                    msg = "{} market_close_on.".format(code)
                    self.sys_log.info(msg)
                    self.msg_bot.sendMessage(chat_id=self.chat_id, text=msg)

                    #   a. out execution logging
                    #       i. market_close_on = True, log_out != None (None 도 logging 가능하긴함)
                    if not self.config.trader_set.backtrade:  # real_trade 의 경우, realtime_price's ts -> lastest_index 사용
                        ex_dict[str(res_df.index[self.config.trader_set.latest_index])] = [log_out]  # insert as list type
                    else:
                        ex_dict[str(res_df.index[self.config.trader_set.complete_index])] = [log_out]
                    self.sys_log.info("ex_dict : {}".format(ex_dict))

                    if not self.config.trader_set.backtrade and not fake_order:
                        # [ API ]
                        #       ii. Binance 는 market_close_order_v2 를 유지한다. (v3 for Stock)
                        out_exec_price_list = self.market_close_order_v2(code, post_order_res_list, close_side, pos_side, open_exec_qty)
                    else:
                        out_exec_price_list = [log_out]

                # self.sys_log.info('CLOSE_REMAIN : ~ out & execution : %.5f' % (time.time() - start_time))

                # 5. calculate profit & log
                #       a. market_order 시 ideal != real gap 발생 가능해짐.
                if all_executed or market_close_on:
                    ideal_profit, real_profit, trade_log_dict = self.calc_ideal_profit_v4(res_df, open_side, ep, ex_dict,
                                                                                          open_exec_price_list, tp_exec_price_list, out_exec_price_list, fee, trade_log_dict)

                    with open(self.dict_wrapper_save_path, "wb") as f:
                        self.trade_log_dict_wrapper.append(trade_log_dict)
                        pickle.dump(self.trade_log_dict_wrapper, f)
                        self.sys_log.info("trade_log_dict_wrapper dumped.")

                    if not fake_order:
                        self.get_income_info(real_balance, leverage, ideal_profit, real_profit, mode=self.config.trader_set.profit_mode)

                    #   z. queue result
                    self.close_remain_data_dict.pop(code)
                    #       i. messenger_on 여부에 따라 stop_socket 사용여부가 결정된다. (일회성 / 영구성을 결정하는 요인 = messenger_on)
                    if self.config.trader_set.messenger_on:
                        self.watch_data_dict.pop(code)
                        self.websocket_client.stop_socket("{}@aggTrade".format(code.lower()))

                # self.sys_log.info('CLOSE_REMAIN : ~ calc profit & income : %.5f\n' % (time.time() - start_time))

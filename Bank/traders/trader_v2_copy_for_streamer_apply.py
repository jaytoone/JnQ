from funcs.kiwoom.bank_module import BankModule
from funcs.kiwoom.constant import *
from funcs.public.broker import intmin_np

import numpy as np  # for np.nan, np.array() ...
from datetime import datetime

import time
import shutil
import importlib
import os

import pickle
import logging.config
from easydict import EasyDict
import json


class Trader:
    """
    v1_4 -> v2
        1. Stock platform 확장.
        2. Loop concept 재사용 -> 자원 낭비 줄일 것.
        3. remove orderside_changed
        4. remove first_iter
        5. min_balance 수정 -> code 에 종속될 수 있도록.
    """

    def __init__(self, paper_name, id_list, config_type, mode="Trader"):

        # 1. queues for loop
        self.watch_data_queue = []  # watch 성사 시, open_data_queue 로 전달
        self.open_data_queue = []  # open_order 유효성 검증 후, open_exec_data_queue 로 전달
        self.open_exec_data_queue = []  # ei_k 성사시 remove / exec 성사시 close_data_queue 로 전달
        self.close_data_queue = []  # close_order 유효성 검증 후, close_exec_data_queue 로 전달
        self.close_exec_data_queue = []  # out, exec 성사시 remove

        if mode == "Trader":  # IDEP 환경에서는 유효하지 않음.
            self.pkg_path = os.path.dirname(os.getcwd())  # pkg_path 가 가리켜야할 곳은 "Bank" 임.
            os.chdir(self.pkg_path)  # log dir 저장위치를 고려해서, chdir() 실행

        # 2. paper info
        #       a. public
        public_name = "Bank.papers.public.{}".format(paper_name)
        self.public = importlib.import_module(public_name)

        #       b. utils
        utils_name_list = ["Bank.papers.utils.{}_{}".format(paper_name, id_) for id_ in id_list]
        self.utils_list = [importlib.import_module(utils_name) for utils_name in utils_name_list]

        #       c. configuration
        if config_type == "backtrade":
            self.config_name_list = ["{}_{}_backtrade.json".format(paper_name, id_) for id_ in id_list]
        elif config_type == "run_validation":
            self.config_name_list = ["{}_{}_run_validation.json".format(paper_name, id_) for id_ in id_list]
        else:
            self.config_name_list = ["{}_{}.json".format(paper_name, id_) for id_ in id_list]

        self.config_list = None

        # 3. config & utils for individual ids
        self.config = None
        self.utils = None

        # 4. get paper_name for log_dir_name
        self.log_dir_name = paper_name

        # 5. balance
        self.available_balance = None
        self.over_balance = None

        # 6. pnl
        self.income = 0.0
        self.accumulated_income = 0.0
        self.accumulated_profit = 1.0  # if profit_mode == "SUM", will be changed to zero
        self.ideal_accumulated_profit = 1.0  # if profit_mode == "SUM", will be changed to zero

        # 7. obj
        self.streamer = None
        self.sys_log = logging.getLogger()

        self.initial_set = 1
        self.limit_leverage = "None"

        # 8. path definition
        if mode == "Trader":
            log_path_list = ["sys_log", "trade_log", "df_log"]
            self.sys_log_path, self.trade_log_path, self.df_log_path = \
                [os.path.join(self.pkg_path, "logs", path_, self.log_dir_name) for path_ in log_path_list]

            for path_ in [self.sys_log_path, self.trade_log_path, self.df_log_path]:
                os.makedirs(path_, exist_ok=True)
                self.config_path_list = [os.path.join(self.pkg_path, "papers/config", name_) for name_ in self.config_name_list]
        else:
            self.config_path_list = [os.path.join("Bank/papers/config", name_) for name_ in self.config_name_list]
            self.read_write_config_list(mode='r')  # IDEP 에서 self.config_list 사용하기 위해 read_write_config_list 선언함

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
            self.sys_log.error("error in read_write_config_list :", e)

    def set_copy_base_log(self, log_name, base_cfg_path, base_cfg_copy_path):
        """
         set logger info. to {ticker}.json - offer realtime modification
         1. base_cfg_copy_path 기준으로 변경 내용 .log 로 적용
        """
        while 1:
            try:
                shutil.copy(base_cfg_path, base_cfg_copy_path)

                with open(base_cfg_copy_path, 'r') as sys_cfg:  # initial_set 과 무관하게 거래 종료 시에도 반영하기 위함 - 윗 initial_set phase 와 분리한 이유
                    sys_log_cfg = EasyDict(json.load(sys_cfg))

                if self.initial_set:  # dump edited_cfg - sys_log_cfg 정의가 필요해서 윗 phase 와 분리됨

                    # 1. edit 내용
                    sys_log_cfg.handlers.file_rot.filename = \
                        os.path.join(self.sys_log_path, log_name + ".log")  # log_file name - log_name 에 종속
                    logging.getLogger("apscheduler.executors.default").propagate = False

                    with open(base_cfg_copy_path, 'w') as edited_cfg:
                        json.dump(sys_log_cfg, edited_cfg, indent=3)

                    logging.config.dictConfig(sys_log_cfg)  # 선언 순서 상관 없음.
                    self.sys_log.info('# ----------- {} ----------- #'.format(self.log_dir_name))
                    self.sys_log.info("pkg_path : {}".format(self.pkg_path))

            except Exception as e:
                print("error in load sys_log_cfg :", e)
                time.sleep(self.config.trader_set.api_term)
                continue
            else:
                return

    def get_balance_info(self, bank_module, min_balance, mode="PROD"):  # bank_module 상속이 아니므로, 인자값으로 받도록 설정함.

        min_balance_bool = 0

        # 0. define initial asset
        #       a. self.config 의 setting 을 기준으로함
        if self.accumulated_income == 0.0:
            self.available_balance = self.config.trader_set.initial_asset  # USDT
            if mode == "SUM":
                self.accumulated_profit = 0.0
                self.ideal_accumulated_profit = 0.0
        else:
            if mode == "PROD":
                self.available_balance += self.income
            # elif mode == "SUM", 항상 일정한 available_balance 사용

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
            max_available_balance = bank_module.get_available_balance(계좌번호=bank_module.account_number, 비밀번호=bank_module.password, 비밀번호입력매체구분="00", 조회구분="3")

            # a. over_balance 가 저장되어 있다면, 현재 사용하는 balance 가 원하는 것보다 작은 상태임.
            #       i. 최대 허용 가능 balance 인 max_balance 와의 지속적인 비교 진행
            if self.over_balance is not None:
                if self.over_balance <= max_available_balance * 0.9:  # 정상화 가능한 상태 (over_balanced 된 양을 허용할 수준의 max_balance)
                    self.available_balance = self.over_balance  # mode="SUM" 의 경우에도 self.over_balance 에는 initial_asset 만큼만 담길 수 있기 때문에 구분하지 않음.
                    self.over_balance = None
                else:  # 상태가 어찌되었든 일단은 지속해서 Bank 를 돌리려는 상황. (후조치 한다는 의미, 중단하는게 아니라)
                    self.available_balance = max_available_balance * 0.9  # max_available_balance 를 넘지 않는 선
                self.sys_log.info('available_balance (temp) : {}'.format(self.available_balance))

            # b. 예기치 못한 오류로 인해 over_balance 상태가 되었을때의 조치
            else:
                if self.available_balance >= max_available_balance:
                    self.over_balance = self.available_balance  # 복원 가능하도록 현재 상태를 over_balance 에 저장
                    self.available_balance = max_available_balance * 0.9
                self.sys_log.info('available_balance : {}'.format(self.available_balance))

            # c. min_balance check
            if self.available_balance < min_balance:
                self.sys_log.info('available_balance {:.3f} < min_balance\n'.format(self.available_balance))
                min_balance_bool = 1

        return min_balance_bool

    def get_income_info(self, real_balance, leverage, ideal_profit, real_profit, mode="PROD"):  # 복수 pos 를 허용하기 위한 api 미사용

        """
        v2 -> _
            1. 단리 / 복리 mode 에 따라 출력값 변경하도록 구성함.
            2. Trader 내부 메소드는 version 을 따로 명명하지 않도록함. -> Trader version 에 귀속되도록하기 위해서.
        """

        real_profit_pct = real_profit - 1
        self.income = real_balance * real_profit_pct
        self.accumulated_income += self.income

        self.sys_log.info('real_profit : {}'.format(real_profit))
        self.sys_log.info('ideal_profit : {}'.format(ideal_profit))

        real_tmp_profit = real_profit_pct * leverage
        ideal_tmp_profit = (ideal_profit - 1) * leverage

        self.sys_log.info('income : {} USDT'.format(self.income))
        self.sys_log.info('temporary profit : {:.2%} ({:.2%})'.format(real_tmp_profit, ideal_tmp_profit))

        if mode == "PROD":
            self.accumulated_profit *= 1 + real_tmp_profit
            self.ideal_accumulated_profit *= 1 + ideal_tmp_profit
            self.sys_log.info('accumulated profit : {:.2%} ({:.2%})'.format((self.accumulated_profit - 1), (self.ideal_accumulated_profit - 1)))
        else:
            self.accumulated_profit += real_tmp_profit
            self.ideal_accumulated_profit += ideal_tmp_profit
            self.sys_log.info('accumulated profit : {:.2%} ({:.2%})'.format(self.accumulated_profit, self.ideal_accumulated_profit))

        self.sys_log.info('accumulated income : {} USDT\n'.format(self.accumulated_income))

    def run(self):

        # 0. we need first configuration set
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

        # 1. set first BankModule
        bank_module = BankModule(self.config)

        # 2. log_name for dict_wrapper_save_path & base_cfg_copy_path (+ sys_log file name)
        log_name = str(datetime.now().timestamp()).split(".")[0]

        #       a. dict_wrapper_save_path define
        dict_wrapper_save_path = os.path.join(self.trade_log_path, log_name + ".pkl")

        #       b. copy base_cfg.json -> {ticker}.json (log_cfg name)
        base_cfg_path = self.sys_log_path.replace(self.log_dir_name, "base_cfg.json")  # dir 와 같은 수준에 위치하는 baes_cfg.json 가져옴
        base_cfg_copy_path = os.path.join(self.sys_log_path, log_name + ".json")
        self.set_copy_base_log(log_name, base_cfg_path, base_cfg_copy_path)

        #       c. posmode_margin_leverage deprecated in Stock.
        # self.limit_leverage = bank_module.posmode_margin_leverage()

        """
        ONE LOOP
        """
        loop_limit = 3  # API request weight.
        static_watch = 1  # watch_queue 고정 (no pop)
        trade_log_dict_wrapper = []  # trade_log_dict 에 저장되는 데이터의 chunk_size 는 1 거래완료를 기준으로한다. (dict 로 운영할시 key 값 중복으로 인해 데이터 상쇄가 발생함.)
        while 1:

            # y. Bank on & off
            #       a. 장 개시 ~ 마감 중으로 설정. (Todo)
            if not self.config.trader_set.run:
                time.sleep(self.config.trader_set.realtime_term)  # give enough time to read config
                continue

            """
            ADD WATCH
            """
            #   1. 전일대비상승률 200개 내에서, Top 10 사용.
            #           a. calibration 적용 (Todo)
            if len(self.watch_data_queue) == 0:
                try:
                    df_high_fluc = bank_module.get_high_fluc_than_day_before(시장구분="101", 정렬구분="1", 거래량조건="0000", 종목조건="0",
                                                                             신용조건="0", 상하한포함="1", 가격조건="0", 거래대금조건="0")
                except Exception as e:
                    self.sys_log.error("error in get_high_fluc_than_day_before : {}".format(e))
                    continue
                else:
                    top_df = df_high_fluc.iloc[:10]
                    top_df["ep_loc_point2"] = 0
                    self.watch_data_queue = top_df[["종목코드", "ep_loc_point2"]].values.tolist()

                    #   b. bank_module 내부에 streamer 선언은 loop 이탈하면 초기화되서, Trader 의 self.streamer 로 유지함.
                    #       i. append streamer code by code
                    #           Todo, 여기서도 초기화될런지 ... -> 확인 필요함.
                    if self.config.trader_set.backtrade:
                        for w_zero_i, watch_data in enumerate(self.watch_data_queue):
                            self.watch_data_queue[w_zero_i] = watch_data.append(bank_module.get_streamer())
                    prev_w_i = 0

            """
            WATCH LOOP
            """
            for w_i, watch_data in zip(reversed(range(len(self.watch_data_queue))), reversed(self.watch_data_queue)):

                # x. set start w_i
                #       a. 시작 w_i 를 변경하는건, static_watch 만 해당됨. (dynamic 은 불필요함.)
                if static_watch:
                    #       i. assert w_i > prev_w_i.
                    if w_i <= prev_w_i:
                        continue

                start_time = time.time()

                code, ep_loc_point2 = watch_data  # 이외의 data 가 필요한지 recheck (Todo)

                # 0. param init.
                open_side = None
                self.utils = None

                # 1. get_new_df
                if self.config.trader_set.backtrade:
                    # Todo, streamer 는 back_data_path 로 선언하기 때문에 단일 종목에 기준하고 있음.
                    #       1. code 에 종속되어 변경될 수 있어야함.
                    res_df, _ = bank_module.get_new_df_onstream(self.streamer)
                else:
                    res_df, _ = bank_module.get_new_df(code)
                self.sys_log.info("WATCH : ~ get_new_df : %.2f" % (time.time() - start_time))

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
                        self.sys_log.info('check res_df.index[-1] : {}\n'.format(res_df.index[-1]))

                except Exception as e:
                    self.sys_log.error("error in self.utils_ : {}".format(e))
                    continue

                # 3. signal check
                if open_side is None:

                    #   a. ep_loc - get_open_side_v2
                    try:
                        papers = (self.public, self.utils_list, self.config_list)
                        open_side, self.utils, self.config = bank_module.get_open_side_v2(res_df, papers, np_timeidx, open_num=1)

                    except Exception as e:
                        self.sys_log.error("error in ep_loc phase : {}".format(e))
                        continue

                    #   b. df logging
                    if self.config.trader_set.df_log:  # save res_df at ep_loc
                        excel_name = str(datetime.now()).replace(":", "").split(".")[0]
                        res_df.reset_index().to_feather(os.path.join(self.df_log_path, "{}.ftr".format(excel_name)), compression='lz4')

                    #   c. open_signal 발생의 경우
                    if open_side is not None:

                        #       i. res_df_open 정의는 이곳에서 해야함 - 아래서 할 경우 res_df_open 갱신되는 문제
                        res_df_open = res_df.copy()  # point2 에 사용됨.

                        #       ii. BankModule new config 로 재선언
                        bank_module = BankModule(self.config, login=False)  # relogin 은 하지 않음.

                        #       iii. market_check_term for market_entry - deprecated
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

                        #       y. set trade_log_dict
                        trade_log_dict = {"last_trading_time": str(datetime.now())}

                        #       z. queue result
                        self.open_data_queue.append([code, res_df_open, res_df, fee, open_side, trade_log_dict])

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

                # z. queue result
                #       a. dynamic_watch 의 경우, watch 수행시마다 pop -> watch 를 비워 new_watch 를 수신하기 위함.
                if not static_watch:
                    self.watch_data_queue.pop(w_i)  # list data 를 지우기 위해서 pop(index) 로 사용함.

                self.sys_log.info('WATCH : ~ signal_check : %.5f' % (time.time() - start_time))

                # 5. loop_limit
                #       a. loop_limit 도달시, 새로운 data_flow 밑으로 내리기. (= "break")
                if w_i % loop_limit == loop_limit - 1:
                    #   b. when prev_w_i reached last_index, set to 0.
                    #       i. w_i 는 len(self.watch_data_queue) - 1 보다 클 수 없음.
                    if w_i == len(self.watch_data_queue) - 1:
                        prev_w_i = 0
                    else:
                        prev_w_i = w_i
                    break

            """
            OPEN
            """
            for o_i, open_data in zip(reversed(range(len(self.open_data_queue))), reversed(self.open_data_queue)):

                start_time = time.time()

                # y. data water_fall
                code, res_df_open, res_df, fee, open_side, trade_log_dict = open_data

                # 0. param init.
                fake_order = 0  # order 하는 것처럼 시간을 소비하기 위함임.

                # 1. get tr_set, leverage
                tp, ep, out, open_side = bank_module.get_tpepout(open_side, res_df_open, res_df)  # Todo, 일단은, ep1 default 로 설정

                # 2. get balance
                #       a. Stock 특성상, min_balance = ep 로 설정함.
                min_balance_bool = self.get_balance_info(bank_module, ep, mode=self.config.trader_set.profit_mode)
                if min_balance_bool:
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

                leverage, liqd_p = self.public.lvrg_liqd_set(res_df, self.config, open_side, ep, out, fee, self.limit_leverage)

                #       b. get precision
                quantity_precision = 0  # fixed in Stock
                tp, ep, out, liqd_p = [bank_module.calc_with_hoga_unit(price_) for price_ in [tp, ep, out, liqd_p]]  # includes half-dynamic tp

                self.sys_log.info('tp : {}'.format(tp))
                self.sys_log.info('ep : {}'.format(ep))
                self.sys_log.info('out : {}'.format(out))
                self.sys_log.info('liqd_p : {}'.format(liqd_p))
                self.sys_log.info('leverage : {}'.format(leverage))
                self.sys_log.info('OPEN : ~ tp ep out leverage set : %.5f' % (time.time() - start_time))

                if leverage is None:
                    self.sys_log.info("leverage_rejection occured.")
                    continue

                # if not self.config.trader_set.backtrade:  # not for Stock
                #     while 1:
                #         try:
                #             bank_module.change_initial_leverage(symbol=self.config.trader_set.symbol, leverage=leverage)
                #         except Exception as e:
                #             self.sys_log.error('error in change_initial_leverage : {}'.format(e))
                #             time.sleep(self.config.trader_set.api_term)
                #             continue  # -->  ep market 인 경우에 조심해야함 - why ..?
                #         else:
                #             self.sys_log.info('leverage changed --> {}'.format(leverage))
                #             break

                #       c. calc. open_quantity
                open_quantity = bank_module.calc_with_precision(self.available_balance / ep * leverage, quantity_precision)
                self.sys_log.info("open_quantity : {}".format(open_quantity))

                # 3. open order
                if not self.config.trader_set.backtrade and not fake_order:
                    # [ back & real-trade validation phase ]
                    order_info = (self.available_balance, leverage)
                    order_no, self.over_balance, error_code = bank_module.limit_order(order_info, code=code, order_side=open_side, qty=open_quantity, price=ep)

                    if not error_code:  # order deny exception, error_code = 0
                        # z. queue result
                        self.open_exec_data_queue.append([code, order_no, tp, ep, out, liqd_p, leverage, fee, open_side, open_quantity, res_df_open, res_df, fake_order, trade_log_dict])
                        self.open_data_queue.pop(o_i)  # 유효성이 검증된 open_order_data 는 지워줄 것.
                    else:
                        continue

                self.sys_log.info('OPEN : ~ open_order : %.5f' % (time.time() - start_time))

            """
            전체 종목에 대한 order_info 조회
            """
            if not self.config.trader_set.backtrade:
                try:
                    order_info = bank_module.get_order_info(계좌번호=bank_module.account_number, 전체종목구분=0, 매매구분="0", 종목코드="", 체결구분="0")

                except Exception as e:
                    self.sys_log.error("error in get_order_info : {}".format(e))
                    continue

                #       1. OPEN EXECUTION LOOP 를 위한 data.
                #               a. broadcast 연산 위해서 open exec 는 이곳에서 수행함.
                open_exec_data_arr = np.array(self.open_exec_data_queue, dtype=object)
                open_order_no_arr = open_exec_data_arr[:, 1]

                open_order_info_valid = order_info.loc[open_order_no_arr]  # order_no validation checked.
                open_exec_series = (open_order_info_valid['체결량'] / open_order_info_valid['주문수량'] >= self.config.trader_set.open_exec_qty_ratio) | (open_order_info_valid['주문상태'] == "체결")

            """
            OPEN EXECUTION LOOP
            """
            for oe_i, open_exec_data in zip(reversed(range(len(self.open_exec_data_queue))), reversed(self.open_exec_data_queue)):

                start_time = time.time()

                # y. data water_fall
                code, order_no, tp, ep, out, liqd_p, leverage, fee, open_side, open_quantity, res_df_open, res_df, fake_order, trade_log_dict = open_exec_data

                # 0. param init.
                #       a. expired 가 없는 경우를 위한 init.
                expired = 0

                # 1. market
                open_exec = 0
                if self.config.ep_set.entry_type == OrderType.MARKET:
                    if self.config.trader_set.backtrade:
                        res_df, _ = bank_module.get_new_df_onstream(self.streamer)  # + index, code 에 종속될 수 있게 수정 (Todo)
                    open_exec = 1

                # 2. limit
                else:
                    #   [ back & real-trade validation phase ]
                    if not self.config.trader_set.backtrade:

                        #   a. ei_k
                        #       ii. expire_tp default
                        realtime_price = open_order_info_valid['현재가'].loc[order_no]
                        expired = bank_module.check_ei_k_v2(res_df_open, res_df, realtime_price, open_side)

                        # b. open_execution
                        if fake_order:
                            if open_side == OrderSide.BUY:
                                if realtime_price <= ep:
                                    open_exec = 1
                            else:
                                if realtime_price >= ep:
                                    open_exec = 1
                        else:
                            open_exec = open_exec_series.loc[order_no]
                    else:
                        res_df, _ = bank_module.get_new_df_onstream(self.streamer)  # + index
                        c_i = self.config.trader_set.complete_index  # 단지, 길이 줄이기 위해 c_i 로 재선언하는 것뿐, phase 가 많지 않아 수정하지 않음.

                        #   a. ei_k
                        #       i. expire_tp default
                        expired = bank_module.check_ei_k_onbarclose_v2(res_df_open, res_df, c_i, c_i, open_side)  # e_j, tp_j

                        # b. open_execution
                        if open_side == OrderSide.BUY:
                            if res_df['low'].to_numpy()[c_i] <= ep:
                                open_exec = 1
                        else:
                            if res_df['high'].to_numpy()[c_i] >= ep:
                                open_exec = 1

                self.sys_log.info('OPEN_EXECUTION : ~ ei_k : %.5f' % (time.time() - start_time))

                if expired or open_exec:

                    #   c. when open order time expired or executed, regardless to position exist, cancel open orders       #
                    if not self.config.trader_set.backtrade and not fake_order:
                        # [ back & real-trade validation phase ]
                        open_exec_price_list, open_exec_qty = bank_module.cancel_order_list(code, open_side, [order_no], order_type=OrderType.LIMIT)
                    else:
                        open_exec_price_list = [ep]
                        open_exec_qty = open_quantity if open_exec else 0

                    if open_exec:
                        self.sys_log.info('open order executed')
                        self.sys_log.info("open_exec_price_list : {}".format(open_exec_price_list))

                        real_balance = open_exec_price_list[0] * open_exec_qty
                        self.sys_log.info("real_balance : {}".format(real_balance))  # define for pnl calc.

                        # d. add trade_log_dict open info.
                        #       i. backtrade's entry timestamp 의 경우, lastest_index 가 complete_index 로. (위에서 c_i 를 사용함)
                        trade_log_dict[str(res_df_open.index[self.config.trader_set.complete_index])] = [open_side, "open"]

                        if not self.config.trader_set.backtrade:
                            date_str = str(datetime.now())
                            trade_log_dict[date_str.replace(date_str.split(':')[-1], "59.999000")] = [ep, open_side, "entry"]
                        else:
                            trade_log_dict[str(res_df.index[self.config.trader_set.complete_index])] = [ep, open_side, "entry"]

                        # z. queue result
                        self.close_data_queue.append([code, tp, ep, out, liqd_p, leverage, fee, open_side, open_exec_price_list, open_exec_qty, real_balance, res_df, fake_order, trade_log_dict])
                        self.open_exec_data_queue.pop(oe_i)

                self.sys_log.info('OPEN_EXECUTION : ~ execution : %.5f' % (time.time() - start_time))

            """
            CLOSE
            """
            for cl_i, close_data in zip(reversed(range(len(self.close_data_queue))), reversed(self.close_data_queue)):

                start_time = time.time()

                # y. data water_fall
                code, tp, ep, out, liqd_p, leverage, fee, open_side, open_exec_price_list, open_exec_qty, real_balance, res_df, fake_order, trade_log_dict = close_data

                # 0. get close_side
                if open_side == OrderSide.BUY:
                    close_side = OrderSide.SELL
                else:
                    close_side = OrderSide.BUY

                # 1. get_new_df => 없앨 예정 (Todo)
                #       a. 기존에 왜 있었던 거지 ..? => dynamic_tp / out 같음.
                # if self.config.trader_set.backtrade:
                #     res_df, load_new_df2 = bank_module.get_new_df_onstream(self.streamer)
                # else:
                #     res_df, load_new_df2 = bank_module.get_new_df(code, mode="CLOSE")
                #
                # try:
                #     res_df = self.public.sync_check(res_df, self.config, mode="CLOSE")
                #     np_timeidx = np.array([intmin_np(date_) for date_ in res_df.index.to_numpy()])  # should be located af. row_slice
                #     res_df = self.public.public_indi(res_df, self.config, np_timeidx, mode="CLOSE")
                #     res_df = self.utils.enlist_tr(res_df, self.config, np_timeidx, mode="CLOSE", show_detail=False)
                #
                # except Exception as e:
                #     self.sys_log.error("error in sync_check / public_indi / utils_ (load_new_df2 phase) : {}".format(e))
                #     continue
                #
                # # 2. tr_set : dynamic 고려된.
                # tp, out, tp_series, out_series = bank_module.get_dynamic_tpout(res_df, open_side, tp, out)

                # 3. limit_tp_order (close loop 의 목적이라, switch 기능 invalid.)
                quantity_precision = 0  # fixed in Stock
                partial_tps, partial_qtys = bank_module.get_partial_tp_qty(ep, tp, open_exec_qty, quantity_precision, close_side)

                if not self.config.trader_set.backtrade and not fake_order:
                    #   a. tp_exectuedQty 감산했던 이유 : dynamic_tp 의 경우 체결된 qty 제외
                    #   b. reduceOnly = False for multi_position

                    # [ back & real-trade validation phase ]
                    order_no_list, error_code = bank_module.partial_limit_order_v4(code, partial_tps, partial_qtys, close_side, open_exec_qty, quantity_precision)

                    # z. queue result
                    if not error_code:
                        #       i. add 0, {} for 누적형 parameter : prev_exec_tp_len, ex_dict
                        self.close_exec_data_queue.append([code, order_no_list, partial_tps, ep, out, liqd_p, leverage, fee, open_side, close_side, open_exec_price_list,
                                                           open_exec_qty, quantity_precision, real_balance, res_df, fake_order, trade_log_dict, 0, {}])
                        self.close_data_queue.pop(cl_i)
                    else:
                        continue

                self.sys_log.info('CLOSE : ~ partial_limit : %.5f' % (time.time() - start_time))

            """
            CLOSE EXECUTION LOOP
            """
            for ce_i, close_exec_data in zip(reversed(range(len(self.close_exec_data_queue))), reversed(self.close_exec_data_queue)):

                start_time = time.time()

                # y. data water_fall
                code, order_no_list, partial_tps, ep, out, liqd_p, leverage, fee, open_side, close_side, open_exec_price_list, \
                open_exec_qty, quantity_precision, real_balance, res_df, fake_order, trade_log_dict, prev_exec_tp_len, ex_dict = close_exec_data

                # 0. param init.
                #       a. tp / out 이 없는 경우를 위한 init.
                tp_exec_price_list, out_exec_price_list = [], []
                exec_tp_len = 0
                all_executed = 0
                market_close_on = 0

                # 1. CLOSE EXECUTION LOOP 를 위한 data.
                #       a. open exec 와는 다르게, order_no_list 로 접근하기 때문에 내부에서 수행함.
                if not self.config.trader_set.backtrade:
                    close_order_info_valid = order_info.loc[order_no_list]
                    close_exec_series = ((close_order_info_valid['주문수량'] - close_order_info_valid['체결량']) < 1 / (10 ** quantity_precision)) | (close_order_info_valid['주문상태'] == "체결")

                # 2. check tp execution
                if not self.config.trader_set.backtrade and not fake_order:
                    # [ back & real-trade validation phase ]
                    #       a. 손매매로 인한 order_info.status == EXPIRED 는 고려 대상이 아님. (손매매 이왕이면 하지 말라는 소리.)
                    exec_tp_len = np.sum(close_exec_series)
                    if len(close_exec_series) == exec_tp_len:
                        all_executed = 1

                    tp_exec_price_list = close_order_info_valid[close_exec_series == 1]["체결가"].tolist()
                else:
                    if not self.config.tp_set.non_tp:
                        res_df, _ = bank_module.get_new_df_onstream(self.streamer)
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
                        close_exec_data[-1] = exec_tp_len
                        self.sys_log.info("ex_dict : {}".format(ex_dict))

                # 2. check out
                #   [ back & real-trade validation phase ]
                if not self.config.trader_set.backtrade:
                    realtime_price = close_order_info_valid['현재가'].loc[order_no_list[0]]  # order_no_list 모두 동일 종목에 대한 주문번호.
                    market_close_on, log_out = bank_module.check_hl_out_v3(res_df, market_close_on, out, liqd_p, realtime_price, open_side)
                else:
                    market_close_on, log_out = bank_module.check_hl_out_onbarclose_v2(res_df, market_close_on, out, liqd_p, open_side)

                if not market_close_on:  # log_out 갱신 방지
                    market_close_on, log_out = bank_module.check_signal_out_v3(res_df, market_close_on, open_side)

                # 3. check all_execution
                if all_executed:
                    fee += self.config.trader_set.limit_fee
                    self.sys_log.info('all_executed.')

                # 4. check market_close_on
                if market_close_on:
                    fee += self.config.trader_set.market_fee
                    self.sys_log.info('market_close_on.')

                    #   a. out execution logging
                    # market_close_on = True, log_out != None (None 도 logging 가능하긴함)
                    if not self.config.trader_set.backtrade:  # real_trade 의 경우, realtime_price's ts -> lastest_index 사용
                        ex_dict[str(res_df.index[self.config.trader_set.latest_index])] = [log_out]  # insert as list type
                    else:
                        ex_dict[str(res_df.index[self.config.trader_set.complete_index])] = [log_out]
                    self.sys_log.info("ex_dict : {}".format(ex_dict))

                    if not self.config.trader_set.backtrade and not fake_order:
                        # [ back & real-trade validation phase ]
                        out_exec_price_list = bank_module.market_close_order_v2(order_no_list, open_exec_qty, code=code, order_side=close_side, qty=open_exec_qty)
                    else:
                        out_exec_price_list = [log_out]

                self.sys_log.info('CLOSE_EXECUTION : ~ out & execution : %.5f' % (time.time() - start_time))

                # 5. calculate profit & log
                #       a. market_order 시 ideal != real gap 발생 가능해짐.
                if all_executed or market_close_on:
                    ideal_profit, real_profit, trade_log_dict = bank_module.calc_ideal_profit_v4(res_df, open_side, ep, ex_dict, open_exec_price_list,
                                                                                                 tp_exec_price_list, out_exec_price_list, fee, trade_log_dict)

                    with open(dict_wrapper_save_path, "wb") as f:
                        trade_log_dict_wrapper.append(trade_log_dict)
                        pickle.dump(trade_log_dict_wrapper, f)
                        self.sys_log.info("trade_log_dict_wrapper dumped.")

                    if not fake_order:
                        self.get_income_info(real_balance, leverage, ideal_profit, real_profit, mode=self.config.trader_set.profit_mode)

                    #   z. queue result
                    self.close_exec_data_queue.pop(ce_i)

                self.sys_log.info('CLOSE_EXECUTION : ~ calc profit & income : %.5f' % (time.time() - start_time))

from funcs.binance.bank_modules_v2 import BankModule
from binance_f.model import *
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
    v1_3 -> v1_4
        1. 단리 모드 도입
        2. liquidation platform 도입 (margin_type : CROSS 허용)
            a. check_hl_out_v2 (onbarclose) 도입
    """

    def __init__(self, paper_name, id_list, config_type, mode="Trader"):

        if mode == "Trader":    # IDEP 환경에서는 유효하지 않음.
            self.pkg_path = os.path.dirname(os.getcwd())  # pkg_path 가 가리켜야할 곳은 "Bank" 임.
            os.chdir(self.pkg_path)  # log dir 저장위치를 고려해서, chdir() 실행

        # -------------- paper info. -------------- #
        # ------- public ------- #
        public_name = "Bank.papers.public.{}".format(paper_name)
        self.public = importlib.import_module(public_name)

        # ------- utils ------- #
        utils_name_list = ["Bank.papers.utils.{}_{}".format(paper_name, id_) for id_ in id_list]
        self.utils_list = [importlib.import_module(utils_name) for utils_name in utils_name_list]

        # ------- configuration ------- #
        if config_type == "backtrade":
            self.config_name_list = ["{}_{}_backtrade.json".format(paper_name, id_) for id_ in id_list]
        elif config_type == "run_validation":
            self.config_name_list = ["{}_{}_run_validation.json".format(paper_name, id_) for id_ in id_list]
        else:
            self.config_name_list = ["{}_{}.json".format(paper_name, id_) for id_ in id_list]

        self.config_list = None

        # ------- config & utils for individual ids ------- #
        self.config = None
        self.utils = None

        # ------ get paper_name for log_dir_name ------ #
        self.log_dir_name = paper_name

        # ------ balance ------ #
        self.available_balance = None
        self.over_balance = None
        self.min_balance = 5.0  # USDT

        # ------ pnl ------ #
        self.income = 0.0
        self.accumulated_income = 0.0
        self.accumulated_profit = 1.0   # if profit_mode == "SUM", will be changed to zero
        self.ideal_accumulated_profit = 1.0   # if profit_mode == "SUM", will be changed to zero

        # ------ obj ------ #
        self.streamer = None
        self.sys_log = logging.getLogger()

        self.initial_set = 1
        self.limit_leverage = "None"

        # ------ path definition ------ #
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

            # ------ opened files should be closed --> 닫지 않으면 reopen 시 error occurs ------ #
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

                    # ------ edit 내용 ------ #
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

    def get_balance_info(self, bank_module, first_iter, mode="PROD"):

        min_balance_bool = 0

        if first_iter:
            # ---------- define initial asset ---------- #
            #       self.config 의 setting 을 기준으로함       #
            if self.accumulated_income == 0.0:
                self.available_balance = self.config.trader_set.initial_asset  # USDT
                if mode == "SUM":
                    self.accumulated_profit = 0.0
                    self.ideal_accumulated_profit = 0.0
            else:
                if mode == "PROD":
                    self.available_balance += self.income
                # elif mode == "SUM", 항상 일정한 available_balance 사용

            # ---------- asset_change - 첫 거래여부와 무관하게 진행 ---------- #
            if self.config.trader_set.asset_changed:
                self.available_balance = self.config.trader_set.initial_asset
                #        1. 이런식으로, cfg 의 값을 변경(dump)하는 경우를 조심해야함 - config_list[0] 에 입력해야한다는 의미
                self.config_list[0].trader_set.asset_changed = 0
                with open(self.config_path_list[0], 'w') as cfg:
                    json.dump(self.config_list[0], cfg, indent=2)

                self.sys_log.info("asset_changed 1 --> 0")

        # ---------- get availableBalance ---------- #
        if not self.config.trader_set.backtrade:
            max_available_balance = bank_module.get_available_balance()

            # 1. over_balance 가 저장되어 있다면, 현재 사용하는 balance 가 원하는 것보다 작은 상태임.
            #   a. 최대 허용 가능 balance 인 max_balance 와의 지속적인 비교 진행
            if self.over_balance is not None:
                if self.over_balance <= max_available_balance * 0.9:  # 정상화 가능한 상태 (over_balanced 된 양을 허용할 수준의 max_balance)
                    self.available_balance = self.over_balance  # mode="SUM" 의 경우에도 self.over_balance 에는 initial_asset 만큼만 담길 수 있기 때문에 구분하지 않음.
                    self.over_balance = None
                else:   # 상태가 어찌되었든 일단은 지속해서 Bank 를 돌리려는 상황. (후조치 한다는 의미, 중단하는게 아니라)
                    self.available_balance = max_available_balance * 0.9  # max_available_balance 를 넘지 않는 선
                self.sys_log.info('available_balance (temp) : {}'.format(self.available_balance))

            # 2. 예기치 못한 오류로 인해 over_balance 상태가 되었을때의 조치
            else:
                if self.available_balance >= max_available_balance:
                    self.over_balance = self.available_balance  # 복원 가능하도록 현재 상태를 over_balance 에 저장
                    self.available_balance = max_available_balance * 0.9
                self.sys_log.info('available_balance : {}'.format(self.available_balance))

            # 3. min_balance check
            if self.available_balance < self.min_balance:
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

        while 1:
            """
            1. read config (refreshed by every trades)
                a. 기본 세팅을 위한 config 는 첫번째 config 사용
            """
            self.read_write_config_list(mode='r')

            for cfg_idx, cfg_ in enumerate(self.config_list):
                if cfg_idx == 0:
                    self.config = cfg_  # 첫 running 을 위한, config 선언
                else:
                    cfg_.trader_set = self.config.trader_set

            # ------ run on / off ------ #
            if not self.config.trader_set.run:
                time.sleep(self.config.trader_set.realtime_term)  # give enough time to read config
                continue

            """
            1. 여러개의 config_list 를 운용하기 위해 상속하지 않고, class 선언해서 사용함.
            2. self.config update 이후로는 BankModule 재선언할 것.
            """
            bank_module = BankModule(self.config)

            """
            initial_set 변경 조건
            1. symbol_changed is True
            """
            if self.config.trader_set.symbol_changed:
                self.config_list[0].trader_set.symbol_changed = 0
                self.initial_set = 1  # symbol changed --> reset initial_set

            """
            log_name 이 특정 상황 (symbol_change...) 에 대해 종속적이기 때문에, initial_set phase 를 둠 (변경 가능하도록)
            """
            if self.initial_set:

                """
                1. bank_module 내부에 streamer 선언은 loop 이탈하면 초기화되서, Trader 의 self.streamer 로 유지함.
                2. streamer 은 config 가 변경될 때 bank_module 과 같이 초기화되기 분리시킴 
                """
                if self.config.trader_set.backtrade:
                    self.streamer = bank_module.get_streamer()

                # ------ rewrite modified self.config_list ------ #
                self.read_write_config_list(mode='w', edited_config_list=self.config_list)

                log_name = "{}_{}".format(self.config.trader_set.symbol, str(datetime.now().timestamp()).split(".")[0])

                # ------ trade_log_fullpath define ------ #
                trade_log_dict = {}  # for ideal ep & tp logging
                trade_log_fullpath = os.path.join(self.trade_log_path, log_name + ".pkl")

                # ------ copy base_cfg.json -> {ticker}.json (log_cfg name) ------ #
                base_cfg_path = self.sys_log_path.replace(self.log_dir_name, "base_cfg.json")  # dir 와 같은 수준에 위치하는 baes_cfg.json 가져옴
                base_cfg_copy_path = os.path.join(self.sys_log_path, log_name + ".json")

                self.set_copy_base_log(log_name, base_cfg_path, base_cfg_copy_path)

                self.limit_leverage = bank_module.posmode_margin_leverage()  # self.config 가 위에서 정의된 상황
                self.sys_log.info("posmode_margin_leverage done\n")

                self.initial_set = 0

            # ------ open param. init by every_trades ------ #
            load_new_df = 1
            self.utils = None
            open_side, pos_side = None, None
            fake_order = 0  # 말그대로, order 하는 척 돌리는 것을 의미함. 특정 id 경우에 fake_order 필요함
            ep_loc_point2 = 0  # use_point2 사용시 loop 내에서 enlist once 를 위한 var.
            expired = 0

            while 1:
                if load_new_df:
                    # ------ log last trading time ------ #
                    #        미체결을 고려해, load_new_df 마다 log 수행       #
                    trade_log_dict["last_trading_time"] = str(datetime.now())
                    with open(trade_log_fullpath, "wb") as dict_f:
                        pickle.dump(trade_log_dict, dict_f)

                    # ------------ get_new_df ------------ #
                    start_ts = time.time()
                    if self.config.trader_set.backtrade:
                        res_df, load_new_df = bank_module.get_new_df_onstream(self.streamer)
                    else:
                        res_df, load_new_df = bank_module.get_new_df()
                    self.sys_log.info("~ load_new_df time : %.2f" % (time.time() - start_ts))

                    # ------------ paper works ------------ #
                    try:
                        res_df = self.public.sync_check(res_df, self.config)  # function usage format maintenance
                        self.sys_log.info('~ sync_check time : %.5f' % (time.time() - start_ts))

                        np_timeidx = np.array([intmin_np(date_) for date_ in res_df.index.to_numpy()])

                        res_df = self.public.public_indi(res_df, self.config, np_timeidx)
                        self.sys_log.info('~ public_indi time : %.5f' % (time.time() - start_ts))

                        # ------ use_point2 사용시, 해당 ID 로만 enlist_ 진행 ------ #
                        if ep_loc_point2:
                            res_df = self.utils.enlist_tr(res_df, self.config, np_timeidx, show_detail=self.config.trader_set.show_detail)
                        else:
                            for utils_, config_ in zip(self.utils_list, self.config_list):
                                res_df = utils_.enlist_tr(res_df, config_, np_timeidx, show_detail=config_.trader_set.show_detail)
                        self.sys_log.info('~ enlist_tr time : %.5f' % (time.time() - start_ts))

                        if not self.config.trader_set.backtrade:
                            self.sys_log.info('check res_df.index[-1] : {}'.format(res_df.index[-1]))
                        else:
                            self.sys_log.info('check res_df.index[-1] : {}\n'.format(res_df.index[-1]))

                    except Exception as e:
                        self.sys_log.error("error in self.utils_ : {}".format(e))
                        load_new_df = 1
                        continue

                    if open_side is None:  # open_signal not exists, check ep_loc
                        try:
                            # ------ df logging ------ #
                            if self.config.trader_set.df_log:  # save res_df at ep_loc
                                excel_name = str(datetime.now()).replace(":", "").split(".")[0]
                                res_df.reset_index().to_feather(os.path.join(self.df_log_path, "{}.ftr".format(excel_name)), compression='lz4')

                            # ------ ep_loc - get_open_side_v2 ------ #
                            papers = (self.public, self.utils_list, self.config_list)
                            open_side, self.utils, self.config = bank_module.get_open_side_v2(res_df, papers, np_timeidx)

                        except Exception as e:
                            self.sys_log.error("error in ep_loc phase : {}".format(e))
                            continue

                        """
                        1. open_signal 발생의 경우
                            a. res_df_open 정의는 이곳에서 해야함 - 아래서 할 경우 res_df_open 갱신되는 문제
                            b. BankModule new config 로 재선언
                        """
                        if open_side is not None:
                            res_df_open = res_df.copy()
                            bank_module = BankModule(self.config)

                # ------------ after load_new_df - check open_signal again ------------ #
                # <-- 이줄에다가 else 로 아래 phase 옮길 수 없나 ? --> 안됨, ep_loc survey 진행후에도 아래 phase 는 실행되어야함
                if open_side is not None:
                    # ------ 1. market_check_term for market_entry ------ # --> 바로 진행할 경우, data loading delay 로 인한 오류 방지
                    if self.config.ep_set.entry_type == "MARKET" and not self.config.trader_set.backtrade:
                        market_check_term = datetime.now().second
                        if market_check_term > self.config.trader_set.market_check_term:
                            open_side = None
                            self.sys_log.warning("market_check_term : {}".format(market_check_term))
                            continue  # ep_loc_check = False 라서 open_side None phase 로 감 -> 무슨 의미 ? 어쨌든 wait_zone 으로 회귀

                    # ------ 2. init fee ------ #
                    if self.config.ep_set.entry_type == "MARKET":
                        fee = self.config.trader_set.market_fee
                    else:
                        fee = self.config.trader_set.limit_fee

                    #        a. ep_loc.point2 - point1 과 동시성이 성립하지 않음, 가 위치할 곳      #
                    #           i. out 이 point 에 따라 변경되기 때문에 이곳이 적합한 것으로 판단     #
                    #        b. load_df phase loop 돌릴 것
                    #        c. continue 바로하면 안되고, 분마다 진행해야할 것 --> waiting_zone while loop 적용
                    #           i. 첫 point2 검사는 바로진행해도 될것
                    #           ii. ei_k check 이 realtime 으로 진행되어야한다는 점
                    #               i. => ei_k check by ID

                    # ------ 3. set selection_id ------ #  not used anymore
                    # selection_id = self.config.selection_id

                    # ------ 4. point2 (+ ei_k) phase ------ #  we don't use now.
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
                    break

                # -------------- open_side is None - no_signal holding phase -------------- #
                #        1. 추후 position change platform 으로 변경 가능
                #           a. first_iter 사용
                #        2. ep_loc.point2 를 위해 이 phase 를 잘 활용해야할 것      #
                while 1:
                    # ------- check bar_ends - latest df confirmation ------- #
                    if self.config.trader_set.backtrade or datetime.timestamp(res_df.index[-1]) < datetime.now().timestamp():
                        # ------ use_point2 == 1,  res_df 갱신 불필요
                        #        반대의 경우, utils 초기화 ------ #
                        if not ep_loc_point2:
                            self.utils = None

                        if not self.config.trader_set.backtrade:
                            self.sys_log.info("res_df[-1] timestamp : {}".format(datetime.timestamp(res_df.index[-1])))
                            self.sys_log.info("current timestamp : {}\n".format(datetime.now().timestamp()))

                            # ------ backtrade 에서 불필요한 단계로 봄, 지연 시간 길어지는 현상 발생하기 때문 ------ #
                            # ------- 1. realtime reaction to configuration change every bar_ends ------- #
                            try:
                                self.read_write_config_list(mode='r')
                            except Exception as e:
                                print("error in load config (waiting zone phase):", e)
                                time.sleep(1)
                                continue
    
                            # ------- 2. sys_log configuration reaction ------- #
                            try:
                                with open(base_cfg_copy_path, 'r') as sys_cfg:
                                    # self.sys_log = logging.getLogger()
                                    sys_log_cfg = EasyDict(json.load(sys_cfg))
                                    logging.config.dictConfig(sys_log_cfg)
                            except Exception as e:
                                print("error in load sys_log_cfg (waiting zone phase):", e)
                                time.sleep(self.config.trader_set.api_term)
                                continue
                                
                            # ------- 3. new config 기준으로 bank_module 설정 ------- #
                            # SubscriptionErrorUnexpected error 가 자주 발생해서 일단은 해당 phase 중지
                            # bank_module = BankModule(self.config)

                        load_new_df = 1
                        break  # return to load_new_df

                    else:
                        time.sleep(self.config.trader_set.realtime_term)  # <-- term for realtime market function
                        # time.sleep(self.config.trader_set.close_complete_term)   # <-- term for close completion => deprecated

                if expired:  # 1m_bar close 까지 기다리는 logic
                    break

            if expired:  # expired init 을 위한 continue
                continue

            first_iter = True  # 포지션 변경하는 경우, 필요함
            while 1:  # <-- loop for 'check order type change condition'

                min_balance_bool = self.get_balance_info(bank_module, first_iter, mode=self.config.trader_set.profit_mode)
                if min_balance_bool:
                    break
                self.sys_log.info('~ get balance time : %.5f' % (time.time() - start_ts))

                # ------ get tr_set ------ #
                tp, ep, out, open_side = bank_module.get_tpepout(open_side, res_df_open, res_df)  # Todo, 일단은, ep1 default 로 설정

                # ------ set pos_side & fake_order & open_price comparison ------ #
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

                # ------ get precision ------ #
                price_precision, quantity_precision = bank_module.get_precision()
                tp, ep, out, liqd_p = [bank_module.calc_with_precision(price_, price_precision) for price_ in [tp, ep, out, liqd_p]]  # includes half-dynamic tp

                self.sys_log.info('tp : {}'.format(tp))
                self.sys_log.info('ep : {}'.format(ep))
                self.sys_log.info('out : {}'.format(out))
                self.sys_log.info('liqd_p : {}'.format(liqd_p))
                self.sys_log.info('leverage : {}'.format(leverage))
                self.sys_log.info('~ tp ep out leverage set time : %.5f' % (time.time() - start_ts))

                if leverage is None:
                    self.sys_log.info("leverage_rejection occured.")
                    if not self.config.trader_set.backtrade:
                        time.sleep(60)  # time_term for consecutive retry.
                    break

                if not self.config.trader_set.backtrade:
                    while 1:
                        try:
                            bank_module.change_initial_leverage(symbol=self.config.trader_set.symbol, leverage=leverage)
                        except Exception as e:
                            self.sys_log.error('error in change_initial_leverage : {}'.format(e))
                            time.sleep(self.config.trader_set.api_term)
                            continue  # -->  ep market 인 경우에 조심해야함 - why ..?
                        else:
                            self.sys_log.info('leverage changed --> {}'.format(leverage))
                            break

                # ---------- calc. open_quantity ---------- #
                open_quantity = bank_module.calc_with_precision(self.available_balance / ep * leverage, quantity_precision)
                self.sys_log.info("open_quantity : {}".format(open_quantity))

                # ---------- open order ---------- #
                orderside_changed = False
                if not self.config.trader_set.backtrade and not fake_order:
                    # [ back & real-trade validation phase ]
                    order_info = (self.available_balance, leverage)
                    post_order_res, self.over_balance, res_code = bank_module.limit_order(self.config.ep_set.entry_type, open_side, pos_side,
                                                                                          ep, open_quantity, order_info)
                    if res_code:  # order deny exception, res_code = 0
                        break

                # ------------ execution wait time ------------ #
                # ------ 1. market : prevent close at open bar ------ #
                open_exec = 0
                if self.config.ep_set.entry_type == OrderType.MARKET:
                    if self.config.trader_set.backtrade:
                        res_df, _ = bank_module.get_new_df_onstream(self.streamer)
                    open_exec = 1

                # ------ 2. limit : check expired (= ei_k & breakout_qty) ------ #
                else:
                    if self.config.trader_set.backtrade:
                        while 1:
                            res_df, _ = bank_module.get_new_df_onstream(self.streamer)

                            c_i = self.config.trader_set.complete_index  # 단지, 길이 줄이기 위해 c_i 로 재선언하는 것뿐, phase 가 많지 않아 수정하지 않음.

                            # [ back & real-trade validation phase ]
                            # Todo, 일단은 expire_k1 default
                            if bank_module.check_ei_k_onbarclose_v2(res_df_open, res_df, c_i, c_i, open_side):  # e_j, tp_j
                                break

                            # ------ entry ------ #
                            if open_side == OrderSide.BUY:
                                if res_df['low'].to_numpy()[c_i] <= ep:
                                    open_exec = 1
                                    break
                            else:
                                if res_df['high'].to_numpy()[c_i] >= ep:
                                    open_exec = 1
                                    break
                    else:
                        first_exec_qty_check = 1
                        check_time = time.time()
                        while 1:
                            # [ back & real-trade validation phase ]
                            if bank_module.check_ei_k_v2(res_df_open, res_df, open_side):
                                break

                            # ------ entry ------ #
                            if fake_order:
                                # realtime_price = get_market_price_v2(self.sub_client)
                                realtime_price = bank_module.get_market_price_v2()
                                if open_side == OrderSide.BUY:
                                    if realtime_price <= ep:
                                        open_exec = 1
                                        break
                                else:
                                    if realtime_price >= ep:
                                        open_exec = 1
                                        break
                            else:
                                new_check_time = time.time()
                                if first_exec_qty_check or new_check_time - check_time >= self.config.trader_set.open_exec_check_term:
                                    open_exec = bank_module.check_open_exec_qty_v2(post_order_res, open_quantity)

                                    check_time = new_check_time
                                    if first_exec_qty_check:
                                        first_exec_qty_check = 0

                                if open_exec:
                                    break

                # ------ move to next bar validation - enough time for open_quantity be consumed ------ #
                if not self.config.trader_set.backtrade:
                    while 1:
                        datetime_now = datetime.now()
                        if datetime_now.timestamp() > datetime.timestamp(res_df.index[-1]):
                            break
                        else:
                            time.sleep(self.config.trader_set.realtime_term)  # <-- for realtime price function

                # ------ when, open order time expired or executed ------ #
                #        regardless to position exist, cancel open orders       #
                if not self.config.trader_set.backtrade and not fake_order:
                    # [ back & real-trade validation phase ]
                    open_exec_price_list, open_exec_qty = bank_module.cancel_order_list([post_order_res], self.config.ep_set.entry_type)
                else:
                    open_exec_price_list = [ep]
                    open_exec_qty = open_quantity if open_exec else 0

                if orderside_changed:  # 추후 side change 기능을 고려해 보류함.
                    first_iter = False
                    self.sys_log.info('orderside_changed : {}\n'.format(orderside_changed))
                    continue
                else:
                    break

            # ------ reject unacceptable amount of asset ------ #
            if min_balance_bool:
                continue

            if leverage is None:
                continue

            if open_exec_qty == 0.0:  # open_exec_qty 는 분명 정의됨
                self.income = 0
            else:
                self.sys_log.info('open order executed')
                self.sys_log.info("open_exec_price_list : {}".format(open_exec_price_list))
                real_balance = open_exec_price_list[0] * open_exec_qty
                self.sys_log.info("real_balance : {}".format(real_balance))  # define for pnl calc.

                # ------ save trade_log_dict ------ #
                trade_log_dict[str(res_df_open.index[self.config.trader_set.complete_index])] = [open_side, "open"]
                #    real_trade 의 경우, datetime.now()'s td 로 입력해야하는 것 아닌가
                #    + backtrade's en_ts log 의 경우, lastest_index 가 complete_index 로 (위에서 c_i 를 사용함)
                if not self.config.trader_set.backtrade:
                    str_ts = str(datetime_now)
                    trade_log_dict[str_ts.replace(str_ts.split(':')[-1], "59.999000")] = [ep, open_side, "entry"]
                else:
                    trade_log_dict[str(res_df.index[self.config.trader_set.complete_index])] = [ep, open_side, "entry"]

                with open(trade_log_fullpath, "wb") as dict_f:
                    pickle.dump(trade_log_dict, dict_f)
                    self.sys_log.info("entry trade_log_dict dumped !\n")

                # ------ set close side ------ #
                if open_side == OrderSide.BUY:
                    close_side = OrderSide.SELL
                else:
                    close_side = OrderSide.BUY

                # ------ param init. ------ #
                #   a. 아래의 조건문을 담을 변수가 필요함 - 병합 불가 (latest)
                #       i. => load_new_df2 는 1 -> 0 으로 변경됨
                use_new_df2 = 1
                # if not self.config.tp_set.static_tp or not self.config.out_set.static_out:
                #     use_new_df2 = 1    # for signal_out, dynamic_out & tp
                load_new_df2 = 1
                limit_tp = 1
                post_order_res_list = []
                # tp_executedQty = 0    # dynamic 미사용으로 invalid
                ex_dict = {}  # exist for ideal_ep
                tp_exec_price_list, out_exec_price_list = [], []
                cross_on = 0  # exist for signal_out (early_out)

                limit_done = 0
                prev_exec_tp_len = 0
                market_close_on = 0
                log_out = None
                load_new_df3 = 1  # 1 is default, use_new_df2 여부에 따라 사용 결정됨
                all_executed = 0
                check_time = time.time()  # for tp_execution check_term

                while 1:
                    if use_new_df2:
                        if load_new_df2:  # dynamic_out & tp phase
                            if self.config.trader_set.backtrade:
                                res_df, load_new_df2 = bank_module.get_new_df_onstream(self.streamer)
                            else:
                                res_df, load_new_df2 = bank_module.get_new_df(mode="CLOSE")

                            try:
                                res_df = self.public.sync_check(res_df, self.config, mode="CLOSE")
                                np_timeidx = np.array([intmin_np(date_) for date_ in res_df.index.to_numpy()])  # should be located af. row_slice
                                res_df = self.public.public_indi(res_df, self.config, np_timeidx, mode="CLOSE")
                                res_df = self.utils.enlist_tr(res_df, self.config, np_timeidx, mode="CLOSE", show_detail=False)

                            except Exception as e:
                                self.sys_log.error("error in sync_check / public_indi / utils_ (load_new_df2 phase) : {}".format(e))
                                load_new_df2 = 1
                                continue

                            tp, out, tp_series, out_series = bank_module.get_dynamic_tpout(res_df, open_side, tp, out)
                            #   series 형태는, dynamic_tp reorder 를 위한 tp change 여부를 확인하기 위함임

                    # --------- limit_tp on/offer --------- #
                    # ------ 1. 첫 limit_tp order 진행햐야하는 상태 ------ #
                    # if len(post_order_res_list) == 0:  #  fake_order 의 post_order_res_list 는 꾸준히 empty (solved)
                    #     limit_tp = 1
                    # else:
                    #     # ------ 2. dynamic_tp reorder ------ #  dynamic 미예정 (solved)
                    #     if not self.config.tp_set.static_tp:    # np.nan != np.nan
                    #         tp_series_np = tp_series.to_numpy()
                    #         if tp_series_np[self.config.trader_set.complete_index] != \
                    #                 tp_series_np[self.config.trader_set.complete_index - 1]:
                    #             #        1. 본래는, remaining 으로 open_exec_qty refresh 진행함,
                    #             #        2. 지금은, 직접 구해야할 것
                    #             #           a. refreshedQty = open_exec_qty - tp_executedQty : error 발생할때마다 recalc.
                    #             #        3. cancel post_order_res in list
                    #             dynamic_tp_executedPrice_list, tp_executedQty = cancel_order_list(self.config.trader_set.symbol, post_order_res_list)
                    #             limit_tp = 1

                    # ------- limit_tp order - while 내부에 있어서 on/off 로 진행 ------- #
                    if limit_tp:
                        # ------ 1. get realtime price, qty precision ------ #
                        price_precision, quantity_precision = bank_module.get_precision()
                        partial_tps, partial_qtys = bank_module.get_partial_tp_qty(ep, tp, open_exec_qty, price_precision, quantity_precision, close_side)

                        # ------ 2. partial_tps limit_order ------ #
                        if not self.config.trader_set.backtrade and not fake_order:
                            #   a. tp_exectuedQty 감산했던 이유 : dynamic_tp 의 경우 체결된 qty 제외
                            #   b. reduceOnly = False for multi_position

                            # [ back & real-trade validation phase ]
                            post_order_res_list = bank_module.partial_limit_order_v4(partial_tps, partial_qtys, close_side, pos_side,
                                                                                     open_exec_qty, quantity_precision)

                        limit_tp = 0

                    # ------------ limit_tp exec. & market_close check ------------ #
                    #            1. limit close (tp) execution check, every minute                 #
                    #            2. check market close signal, simultaneously
                    while 1:
                        # ------ 1. load_new_df3 every minutes ------ #
                        #           a. ohlc data, log_ts, back_pr wait_time 를 위해 필요함
                        if not use_new_df2:
                            if load_new_df3:
                                if self.config.trader_set.backtrade:
                                    res_df, load_new_df3 = bank_module.get_new_df_onstream(self.streamer)
                                else:
                                    res_df, load_new_df3 = bank_module.get_new_df(calc_rows=False, mode="CLOSE")

                        # ------ 2. tp execution check ------ #
                        if not self.config.trader_set.backtrade and not fake_order:
                            new_check_time = time.time()
                            # [ back & real-trade validation phase ]
                            if new_check_time - check_time >= self.config.trader_set.tp_exec_check_term:  # 0.25 second 로 설정시 연결 끊김. --> api limits 에 대한 document 참조할 것.
                                all_executed, tp_exec_price_list = bank_module.check_limit_tp_exec_v2(post_order_res_list,
                                                                                                      quantity_precision,
                                                                                                      return_price=True)
                                # dynamic_tp 안만듬 - 미예정임 (solved)
                                check_time = new_check_time
                            exec_tp_len = len(tp_exec_price_list)
                        else:
                            if not self.config.tp_set.non_tp:
                                if open_side == OrderSide.BUY:
                                    exec_tp_len = np.sum(res_df['high'].to_numpy()[self.config.trader_set.complete_index] >= np.array(partial_tps))
                                else:
                                    exec_tp_len = np.sum(res_df['low'].to_numpy()[self.config.trader_set.complete_index] <= np.array(partial_tps))

                                tp_exec_price_list = partial_tps[:exec_tp_len]  # 지속적 갱신
                                all_executed = 1 if exec_tp_len == len(partial_tps) else 0

                            # ------ a. tp execution logging ------ #
                        if not self.config.tp_set.non_tp:
                            if prev_exec_tp_len != exec_tp_len:  # logging 기준, 체결이 되면 prev_exec_tp_len != exec_tp_len
                                ex_dict[str(res_df.index[self.config.trader_set.complete_index])] = partial_tps[prev_exec_tp_len:exec_tp_len]
                                prev_exec_tp_len = exec_tp_len
                                self.sys_log.info("ex_dict : {}".format(ex_dict))

                            # ------ b. all_execution ------ #
                        if all_executed:
                            limit_done = 1
                            break

                        # ------ 3. out check ------ #
                        try:
                            # [ back & real-trade validation phase ]
                            if not self.config.trader_set.backtrade:
                                market_close_on, log_out = bank_module.check_hl_out_v2(res_df, market_close_on, log_out, out, liqd_p, open_side)
                            else:
                                market_close_on, log_out = bank_module.check_hl_out_onbarclose_v2(res_df, market_close_on, log_out, out, liqd_p, open_side)

                            if not market_close_on:  # log_out 갱신 방지
                                market_close_on, log_out, cross_on = bank_module.check_signal_out_v2(res_df, market_close_on, log_out, cross_on, open_side)

                            if market_close_on:
                                self.sys_log.info("market_close_on is True")
                                # ------ out execution logging ------ #
                                # market_close_on = True, log_out != None (None 도 logging 가능하긴함)
                                if not self.config.trader_set.backtrade:  # real_trade 의 경우, realtime_price's ts -> lastest_index 사용
                                    ex_dict[str(res_df.index[self.config.trader_set.latest_index])] = [log_out]  # insert as list type
                                else:
                                    ex_dict[str(res_df.index[self.config.trader_set.complete_index])] = [log_out]
                                self.sys_log.info("ex_dict : {}".format(ex_dict))
                                break

                        except Exception as e:
                            self.sys_log.error('error in checking market_close_on : {}'.format(e))
                            continue

                        # ------ 3. bar_end phase - loop selection ------ #
                        if not self.config.trader_set.backtrade:
                            if datetime.now().timestamp() > datetime.timestamp(res_df.index[-1]):   # database timestamp 형태에 따라 다소 변경될 여지가 있음.
                                if use_new_df2:
                                    load_new_df2 = 1  # return to outer loop - get df2's data
                                    break
                                else:
                                    load_new_df3 = 1  # return to current loop
                            # else:  # realtime_price 를 위해 realtime 으로 진행
                            #     time.sleep(self.config.trader_set.realtime_term)
                        else:
                            if use_new_df2:
                                load_new_df2 = 1  # return to outer loop - get df2's data
                                break  # get next(gen.)
                            else:
                                load_new_df3 = 1

                    # ------------ limit / market execution or load_new_df2 ------------ #
                    # ------ 1. all partial_tps executed ------ #
                    if limit_done:
                        fee += self.config.trader_set.limit_fee
                        break

                    if not market_close_on:  # = load_new_df2
                        continue
                    else:
                        # ------ 2. hl & signal_out ------ #
                        fee += self.config.trader_set.market_fee
                        if not self.config.trader_set.backtrade and not fake_order:
                            # [ back & real-trade validation phase ]
                            out_exec_price_list = bank_module.market_close_order_v2(post_order_res_list, close_side, pos_side, open_exec_qty)
                        else:
                            out_exec_price_list = [log_out]
                        break  # ---> break close order loop

                # ------------ calc_ideal_profit - market_order 시 ideal <-> real gap 발생 가능해짐 ------------ #
                ideal_profit, real_profit, trade_log_dict = bank_module.calc_ideal_profit_v3(res_df, open_side, ep, ex_dict, open_exec_price_list,
                                                                                             tp_exec_price_list, out_exec_price_list, fee,
                                                                                             trade_log_dict)

                with open(trade_log_fullpath, "wb") as dict_f:
                    pickle.dump(trade_log_dict, dict_f)
                    self.sys_log.info("exit trade_log_dict dumped !")

                # ------------ get total income from this trade ------------ #
                if not fake_order:
                    self.get_income_info(real_balance, leverage, ideal_profit, real_profit, mode=self.config.trader_set.profit_mode)

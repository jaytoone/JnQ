import os
import sys


def resource_path(relative_path, return_base_path=False):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    if return_base_path:
        return base_path
    return os.path.join(base_path, relative_path)


pkg_path = resource_path(__file__, return_base_path=True)
# pkg_path = r"C:\Users\Lenovo\PycharmProjects\System_Trading\JnQ\scalper"  # Todo, system env. 에 따라 가변적 + check key_abspath in test.py
os.chdir(pkg_path)

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic
# from PyQt5.QtGui import *

from funcs_binance.binance_futures_modules import *  # math, pandas, bot_config (API_key & clients)
from funcs_binance.funcs_trader_modules_scalper import init_set, get_p_tpqty
from funcs_binance.funcs_order_logger_hedge_scalper import limit_order, partial_limit_order_v4, cancel_order_list, market_close_order_v2

import logging.config
# from pathlib import Path
from easydict import EasyDict
import shutil
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


# ----------- UI set -----------#
form = uic.loadUiType(resource_path("scalper_v.ui"))[0]
# form = uic.loadUiType("scalper.ui")[0]


class OrderWindow(QMainWindow, form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("scalper")
        self.setWindowFlags(Qt.WindowStaysOnTopHint)

        ag = QDesktopWidget().availableGeometry()
        sg = QDesktopWidget().screenGeometry()

        widget = self.geometry()
        # x = ag.width() - widget.width()
        # y = 2 * ag.height() - sg.height() - widget.height()
        y = (sg.height() - widget.height()) / 2
        self.move(0, y)

        # self.pkg_path = r"C:\Users\Lenovo\PycharmProjects\System_Trading\JnQ\scalper"  # Todo, system env. 에 따라 가변적 + check key_abspath in test.py
        # self.pkg_path = resource_path(__file__, return_base_path=True)  # Todo, system env. 에 따라 가변적 + check key_abspath in test.py
        self.pkg_path = pkg_path  # Todo, system env. 에 따라 가변적 + check key_abspath in test.py
        # os.chdir(self.pkg_path)

        self.ticker = "ETHUSDT"     # default
        self.fee = 0.0006           # all trading sequence's fee
        self.p_ranges = "[1]"
        self.p_qty_ratio = "[1]"

        self.sub_client = None
        self.sys_log = None

        # ------ window_set ------ #
        self.label_status.setWordWrap(True)

        self.lineEdit_ticker.setText(self.ticker)  # default
        self.lineEdit_loss_volume.setText(str(50))  # default
        self.lineEdit_limit_lvrg.setText(str(10))  # default

        self.param_init()
        self.sys_log_set()

        # self.lineEdit_coin.returnPressed.connect(lambda: self.ticker_enter())


        # self.lineEdit_ep.returnPressed.connect(lambda: self.enlist_ep())
        # self.lineEdit_loss_volume.returnPressed.connect(lambda: self.exit_clicked())

        # ------ executed_quantity & pos's pnl thread ------ #
        self.thread = StatusThread(self)
        self.thread.start()

        # ------------ click options ------------ #
        # self.pushButton_init.clicked.connect(lambda: self.init_click())

        self.pushButton_long.clicked.connect(lambda: self.long_click())
        self.pushButton_short.clicked.connect(lambda: self.short_click())

        # ------ market order ------ #
        self.pushButton_market.clicked.connect(lambda: self.market_click())

        # ------------ enter options ------------ #
        # ------ limit_tp order ------ #
        self.lineEdit_tp.returnPressed.connect(lambda: self.tp_enter())

    def param_init(self):
        self.tp = None
        self.open_side = None
        self.close_side = None
        self.pos_side = None
        self.over_balance = None
        self.open_quantity = 0
        self.open_executedQty = 0
        self.open_exec_ratio = 0
        self.close_exec_ratio = 0
        self.post_order_res = None
        self.post_order_res_list = [None]
        self.available_balance = get_availableBalance()
        self.label_status.setText("Asset : {} USDT".format(self.available_balance))

    # def init_click(self):
    #     if self.tp is not None and self.exec_ratio > 0.99:  # = trade end (close 거래 완료시에만 init 가능하도록)
    #         self.param_init()

    def sys_log_set(self):

        # ------ copy base_cfg.json -> {ticker}.json (log_cfg name) ------ #
        exec_file_name = str(datetime.now().timestamp())
        sys_log_path = os.path.join(self.pkg_path, "sys_log")
        src_cfg_path = os.path.join(sys_log_path, "base_cfg.json")
        dst_cfg_path = os.path.join(sys_log_path, "{}.json".format(exec_file_name))

        try:
            shutil.copy(src_cfg_path, dst_cfg_path)
        except Exception as e:
            print("error in shutil.copy() :", e)
            quit()

        try:
            with open(dst_cfg_path, 'r') as sys_cfg:  # trade end 시에도 반영하기 위함 - 윗 phase 와 분리 이유
                sys_log_cfg = EasyDict(json.load(sys_cfg))

            sys_log_cfg.handlers.file_rot.filename = \
                os.path.join(sys_log_path, "{}.log".format(exec_file_name))  # log_file name - trade_log_name 에 종속
            logging.getLogger("apscheduler.executors.default").propagate = False

            with open(dst_cfg_path, 'w') as edited_cfg:
                json.dump(sys_log_cfg, edited_cfg, indent=3)

            logging.config.dictConfig(sys_log_cfg)
            self.sys_log = logging.getLogger()
            self.sys_log.info('# ----------- {} ----------- #'.format(exec_file_name))
            self.sys_log.info("self.pkg_path : {}".format(self.pkg_path))

            _, self.sub_client = init_set(self)    # self.limit_leverage, self.sub_client

            self.sys_log.info("initial_set done\n")

        except Exception as e:
            print("error in load sys_log_cfg :", e)


    def lvrg_set(self):

        # Todo,
        # loss_volume calc. --> max_balance 사용 + lvrg 최대한 낮추기.
        #   1. balance call
        # self.available_balance = get_availableBalance()

        #   2. calc. lvrg
        self.ticker = self.lineEdit_ticker.text()
        price_precision, quantity_precision = get_precision(self.ticker)

        # ------ a. validation ------ #
        if self.lineEdit_loss_volume.text() != "":
            self.loss_volume = float(self.lineEdit_loss_volume.text())   # USDT
        else:
            self.label_status.setText("invalid loss_volume")
            return 0
        if self.lineEdit_lsp.text() != "":
            self.lsp = calc_with_precision(float(self.lineEdit_lsp.text()), price_precision)
        else:
            self.label_status.setText("invalid lsp")
            return 0
        if self.lineEdit_ep.text() != "":
            self.ep = calc_with_precision(float(self.lineEdit_ep.text()), price_precision)
        else:
            self.label_status.setText("invalid ep")
            return 0

        if self.open_side == OrderSide.SELL:
            # --- lsp & ep validation --- #
            if self.lsp <= self.ep:
                status_msg = "lsp & ep validation \n lsp : {} \n ep : {}".format(self.lsp, self.ep)
                self.sys_log.error(status_msg)
                self.label_status.setText(status_msg)
                return 0
            loss_range = (self.lsp / self.ep)
        else:
            if self.lsp >= self.ep:
                status_msg = "lsp & ep validation \n lsp : {} \n ep : {}".format(self.lsp, self.ep)
                self.sys_log.error(status_msg)
                self.label_status.setText(status_msg)
                return 0
            loss_range = (self.ep / self.lsp)

        self.leverage = self.loss_volume / ((loss_range - 1 - self.fee) * self.available_balance)

        if self.leverage < 1:
            self.leverage = 1
            self.available_balance = self.loss_volume / (loss_range - 1 - self.fee)
        else:
            self.leverage = int(self.leverage)

        self.limit_lvrg = int(self.lineEdit_limit_lvrg.text())
        if self.leverage < self.limit_lvrg:
            try:
                request_client.change_initial_leverage(symbol=self.ticker, leverage=self.leverage)
            except Exception as e:
                self.sys_log.error('error in change_initial_leverage : {}'.format(e))
            else:
                status_msg = "leverage changed --> {}".format(self.leverage)
                self.label_status.setText(status_msg)
                self.sys_log.info(status_msg)

        else:
            status_msg = "leverage rejection occured. \n self.limit_lvrg : {}".format(self.limit_lvrg)
            self.label_status.setText(status_msg)
            self.sys_log.warning(status_msg)
            return 0

        # ---------- calc. self.open_quantity ---------- #
        self.open_quantity = calc_with_precision(self.available_balance / self.ep * self.leverage, quantity_precision)

    def long_click(self):
        if self.post_order_res is not None:  # all open_order should be closed / reject order duplication (order traveling not available)
            if get_order_info(self.ticker, self.post_order_res.orderId).status != "CANCELED" or self.open_executedQty != 0:
                return

        self.open_side = OrderSide.BUY
        self.close_side = OrderSide.SELL
        self.pos_side = PositionSide.LONG

        if self.lvrg_set() == 0:
            return

        order_info = (self.available_balance, self.leverage)
        self.post_order_res, self.over_balance, res_code = limit_order(self, "LIMIT", self.open_side, self.pos_side,
                                                                  self.ep, self.open_quantity, order_info)
        if self.post_order_res is None:
            return

        status_msg = "long open_order enlisted. \n loss_volume : {}\n lsp : {} \n ep : {} \n leverage : {} \n self.open_quantity : {}".format(self.loss_volume, self.lsp,
                                                                                                 self.ep, self.leverage, self.open_quantity)
        self.label_status.setText(status_msg)
        self.sys_log.warning(status_msg)

    def short_click(self):
        if self.post_order_res is not None:  # all open_order should be closed / reject order duplication (order traveling not available)
            if get_order_info(self.ticker, self.post_order_res.orderId).status != "CANCELED" or self.open_executedQty != 0:
                return

        self.open_side = OrderSide.SELL
        self.close_side = OrderSide.BUY
        self.pos_side = PositionSide.SHORT

        if self.lvrg_set() == 0:
            return

        order_info = (self.available_balance, self.leverage)
        self.post_order_res, self.over_balance, res_code = limit_order(self, "LIMIT", self.open_side, self.pos_side,
                                                              self.ep, self.open_quantity, order_info)
        if self.post_order_res is None:
            return

        status_msg = "short open_order enlisted. \n loss_volume : {}\n lsp : {} \n ep : {} \n leverage : {} \n self.open_quantity : {}".format(self.loss_volume, self.lsp,
                                                                                                 self.ep, self.leverage, self.open_quantity)
        self.label_status.setText(status_msg)
        self.sys_log.warning(status_msg)

    def tp_enter(self):

        # 1. cancel order
        _, self.open_executedQty = cancel_order_list(self.ticker, [self.post_order_res], "LIMIT")

        status_msg = "cancel order successed. \n self.open_executedQty : {}".format(self.open_executedQty)
        self.label_status.setText(status_msg)
        self.sys_log.info(status_msg)

        # 2. limit_tp order #=> criteria : positionAmt
        # if get_position_info(self.ticker).positionAmt == 0.0:
        if self.open_executedQty == 0.0:
            return

        # if self.lineEdit_tp.text() == "":
        #     return

        # ------ a. get realtime price, qty precision ------ #
        if self.lineEdit_tp.text() != "":
            price_precision, quantity_precision = get_precision(self.ticker)
            self.tp = calc_with_precision(float(self.lineEdit_tp.text()), price_precision)

            # --- tp validation --- # => limit_loss 도 가능함
            # if self.open_side == OrderSide.BUY:
            #     if self.tp < self.ep:
            #         status_msg = "tp validation \n tp, ep : {}, {}".format(self.tp, self.ep)
            #         self.sys_log.error(status_msg)
            #         self.label_status.setText(status_msg)
            #         return
            # else:
            #     if self.tp > self.ep:
            #         status_msg = "tp validation \n tp, ep : {}, {}".format(self.tp, self.ep)
            #         self.sys_log.error(status_msg)
            #         self.label_status.setText(status_msg)
            #         return

            p_tps, p_qtys = get_p_tpqty(self, price_precision, quantity_precision)

            # ------ b. p_tps limit_order ------ #
            # while 1:
            try:
                #   a. tp_exectuedQty 감산했던 이유 : dynamic_tp 의 경우 체결된 qty 제외
                #   b. reduceOnly = False for multi_position
                self.post_order_res_list = partial_limit_order_v4(self, p_tps, p_qtys, quantity_precision)
            except Exception as e:
                status_msg = "error in partial_limit_order() : {}".format(e)
                self.label_status.setText(status_msg)
                self.sys_log.error(status_msg)
            else:
                status_msg = "limit tp order enlisted. \n p_tps : {} \n p_qtys {}".format(p_tps, p_qtys)
                self.label_status.setText(status_msg)
                self.sys_log.info(status_msg)
                # break


    def market_click(self):
        market_close_order_v2(self)



class StatusThread(QThread):

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

    def run(self):

        while 1:
            exec_ratio = 0
            pnl = 0
            expected_pnl = 0

            # ------ get executed_ratio ------ # post_order_res 을 reset 하면 추적이 끊김.
            if self.parent.post_order_res is not None:  # = open_order enlisted once
                if self.parent.tp is not None:   # = tp_order enlisted once (tp default : None)
                    order_info_list = [get_order_info(self.parent.ticker, post_order_res.orderId)
                                       for post_order_res in self.parent.post_order_res_list if post_order_res is not None]
                    order_exec_ratio_list = [get_exec_ratio(order_info) for order_info in order_info_list]
                    self.parent.close_exec_ratio = sum(order_exec_ratio_list) / len(order_exec_ratio_list)
                    exec_ratio = self.parent.close_exec_ratio

                    # ------ expected_pnl ------ #
                    if self.parent.open_side == OrderSide.BUY:
                        expected_pnl = (self.parent.tp / self.parent.ep - self.parent.fee - 1) * self.parent.open_executedQty * self.parent.ep
                    else:
                        expected_pnl = (self.parent.ep / self.parent.tp - self.parent.fee - 1) * self.parent.open_executedQty * self.parent.tp
                else:
                    open_order_info = get_order_info(self.parent.ticker, self.parent.post_order_res.orderId)
                    self.parent.ep = get_execPrice(open_order_info)
                    self.parent.open_executedQty = get_execQty(open_order_info)
                    self.parent.open_exec_ratio = get_exec_ratio(open_order_info)
                    exec_ratio = self.parent.open_exec_ratio

                # --- trade end --- #
                if self.parent.close_exec_ratio > 0.99:
                    self.parent.param_init()  # param_init 은 run() phase 에서만 진행 (오류 방지)
                    continue

                # ------ realtime_pnl ------ #
                market_price = get_market_price_v2(self.parent.sub_client)
                if self.parent.open_side == OrderSide.BUY:
                    pnl = (market_price / self.parent.ep - self.parent.fee - 1) * self.parent.open_executedQty * self.parent.ep
                else:
                    pnl = (self.parent.ep / market_price - self.parent.fee - 1) * self.parent.open_executedQty * market_price

            self.parent.label_realtime.setText("exec_ratio : {:.0f}% \n PnL : {:.2f} USDT \n "
                                               "exp_PnL : {:.2f} USDT".format(exec_ratio * 100, pnl, expected_pnl))
            time.sleep(0.2)




if __name__=="__main__":

    app = QApplication(sys.argv)
    window = OrderWindow()
    window.show()
    app.exec_()





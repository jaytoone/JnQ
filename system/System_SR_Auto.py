import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic
# from PyQt5.QtGui import *
import time
from datetime import datetime
import random
import pybithumb
from Funcs_For_Trade import *
import warnings

warnings.filterwarnings("ignore")

# ----------- KEY SETTING -----------#
with open("Keys.txt") as f:
    lines = f.readlines()
    key = lines[0].strip()
    secret = lines[1].strip()
    bithumb = pybithumb.Bithumb(key, secret)

# ----------- UI SETTING -----------#
form = uic.loadUiType("SR_ui_krw.ui")[0]


class OrderWindow(QMainWindow, form):
    #       VARIABLE DECLARTION     #
    Coin = None
    buy_wait = 30  # minute
    profits = 1.
    Orderinfo = None
    krw = None
    krw_value = 0
    fee = 0.005

    #   PRICE   #
    limit_buy_price = None
    limit_sell_price = None
    limit_exit_price = None

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Bithumb Trader")
        self.setGeometry(0, 0, 450, 340)

        self.timer_buy = QTimer(self)
        self.timer_buy.start(1000)
        self.timer_buy.timeout.connect(self.realtime)

        # 입력값 # 문자열 입력에 대한 에러처리를 해주어야한다.
        self.slider_krw.valueChanged.connect(lambda: self.select_krw())
        self.lineEdit_coin.returnPressed.connect(lambda: self.coin_clicked())
        self.lineEdit_buy.returnPressed.connect(lambda: self.buy_clicked())
        self.lineEdit_sell.returnPressed.connect(lambda: self.sell_clicked())
        self.lineEdit_exit.returnPressed.connect(lambda: self.exit_clicked())

        #      자동 거래를 위한 쓰레드 생성     #
        self.thread = TradeThread(parent=self)

        # Button
        #           자동 주문 개시        #
        self.pushButton_trade.clicked.connect(lambda: self.trade_clicked())
        #           주문 취소            #
        self.pushButton_cancel.clicked.connect(lambda: self.cancel_clicked())

    def realtime(self):
        current = QTime.currentTime()
        realtime = current.toString("hh:mm:ss")

        try:
            balance = bithumb.get_balance(self.Coin)
            current_price = pybithumb.get_current_price(self.Coin)
            total_krw = balance[0] * current_price + balance[2]
            self.statusBar().showMessage("실시간 보유 자산 : %.f KRW / 시간 : %s" % (total_krw, realtime))

        except Exception as e:
            pass

    def select_krw(self):

        try:
            self.krw_value = self.slider_krw.value() / 100

            balance = bithumb.get_balance(self.Coin)
            self.statusBar().showMessage("배팅 금액 : %.f KRW" % (balance[2] * self.krw_value))

        except Exception as e:
            print('Error in select krw :', e)
            pass

    def buy_clicked(self):  # 엔터치면 매수 동작하는 걸로 가능? = returnPressed()
        print("매수가 등록")

        #           Initializing Sell, Exit price           #
        self.limit_sell_price = None
        self.limit_exit_price = None

        # 매수 동작
        try:
            # self.timer_buy.stop()
            # self.timer_buy.start(1000)
            # self.timer_buy.timeout.connect(lambda: self.hogachart(Coin))

            self.limit_buy_price = clearance(float(self.lineEdit_buy.text()))
            self.lineEdit_status.setText("%s KRW 매수가 등록" % self.limit_buy_price)

        except Exception as e:
            self.lineEdit_status.setText(str(e))
            print(e)
            print()

    def sell_clicked(self):
        print("수익률 등록")

        try:
            # self.timer_buy.stop()
            # self.timer_sell.stop()
            # self.timer_sell.start(1000)
            # self.timer_sell.timeout.connect(lambda: self.hogachart(Coin))

            # ---------------- 매도 등록 ----------------#
            self.limit_sell_price = clearance(float(self.lineEdit_sell.text()))
            if self.limit_buy_price != 0:
                self.lineEdit_status.setText(
                    "%.2f%% 수익률 등록" % (self.limit_sell_price / self.limit_buy_price * 100 - 100 - self.fee * 100))
            else:
                self.lineEdit_status.setText("%s KRW 매도가 등록" % self.limit_sell_price)

            time.sleep(1)

        except Exception as e:
            print("매도가 등록 중 에러 발생!")
            self.lineEdit_status.setText(str(e))
            print(e)
            print()

    def exit_clicked(self):
        print("손절가율 등록")

        try:
            self.limit_exit_price = clearance(float(self.lineEdit_exit.text()))

            if self.limit_buy_price != 0:
                self.lineEdit_status.setText(
                    "%.2f%% 손절가율 등록" % ((self.limit_exit_price / self.limit_buy_price - 1.) * 100 - self.fee * 100))
            else:
                self.lineEdit_status.setText("%s KRW 손절가 등록" % self.limit_exit_price)

            time.sleep(1)

        except Exception as e:
            print("손절가율 등록 중 에러 발생!")
            self.lineEdit_status.setText(str(e))
            print(e)
            print()

    def cancel_clicked(self):
        print("주문 취소 동작")

        # 미체결, 체결 완료 정보를 받아서 매수/매도 주문 잔량을 모두 취소한다.
        try:
            CancelOrder = bithumb.cancel_order(self.Orderinfo)
            print("주문 취소 : ", CancelOrder)
            print()

        except Exception as e:
            print("Error in cancel")
            self.lineEdit_status.setText(str(e))
            print(e)
            print()

    def coin_clicked(self):
        print("코인 입력 동작")

        try:
            self.Coin = self.lineEdit_coin.text().upper()
            self.lineEdit_status.setText("%s 코인 선택" % self.Coin)

        except Exception as e:
            self.lineEdit_status.setText(str(e))
            print("입력 오류!")
            print(e)
            print()

    def trade_clicked(self):
        self.thread.start()


class TradeThread(QThread):

    def __init__(self, parent):
        super(TradeThread, self).__init__(parent)
        self.thread_parent = parent

    def run(self):

        var_list = ['코인', '매수가', '매도가']
        for i, var in enumerate(
                [self.thread_parent.Coin, self.thread_parent.limit_buy_price, self.thread_parent.limit_sell_price]):
            if var is None:
                self.thread_parent.lineEdit_status.setText("%s를 입력해주세요." % var_list[i])
                return

        #       매수 / 매도 등록 코드       #
        if self.thread_parent.limit_buy_price != 0:

            #                매수 등록                  #
            try:
                # 호가단위, 매수가, 수량

                # -------------- 보유 원화 확인 --------------#
                balance = bithumb.get_balance(self.thread_parent.Coin)
                start_coin = balance[0]
                krw = balance[2]
                self.thread_parent.krw = krw
                print()
                print("보유 원화 :", krw, end=' ')
                # invest_krw = 3000
                invest_krw = krw * self.thread_parent.krw_value
                print("주문 원화 :", invest_krw)
                if krw < 1000:
                    print("거래가능 원화가 부족합니다.\n")
                    self.thread_parent.lineEdit_status.setText("거래가능 원화가 부족합니다.")
                    return
                print()

                # 매수량
                buyunit = int((invest_krw / self.thread_parent.limit_buy_price) * 10000) / 10000.0

                # 매수 등록
                BuyOrder = bithumb.buy_limit_order(self.thread_parent.Coin, self.thread_parent.limit_buy_price, buyunit,
                                                   "KRW")
                print("    %s %s KRW 매수 등록    " % (self.thread_parent.Coin, self.thread_parent.limit_buy_price),
                      end=' ')
                print(BuyOrder)
                self.thread_parent.Orderinfo = BuyOrder
                self.thread_parent.lineEdit_status.setText(
                    "%s %s KRW 매수 등록" % (self.thread_parent.Coin, self.thread_parent.limit_buy_price))

            except Exception as e:
                print("매수 등록 중 에러 발생 :", e)
                self.thread_parent.lineEdit_status.setText("매수 등록 중 에러 발생 : %s" % e)
                return

            #                   매수 대기                    #

            start = time.time()
            Complete = 0
            while True:
                try:
                    #       BuyOrder is dict / in Error 걸러주기        #
                    #       체결되어도 buy_wait은 기다려야한다.
                    if type(BuyOrder) != tuple:
                        balance = bithumb.get_balance(self.thread_parent.Coin)
                        if (balance[0] - start_coin) * self.thread_parent.limit_buy_price > 1000:
                            pass
                        else:
                            print('BuyOrder is not tuple')
                            self.thread_parent.lineEdit_status.setText('BuyOrder is not tuple')
                            break

                    #   매수가 취소된 경우를 걸러주어야 한다.
                    else:
                        if bithumb.get_outstanding_order(BuyOrder) is None:
                            balance = bithumb.get_balance(self.thread_parent.Coin)
                            #   체결량이 존재하는 경우는 buy_wait 까지 기다린다.
                            if (balance[0] - start_coin) * self.thread_parent.limit_buy_price > 1000:
                                pass
                            else:
                                print("매수가 취소되었습니다.")
                                self.thread_parent.lineEdit_status.setText('매수가 취소되었습니다.')

                                #       TERM FOR PRINTING ABOVE MESSAGE     #
                                time.sleep(0.5)
                                remain = self.thread_parent.krw - balance[2]
                                print("거래해야할 원화 : ", remain)
                                print()
                                self.thread_parent.lineEdit_status.setText("거래해야할 원화 : %.f KRW" % remain)
                                break

                    self.thread_parent.lineEdit_status.setText(
                        "실시간 체결량 : %.2f %%" % ((balance[0] - start_coin) / buyunit * 100))

                    #   반 이상 체결된 경우 : 체결된 코인량이 매수 코인량의 반 이상인경우
                    if (balance[0] - start_coin) / buyunit >= 0.8:
                        Complete = 1
                        #   부분 체결되지 않은 미체결 잔량을 주문 취소
                        print("    매수 체결    ", end=' ')
                        CancelOrder = bithumb.cancel_order(BuyOrder)
                        print("부분 체결 :", CancelOrder)
                        print()
                        self.thread_parent.lineEdit_status.setText('매수 체결 | 부분 체결 : %s' % CancelOrder)
                        time.sleep(1 / 80)
                        break

                    #   buy_wait 동안 매수 체결을 대기한다.    #
                    if time.time() - start > 60 * self.thread_parent.buy_wait:

                        balance = bithumb.get_balance(self.thread_parent.Coin)
                        if (balance[0] - start_coin) * self.thread_parent.limit_buy_price > 1000:
                            Complete = 1
                            print("    매수 체결    ")
                            CancelOrder = bithumb.cancel_order(BuyOrder)
                            self.thread_parent.lineEdit_status.setText('매수 체결 | 부분 체결 : %s' % CancelOrder)

                        else:
                            # outstanding >> None 출력되는 이유 : BuyOrder == None, 체결 완료
                            if bithumb.get_outstanding_order(BuyOrder) is None:
                                if type(BuyOrder) == tuple:
                                    # 한번 더 검사 (get_outstanding_order 찍으면서 체결되는 경우가 존재한다.)
                                    if (balance[0] - start_coin) * self.thread_parent.limit_buy_price > 1000:
                                        Complete = 1
                                        print("    매수 체결    ")
                                        CancelOrder = bithumb.cancel_order(BuyOrder)
                                        self.thread_parent.lineEdit_status.setText('매수 체결 | 부분 체결 : %s' % CancelOrder)

                                else:
                                    # BuyOrder is None 에도 체결된 경우가 존재함
                                    # 1000 원은 거래 가능 최소 금액
                                    if (balance[0] - start_coin) * self.thread_parent.limit_buy_price > 1000:
                                        Complete = 1
                                        print("    매수 체결    ")
                                        CancelOrder = bithumb.cancel_order(BuyOrder)
                                        self.thread_parent.lineEdit_status.setText('매수 체결 | 부분 체결 : %s' % CancelOrder)

                                    else:
                                        print("미체결 또는 체결량 1000 KRW 이하\n")
                                        print()
                                        self.thread_parent.lineEdit_status.setText("미체결 또는 체결량 1000 KRW 이하")
                            else:
                                if type(BuyOrder) == tuple:
                                    CancelOrder = bithumb.cancel_order(BuyOrder)
                                    print("미체결 또는 체결량 1000 KRW 이하")
                                    print(CancelOrder)
                                    print()
                                    self.thread_parent.lineEdit_status.setText("미체결 또는 체결량 1000 KRW 이하")
                        break

                except Exception as e:
                    print('매수 체결 여부 확인중 에러 발생 :', e)

        else:
            Complete = 1

        # 지정 시간 초과로 루프 나온 경우
        if Complete == 0:
            print()
            return
        #                    매수 체결                     #

        else:

            #           매도 대기           #
            while True:

                try:
                    #   매도 진행
                    balance = bithumb.get_balance(self.thread_parent.Coin)
                    sellunit = int((balance[0]) * 10000) / 10000.0
                    SellOrder = bithumb.sell_limit_order(self.thread_parent.Coin, self.thread_parent.limit_sell_price,
                                                         sellunit, 'KRW')
                    print("    %s 지정가 매도     " % self.thread_parent.Coin, end=' ')
                    # print("    %s 시장가 매도     " % self.thread_parent.Coin, end=' ')
                    # SellOrder = bithumb.sell_limit_order(self.thread_parent.Coin, limit_sell_pricePlus, sellunit, "KRW")
                    print(SellOrder)
                    self.thread_parent.Orderinfo = SellOrder
                    self.thread_parent.lineEdit_status.setText(
                        "%s %s KRW 매도 등록" % (self.thread_parent.Coin, self.thread_parent.limit_sell_price))
                    break

                except Exception as e:
                    print('Error in %s high predict :' % self.thread_parent.Coin, e)

            sell_switch = 0
            exit_count = 0
            while True:

                #                   매도 재등록                    #
                try:
                    if sell_switch == 1:

                        #   SellOrder Initializing
                        CancelOrder = bithumb.cancel_order(SellOrder)
                        if CancelOrder is False:  # 남아있는 매도 주문이 없다. 취소되었거나 체결완료.
                            #       체결 완료       #
                            if bithumb.get_order_completed(SellOrder)['data']['order_status'] != 'Cancel':
                                print("    매도 체결    ", end=' ')
                                self.thread_parent.lineEdit_status.setText("매도 체결")

                                if self.thread_parent.limit_buy_price != 0:
                                    #               매도가 / 매수가 - 수수료             #
                                    self.thread_parent.profits *= \
                                        self.thread_parent.limit_sell_price / self.thread_parent.limit_buy_price - self.thread_parent.fee
                                    print("Accumulated Profits : %.6f\n" % self.thread_parent.profits)
                                break
                            #       취소      #
                            else:
                                pass
                        elif CancelOrder is None:  # SellOrder = none 인 경우
                            # 매도 재등록 해야함 ( 등록될 때까지 )
                            pass
                        else:
                            print("    매도 취소    ", end=' ')
                            print(CancelOrder)
                        print()

                        balance = bithumb.get_balance(self.thread_parent.Coin)
                        sellunit = int((balance[0]) * 10000) / 10000.0
                        if exit_count != -1:
                            SellOrder = bithumb.sell_limit_order(self.thread_parent.Coin,
                                                                 self.thread_parent.limit_sell_price, sellunit,
                                                                 "KRW")
                            print("    %s 지정가 매도     " % self.thread_parent.Coin, end=' ')
                            self.thread_parent.lineEdit_status.setText(
                                "%s %s KRW 매도 등록" % (self.thread_parent.Coin, self.thread_parent.limit_sell_price))
                        else:
                            while True:
                                if datetime.now().second >= 55:
                                    break
                            SellOrder = bithumb.sell_market_order(self.thread_parent.Coin, sellunit, 'KRW')
                            print("    %s 시장가 매도     " % self.thread_parent.Coin, end=' ')

                            if type(SellOrder) is str:
                                self.thread_parent.lineEdit_status.setText("%s 시장가 매도" % self.thread_parent.Coin)
                                SellOrder = ('ask', self.thread_parent.Coin, SellOrder, 'KRW')

                        print(SellOrder)
                        self.thread_parent.Orderinfo = SellOrder

                        sell_switch = -1
                        time.sleep(1 / 80)  # 너무 빠른 거래로 인한 오류 방지

                        # 지정 매도 에러 처리
                        if type(SellOrder) in [tuple, str]:
                            pass
                        elif SellOrder is None:  # 매도 함수 자체 오류 ( 서버 문제 같은 경우 )
                            sell_switch = 1
                            continue
                        else:  # dictionary
                            # 체결 여부 확인 로직
                            ordersucceed = bithumb.get_balance(self.thread_parent.Coin)
                            #   매도 주문 넣었을 때의 잔여 코인과 거래 후 코인이 다르고 사용중인 코인이 없으면 매도 체결
                            #   거래 중인 코인이 있으면 매도체결이 출력되지 않는다.
                            if ordersucceed[0] != balance[0] and ordersucceed[1] == 0.0:
                                print("    매도 체결    ", end=' ')
                                self.thread_parent.lineEdit_status.setText("매도 체결")
                                if self.thread_parent.limit_buy_price != 0:
                                    self.thread_parent.profits *= \
                                        self.thread_parent.limit_sell_price / self.thread_parent.limit_buy_price - self.thread_parent.fee
                                    print("Accumulated Profits : %.6f\n" % self.thread_parent.profits)
                                    self.thread_parent.lineEdit_status.setText(
                                        "Accumulated Profits : %.6f\n" % self.thread_parent.profits)
                                break
                            else:
                                sell_switch = 1
                                continue

                except Exception as e:
                    print("매도 재등록 중 에러 발생 :", e)
                    self.thread_parent.lineEdit_status.setText("매도 재등록 중 에러 발생 : %s" % e)
                    sell_switch = 1
                    continue

                #       매도 상태 Check >> 취소 / 체결완료 / SellOrder = None / dictionary      #
                try:
                    if bithumb.get_outstanding_order(SellOrder) is None:
                        # 서버 에러에 의해 None 값이 발생할 수 도 있음..
                        if type(SellOrder) in [tuple, str]:  # 서버에러는 except 로 가요..
                            try:
                                # 체결 여부 확인 로직
                                ordersucceed = bithumb.get_balance(self.thread_parent.Coin)
                                if ordersucceed[0] != balance[0] and ordersucceed[1] == 0.0:
                                    print("    매도 체결    ", end=' ')
                                    self.thread_parent.lineEdit_status.setText("매도 체결")
                                    if self.thread_parent.limit_buy_price != 0:
                                        self.thread_parent.profits *= \
                                            self.thread_parent.limit_sell_price / self.thread_parent.limit_buy_price - self.thread_parent.fee
                                        print("Accumulated Profits : %.6f\n" % self.thread_parent.profits)
                                        self.thread_parent.lineEdit_status.setText(
                                            "Accumulated Profits : %.6f\n" % self.thread_parent.profits)
                                    break
                                elif bithumb.get_outstanding_order(SellOrder) is not None:  # 혹시 모르는 미체결
                                    continue
                                else:
                                    print("매도 주문이 취소되었습니다.\n")
                                    self.thread_parent.lineEdit_status.setText("매도 주문이 취소되었습니다.")

                                    if sell_switch in [0, -1]:
                                        sell_switch = 1
                                        # elif sell_switch == 0:
                                        #     sppswitch = 1
                                        time.sleep(random.random() * 5)
                                    continue

                            except Exception as e:
                                print('SellOrder in [tuple, str] ? 에서 에러발생 :', e)
                                self.thread_parent.lineEdit_status.setText(
                                    'SellOrder in [tuple, str] ? 에서 에러발생 : %s' % e)

                                time.sleep(random.random() * 5)  # 서버 에러인 경우
                                continue

                        # 매도 등록 에러라면 ? 제대로 등록 될때까지 재등록 ! 지정 에러, 하향 에러
                        elif SellOrder is None:  # limit_sell_order 가 아예 안되는 경우
                            if sell_switch in [0, -1]:
                                sell_switch = 1
                                # elif sell_switch == 0:
                                #     sppswitch = 1
                                time.sleep(random.random() * 5)
                            continue

                        else:  # dictionary
                            # 체결 여부 확인 로직
                            ordersucceed = bithumb.get_balance(self.thread_parent.Coin)
                            if ordersucceed[0] != balance[0] and ordersucceed[1] == 0.0:
                                print("    매도 체결    ", end=' ')
                                self.thread_parent.lineEdit_status.setText("매도 체결")
                                if self.thread_parent.limit_buy_price != 0:
                                    self.thread_parent.profits *= \
                                        self.thread_parent.limit_sell_price / self.thread_parent.limit_buy_price - self.thread_parent.fee
                                    print("Accumulated Profits : %.6f\n" % self.thread_parent.profits)
                                    self.thread_parent.lineEdit_status.setText(
                                        "Accumulated Profits : %.6f\n" % self.thread_parent.profits)
                                break
                            else:
                                if sell_switch in [0, -1]:
                                    sell_switch = 1
                                    # elif sell_switch == 0:
                                    #     sppswitch = 1
                                    time.sleep(random.random() * 5)
                                continue

                    #       미체결 대기 / 손절 시기 파악        #
                    else:
                        try:
                            if pybithumb.get_current_price(
                                    self.thread_parent.Coin) < self.thread_parent.limit_exit_price and exit_count == 0:
                                self.thread_parent.limit_sell_price = self.thread_parent.limit_exit_price
                                sell_switch = 1
                                exit_count = 1
                                exit_start = time.time()

                            else:
                                time.sleep(1 / 80)

                        except Exception as e:
                            pass

                        #       손절 매도 미체결 시      #
                        #       손절 매도 등록하고 지정한 초가 지나면,   55초 이후로 시장가 매도 #
                        #       시장가 미체결시 계속해서 매도 등록한다.      #
                        try:
                            if time.time() - exit_start >= 30:
                                sell_switch = 1
                                exit_count = -1

                        except Exception as e:
                            pass

                except Exception as e:  # 지정 매도 대기 중 에러나서 이탈 매도가로 팔아치우는거 방지하기 위함.
                    print('취소 / 체결완료 / SellOrder = None / dict 확인중 에러 발생 :', e)
                    self.thread_parent.lineEdit_status.setText("취소 / 체결완료 / SellOrder = None / dict 확인중 에러 발생 : %s" % e)
                    continue


if __name__ == "__main__":
    #           RUN UI          #
    app = QApplication(sys.argv)
    window = OrderWindow()
    window.show()
    app.exec_()

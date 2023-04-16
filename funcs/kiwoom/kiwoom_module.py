from funcs.kiwoom.pykiwoom import Kiwoom
from funcs.kiwoom.pykiwoom import ReturnCode
import pandas as pd
import math
import time
import pickle
import os


class KiwoomModule(Kiwoom):
    def __init__(self, secret_key):

        super().__init__()

        self.comm_connect()

        if self.get_connect_state():
            self.sys_log.warning("connected.")
        else:
            self.sys_log.warning("disconnected.")

        self.screen_number = "2000"  # 추후 module_phase 에 따라 다변화할 것.
        self.account_number = self.get_login_info("ACCNO").split(";")[0]
        self.password = secret_key  # 사용하게되면 역전파하기 고민되는 부분.

        self.data_opt10004 = None  # 주식호가요청
        self.data_opt10027 = None  # 전일대비등락상위요청
        self.data_opt10075 = None  # 미체결요청
        self.data_opt10080 = None  # 분봉차트요청
        self.realtime_price = None  # 체결정보요청
        # self.realtime_data = {}  # declared for receive_real_data

        # self.OnReceiveRealData.connect(self.receive_real_data)

    def concat_candlestick(self, cnts, api_term=0.3, **kwargs):

        """
        현재, 모든 거래소로부터 얻는 ohlcv dataframe 의 interval 기준은 T (1m) 으로 설정함.
            1. T 를 기준으로 to_htf() 를 사용해 custom htf data 를 이용하도록함.

        종목코드 = 전문 조회할 종목코드
        틱범위 = 1:1분, 3:3분, 5:5분, 10:10분, 15:15분, 30:30분, 45:45분, 60:60분
        수정주가구분 = 0 or 1, 수신데이터 1:유상증자, 2:무상증자, 4:배당락, 8:액면분할, 16:액면병합, 32:기업합병, 64:감자, 256:권리락
        """

        for key, value in kwargs.items():
            self.set_input_value(key, value)
        self.comm_rq_data("주식분봉차트조회요청", "opt10080", 0, self.screen_number)

        ohlcv = self.data_opt10080

        # concat
        for _ in range(cnts - 1):
            time.sleep(api_term)
            for key, value in kwargs.items():
                self.set_input_value(key, value)
            self.comm_rq_data("주식분봉차트조회요청", "opt10080", 2, self.screen_number)
            ohlcv = pd.concat([ohlcv, self.data_opt10080])

        # 1. col_name 변경
        ohlcv.columns = ['index', 'open', 'high', 'low', 'close', 'volume']

        # 2. index 수정 및 data type float 사용 : ta-lib 연산을 위해서. (주문시에는 calc_with_hoga_unit 적용하기 때문에 무관함.)
        ohlcv = ohlcv.set_index('index').astype(float).abs()

        # 3. 시계열 순서로 데이터 정렬 및 index datetime type 으로 변환.
        if ohlcv.index.dtype != str:
            timeindex = pd.to_datetime(ohlcv.index.astype(str))
            ohlcv.index = timeindex
            ohlcv = ohlcv.sort_index()

        return ohlcv[~ohlcv.index.duplicated(keep='last')]  # 중복 제거.

    def get_high_fluc_than_day_before(self, **kwargs):

        """
        시장구분 = 000:전체, 001:코스피, 101:코스닥
        정렬구분 = 1:상승률, 2:상승폭, 3:하락률, 4:하락폭, 5:보합
        거래량조건 = 0000:전체조회, 0010:만주이상, 0050:5만주이상, 0100:10만주이상, 0150:15만주이상, 0200:20만주이상, 0300:30만주이상, 0500:50만주이상, 1000:백만주이상
        종목조건 = 0:전체조회, 1:관리종목제외, 4:우선주+관리주제외, 3:우선주제외, 5:증100제외, 6:증100만보기, 7:증40만보기, 8:증30만보기, 9:증20만보기, 11:정리매매종목제외, 12:증50만 보기, 13:증60만 보기, 14:ETF제외, 15:스팩제외, 16:ETF+ETN제외
        신용조건 = 0:전체조회, 1:신용융자A군, 2:신용융자B군, 3:신용융자C군, 4:신용융자D군, 5:신용한도초과제외, 8:신용대주, 9:신용융자전체
        상하한포함 = 0:불 포함, 1:포함
        가격조건 = 0:전체조회, 1:1천원미만, 2:1천원~2천원, 3:2천원~5천원, 4:5천원~1만원, 5:1만원이상, 8:1천원이상, 10:1만원미만
        거래대금조건 = 0:전체조회, 3:3천만원이상, 5:5천만원이상, 10:1억원이상, 30:3억원이상, 50:5억원이상, 100:10억원이상, 300:30억원이상, 500:50억원이상, 1000:100억원이상, 3000:300억원이상, 5000:500억원이상
        """

        for key, value in kwargs.items():
            self.set_input_value(key, value)
        self.comm_rq_data("전일대비등락률상위요청", "opt10027", 0, self.screen_number)

        return self.data_opt10027

    def get_available_balance(self, **kwargs):

        """
        지금은 여러개의 결과값을 출력하지만, 추후에 단일 데이터 출력할 것으로 예상함.

        계좌번호 = 전문 조회할 보유계좌번호
        비밀번호 = 사용안함(공백)
        비밀번호입력매체구분 = 00
        조회구분 = 1:합산, 2:개별
        """

        for key, value in kwargs.items():
            self.set_input_value(key, value)
        self.comm_rq_data("예수금상세현황요청", "opw00001", 0, self.screen_number)

        # int_cols = ["예수금", "주문가능금액", "100%종목주문가능금액", "d+1추정예수금", "d+2추정예수금"]
        # for col in int_cols:
        #     self.data_opw00001[col] = self.data_opw00001[col].astype(int)
        #
        # for key, value in kwargs.items():
        #     self.set_input_value(key, value)
        #     if value == "비밀번호":  # 비밀번호까지만 동일하게 입력
        #         break
        # self.comm_rq_data("계좌평가잔고내역요청", "opw00018", 0, self.screen_number)
        # while self.inquiry == '2':
        #     for key, value in kwargs.items():
        #         self.set_input_value(key, value)
        #         if value == "비밀번호":  # 비밀번호까지만 동일하게 입력
        #             break
        #     self.comm_rq_data("계좌평가잔고내역요청", "opw00018", 0, self.screen_number)
        #
        # int_cols = ["총매입금액", "총평가금액", "총평가손익금액", "총수익률(%)", "추정예탁자산"]
        # for col in int_cols:
        #     self.data_opw00018["account_evaluation"][col] = self.data_opw00018["account_evaluation"][col].astype(int)
        #     if col == "총수익률(%)":
        #         self.data_opw00018["account_evaluation"][col] /= 100
        #
        # # stocks data 의 경우 종목명을 index 로 변경함 => tr_data 에 key 값으로 접근하기 편해짐
        # self.data_opw00018["stocks"] = self.data_opw00018["stocks"].set_index("종목명")
        # int_cols = ["보유수량", "매입가", "현재가", "평가손익", "수익률(%)"]
        # for col in int_cols:
        #     self.data_opw00018["stocks"][col] = self.data_opw00018["stocks"][col].astype(int)
        #     if col == "수익률(%)":
        #         self.data_opw00018["stocks"][col] /= 100

        return self.data_opw00001  # , self.data_opw00018

    def get_order_info(self, **kwargs):

        """
        by 미체결요청
            계좌번호 = 전문 조회할 보유계좌번호
            전체종목구분 = 0:전체, 1:종목
            매매구분 = 0:전체, 1:매도, 2:매수
            종목코드 = 전문 조회할 종목코드 (공백허용, 공백입력시 전체종목구분 "0" 입력하여 전체 종목 대상으로 조회)
            체결구분 = 0:전체, 2:체결, 1:미체결
        """

        for key, value in kwargs.items():
            self.set_input_value(key, value)
        self.comm_rq_data("미체결요청", "opt10075", 0, self.screen_number)

        self.data_opt10075 = self.data_opt10075.set_index("주문번호")
        if len(self.data_opt10075) > 1:  # data validation
            int_cols = ["주문수량", "미체결수량", "체결가", "체결량", "현재가"]
            for col in int_cols:
                self.data_opt10075[col] = self.data_opt10075[col].astype(int).abs()

        return self.data_opt10075

    # deprecated by get_order_info
    def get_market_price(self, **kwargs):

        for key, value in kwargs.items():
            self.set_input_value(key, value)
        self.comm_rq_data("체결정보요청", "opt10003", 0, self.screen_number)

        return self.realtime_price

    def get_precision_by_price(self, **kwargs):

        for key, value in kwargs.items():
            self.set_input_value(key, value)
        self.comm_rq_data("주식호가요청", "opt10004", 0, self.screen_number)

        # self.data_opt10004 = self.data_opt10004.astype(int)
        # return abs(self.data_opt10004["매수최우선호가"] - self.data_opt10004["매수2차선호가"])[0]
        return self.data_opt10004

    def calc_with_hoga_unit(self, price):  # Stock 에서는 price_precision 이 무효함.

        """
        1. 체결을 고려해 올림을 기본으로 함 (+ 1 을 한 이유.)
            a. 예로, 22580 -> 22600 이 매수가로 적당할 건데, 22500 은 매수되지 않을 가능성 있음.
                i. 일단은, tp / ep / out 모두 + 1 hoga_unit 상태.
        2. integer
        """

        hoga_unit = self.get_hoga_unit(price)

        return int(((price // hoga_unit) + 1) * hoga_unit)

    def tr_data_to_dataframe(self, tr_code, request_name, rows, outputs):

        data_list = []
        for row in range(rows):
            row_data = []
            for output in outputs:
                data = self.comm_get_data(tr_code, "", request_name, row, output)
                row_data.append(data)
            data_list.append(row_data)

        return pd.DataFrame(data=data_list, columns=outputs)

    def event_connect(self, return_code):

        try:
            if return_code == ReturnCode.OP_ERR_NONE:
                # 계좌비밀번호 설정창
                # self.KOA_Functions("ShowAccountWindow", "")

                if self.get_login_info("GetServerGubun", True):
                    self.log.warning("실서버 연결 성공")
                else:
                    self.log.warning("모의투자서버 연결 성공")
            else:
                # a. 통신 종료에 대해서, 프로그램 종료 처리함
                #       i. 프로그램 재시작 환경이 준비되면, 장중 시간에 대해서는 재시작할 것.
                #       ii. 프로그램 종료시 data 저장.
                #               1. bank_module 의 instance 가 kiwoom_module 로 역전파되서 self.log_dir_name 호출이 가능해짐.
                self.save_data()
                quit()          # it works.
                # sys.exit(1)   # it doesn't work.

        except Exception as error:
            self.log.error('eventConnect {}'.format(error))
        finally:
            # commConnect() 메서드에 의해 생성된 루프를 종료시킨다.
            # 로그인 후, 통신이 끊길 경우를 대비해서 예외처리함.
            try:
                self.login_loop.exit()
            except AttributeError:
                pass

    def on_receive_tr_data(self, screen_no, request_name, tr_code, record_name, inquiry, unused0, unused1, unused2,
                           unused3):
        """
        TR 수신 이벤트
        조회요청 응답을 받거나 조회데이터를 수신했을 때 호출됩니다.
        request_name tr_code comm_rq_data()메소드의 매개변수와 매핑되는 값 입니다.
        조회데이터는 이 이벤트 메서드 내부에서 comm_get_data() 메서드를 이용해서 얻을 수 있습니다.
        :param screen_no: string - 화면번호(4자리)
        :param request_name: string - TR 요청명(comm_rq_data() 메소드 호출시 사용된 requestName)
        :param tr_code: string
        :param record_name: string
        :param inquiry: string - 조회('0': 남은 데이터 없음, '2': 남은 데이터 있음)
        """

        self.sys_log.warning("on_receive_tr_data 실행: screen_no: %s, request_name: %s, tr_code: %s, record_name: %s, inquiry: %s" % (
            screen_no, request_name, tr_code, record_name, inquiry))

        # 주문번호와 주문루프
        self.order_no = self.comm_get_data(tr_code, "", request_name, 0, "주문번호")
        self.sys_log.warning("self.order_no : {}".format(self.order_no))

        try:
            self.order_loop.exit()
        except AttributeError:
            pass

        self.inquiry = inquiry

        rows = self.get_repeat_cnt(tr_code, request_name)
        if rows == 0:
            rows = 1

        if request_name == "주식분봉차트조회요청":
            outputs = ["체결시간", "시가", "고가", "저가", "현재가", "거래량"]
            self.data_opt10080 = self.tr_data_to_dataframe(tr_code, request_name, rows, outputs)

        if request_name == "전일대비등락률상위요청":
            outputs = ['종목분류', '종목코드', '종목명', '현재가', '전일대비기호', '전일대비', '등락률']
            self.data_opt10027 = self.tr_data_to_dataframe(tr_code, request_name, rows, outputs)

        if request_name == "예수금상세현황요청":
            # outputs = ["예수금", "주문가능금액", "100%종목주문가능금액", "d+1추정예수금", "d+2추정예수금"]
            # self.data_opw00001 = self.tr_data_to_dataframe(tr_code, request_name, rows, outputs)

            self.data_opw00001 = int(self.comm_get_data(tr_code, "", request_name, 0, "d+2추정예수금"))

        if request_name == '계좌평가잔고내역요청':
            # # init data_opw00018
            # self.data_opw00018 = {'account_evaluation': [], 'stocks': []}

            # 계좌 평가 정보
            #     1. 한 request phase 에서 여러 요청을 수렴하는 경우 rows 사용 주의.
            outputs = ["총매입금액", "총평가금액", "총평가손익금액", "총수익률(%)", "추정예탁자산"]
            self.data_opw00018['account_evaluation'] = self.tr_data_to_dataframe(tr_code, request_name, 1, outputs)

            # 보유 종목 정보
            outputs = ["종목번호", "종목명", "보유수량", "매입가", "현재가", "평가손익", "수익률(%)"]
            self.data_opw00018['stocks'] = self.tr_data_to_dataframe(tr_code, request_name, rows, outputs)

        if request_name == "체결정보요청":
            # 특정 주문 번호가 정의되지 않은 상태에서, 종목에 대한 현재가 조회
            self.realtime_price = self.comm_get_data(tr_code, "", request_name, 0, "현재가")

        if request_name == "주식호가요청":
            # outputs = ["매수최우선호가", "매수2차선호가", "매수최우선잔량"]
            # self.data_opt10004 = self.tr_data_to_dataframe(tr_code, request_name, rows, outputs)
            price_1st = self.comm_get_data(tr_code, "", request_name, 0, "매수최우선호가")
            price_2nd = self.comm_get_data(tr_code, "", request_name, 0, "매수2차선호가")
            self.data_opt10004 = abs(int(price_1st) - int(price_2nd))

        if request_name == "미체결요청":
            # 1. 미체결요청 tr_code 로 체결에 관한 데이터도 충분히 조회가 가능함
            #     a. 특정 주문번호에 대한 데이터 조회시, 종목 조회 후 주문번호 키매칭.
            # 2. 현재가는 hl_out 을 위해서 사용할 것
            outputs = ["종목코드", "종목명", "주문번호", "주문구분", "주문상태", "주문수량", "미체결수량", "체결가", "체결량", "현재가"]
            self.data_opt10075 = self.tr_data_to_dataframe(tr_code, request_name, rows, outputs)

        try:
            self.request_loop.exit()
        except AttributeError:
            pass

    @staticmethod
    def get_hoga_unit(price):

        if price < 2000:
            return 1
        elif price < 5000:
            return 5
        elif price < 20000:
            return 10
        elif price < 50000:
            return 50
        elif price < 200000:
            return 100
        elif price < 500000:
            return 500
        else:
            return 1000

    @staticmethod
    def calc_with_precision(data, data_precision, def_type='floor'):

        """
        1. quantity 는 Stock 특성상 '0' precision 사용함.
        """

        if not pd.isna(data):
            if data_precision > 0:
                if def_type == 'floor':
                    data = math.floor(data * (10 ** data_precision)) / (10 ** data_precision)
                elif def_type == 'round':
                    data = float(round(data, data_precision))
                else:
                    data = math.ceil(data * (10 ** data_precision)) / (10 ** data_precision)
            else:
                data = int(data)

        return data

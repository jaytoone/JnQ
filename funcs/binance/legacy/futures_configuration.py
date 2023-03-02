from binance_f import RequestClient
from binance_f import SubscriptionClient
from binance_f.constant.test import *
from binance_f.model import *
from binance_f.base.printobject import *
from binance_f.exception.binanceapiexception import BinanceApiException
import pprint
import json
import logging


request_client = RequestClient(api_key=g_api_key, secret_key=g_secret_key)

logger = logging.getLogger("binance-client")
logger.setLevel(level=logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# sub_client = SubscriptionClient(api_key=g_api_key, secret_key=g_secret_key)
sub_client = SubscriptionClient(api_key=g_api_key, secret_key=g_secret_key,
                                receive_limit_ms=1000*3600)


class LabelType:
    OPEN = 'Open'
    CLOSE = 'Close'
    PROFIT = 'Profit'
    TRAIN = 'Train'
    TEST = 'Test'


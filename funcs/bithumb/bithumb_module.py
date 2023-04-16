import requests

url = "https://api.bithumb.com/public/candlestick/BTC_KRW/1m"

headers = {"accept": "application/json"}

response = requests.get(url, headers=headers)

print(response.text)
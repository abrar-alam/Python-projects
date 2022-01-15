import requests
import datetime
import pandas as pd
import os
from twilio.rest import Client

STOCK = "TSLA"
COMPANY_NAME = "Tesla Inc"

STOCK_ENDPOINT = "https://www.alphavantage.co/query"
STOCK_API_KEY = "xxxxxxxx"
NEWS_ENDPOINT = "https://newsapi.org/v2/everything"
NEWS_API_KEY = "xxxxxxxxxxx"
TWILLIO_ACCOUNT_SID = 'AC4a4ea53e5fb77af2da0f737d5f010d5b'
TWILLIO_AUTH_TOKEN = 'xxxxxxxxxx'
stock_api_parameters = {
    'function': 'TIME_SERIES_DAILY',
    'symbol': STOCK,
    'outputsize': 'compact',
    'apikey': STOCK_API_KEY
}
news_api_parameters = {
    'q': STOCK,
    'language': 'en',
    'apiKey': NEWS_API_KEY
}
today = datetime.datetime.today()
stock_api_response = requests.get(url=STOCK_ENDPOINT, params=stock_api_parameters)
stock_api_response.raise_for_status()
daily_stock_data = stock_api_response.json()["Time Series (Daily)"]
# print(daily_stock_data)


def get_last_two_days(d):
    dt = None
    dt2 = None
    for offset in range(1, 8):
        dt = (today - pd.tseries.offsets.BusinessDay(offset))
        if dt.date().__str__() in d:
            break
    for offset in range(1, 8):
        dt2 = (dt - pd.tseries.offsets.BusinessDay(offset))
        if dt2.date().__str__() in d:
            break
    return (dt.date().__str__(), dt2.date().__str__())


# print(get_last_two_days(daily_stock_data))


def get_percent_diff(tup, dict_data):
    # tup[0] is the most recent day
    diff = abs(float(dict_data[tup[0]]['4. close']) - float(dict_data[tup[1]]['4. close']))
    # print(f"Diff: {diff}")
    max_val = max(float(dict_data[tup[0]]['4. close']), float(dict_data[tup[1]]['4. close']))
    # print(f"Max value: {max_val}")
    percent_diff = (diff / max_val) * 100
    return percent_diff



percent_diff = get_percent_diff(get_last_two_days(daily_stock_data), daily_stock_data)
if percent_diff > 0.00002:
    # print("Get News!")
    news_api_response = requests.get(url=NEWS_ENDPOINT, params=news_api_parameters)
    news_api_response.raise_for_status()
    news_data = news_api_response.json()['articles'][:3]
    print(news_data)
    client = Client(TWILLIO_ACCOUNT_SID, TWILLIO_AUTH_TOKEN)
    for news in news_data:
        body = f"{STOCK}: 🔺{percent_diff}%\n\
        Headline: {news['title']}\n\
        Brief: {news['description']}\n\
        source: {news['url']}"
        message = client.messages \
            .create(body=body, from_='+15203467898', to='+16475258355')
        print(message.status)
else:
    print("No need to get news")






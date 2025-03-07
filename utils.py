import json
import requests
import sqlite3

from datetime import date
import yfinance as yf
import bs4 as bs
import pandas as pd
import matplotlib.pyplot as plt

from constants import ALL_MEASURES


conn = sqlite3.connect("stock_screener.db")
cur = conn.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS equity_info(ticker, info, lastmod)")
conn.commit()

def get_ticker_info(ticker):
    ticker = ticker.upper()
    res = cur.execute('SELECT * FROM equity_info WHERE ticker =?', (ticker,))
    cached = res.fetchone()
    if cached and cached[2] == date.today().isoformat():
        return json.loads(cached[1])
    stock = yf.Ticker(ticker)
    if not stock or not hasattr(stock, 'info'):
        return None
    info = stock.info
    cur.execute('INSERT OR REPLACE INTO equity_info VALUES(?, ?, ?)',
        (ticker, json.dumps(info), date.today().isoformat()))
    conn.commit()
    return info
    
def spy_components():
    result = []
    resp = requests.get('https://stockanalysis.com/list/sp-500-stocks/')
    soup = bs.BeautifulSoup(resp.text, 'html')
    table = soup.find('table', {'id': 'main-table'})
    for row in table.find_all('tr')[1:]:
        fields = row.find_all('td')
        if not fields:
            continue
        ticker = row.find_all('td')[1].text
        result.append(ticker)
    return result

def toDF(stocks):
    infos = [get_ticker_info(stock) for stock in stocks]
    infos = [x for x in infos if x is not None]
    row_keys = ['symbol', 'industryKey','sectorKey']
    df = pd.DataFrame(infos, columns=row_keys + ALL_MEASURES)
    df.set_index('symbol', inplace=True)
    return df
     
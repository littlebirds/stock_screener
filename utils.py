import json
import requests
import sqlite3

from datetime import date
import yfinance as yf
import bs4 as bs
import pandas as pd
import matplotlib.pyplot as plt

conn = sqlite3.connect("stock_screener.db")
cur = conn.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS fundamental(ticker, info, lastmod)")
conn.commit()

def update_spy_components_info():
    resp = requests.get('https://stockanalysis.com/list/sp-500-stocks/')
    soup = bs.BeautifulSoup(resp.text, 'html')
    table = soup.find('table', {'id': 'main-table'})
    tickers = []

    for row in table.find_all('tr')[1:]:
        fields = row.find_all('td')
        if not fields:
            continue
        ticker = row.find_all('td')[1].text
        res = cur.execute('SELECT * FROM fundamental WHERE ticker =?', (ticker,))
        cached = res.fetchone()
        if cached is None or cached[2] != date.today().isoformat():
            info = yf.Ticker(ticker).info
            cur.execute('INSERT OR REPLACE INTO fundamental VALUES(?, ?, ?)',
                        (ticker, json.dumps(info), date.today().isoformat()))
            conn.commit()
        tickers.append(ticker)    

    return tickers


def init_db():

    return cur




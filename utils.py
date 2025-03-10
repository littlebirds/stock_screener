import bs4
import json
import requests
import logging
import time

import yfinance as yf
import pandas as pd
from datetime import date
from typing import List, Optional, Dict, Any
from constants import FA_INDICATORS, FA_INDICATORS_ALIAS
from db import session, FundmentalInfo


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_basic_info(ticker: str) -> Optional[Dict[str, Any]]:
    ticker = ticker.upper()
    existing = session.get(FundmentalInfo, ticker)
    if existing and existing.last_updated == date.today():
        return existing.info
    
    logger.debug("Company information for '%s' is outdated. Fetching from Yahoo Finance...", ticker)
    try:
        stock = yf.Ticker(ticker)
    except Exception as e:
        time.sleep(1)
    if not stock or not hasattr(stock, 'info'):
        logger.warning("%s does not have company infroamtion in yahoo finance", ticker)
        return None
    if existing:
        existing.info = stock.info
        existing.last_updated = date.today()
    else:
        newrec = FundmentalInfo(ticker=ticker, info=stock.info, last_updated=date.today())
        session.add(newrec)
    session.commit()
    return stock.info

def spy_components() -> List[str]:
    result = []
    resp = requests.get('https://stockanalysis.com/list/sp-500-stocks/')
    soup = bs4.BeautifulSoup(resp.text, 'html.parser')
    table = soup.find('table', {'id': 'main-table'})
    for row in table.find_all('tr')[1:]:
        fields = row.find_all('td')
        if not fields:
            continue
        ticker = row.find_all('td')[1].text.upper()
        if ticker == 'BRK.B':
            continue
        result.append(ticker.upper())
    return result

def toDF(stocks: List[str]) -> pd.DataFrame:
    infos = [get_basic_info(stock) for stock in stocks]
    infos = [x for x in infos if x is not None]
    row_keys = ['symbol', 'industryKey', 'sectorKey'] + FA_INDICATORS
    df = pd.DataFrame(infos, columns=row_keys )
    df.set_index('symbol', inplace=True)
    df.rename(columns=FA_INDICATORS_ALIAS, inplace=True)
    return df

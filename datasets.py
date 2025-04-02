import math
import json
import logging
import os
import torch
import numpy as np
import pandas as pd
import yfinance as yf

from utils import DATA_DIR, spy_components
from torch.utils.data import Dataset
from bisect import bisect
 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def feature_augment(df, forward_roc_periods=[5, 15]):
  # remove where volume is 0
  prev_close = df.Close.shift(1)
  prev_vol = df.Volume.shift(1)
  ret = pd.DataFrame()
  for i in forward_roc_periods:
    ret['leadingROC' + str(i)] = df.Close / df.Close.shift(-i)

  ret['high'] = df.High / prev_close
  ret['low'] = df.Low / prev_close
  ret['open'] = df.Open / prev_close
  ret['close'] = df.Close / prev_close
  ret['vol'] = df.Volume / prev_vol
  # cross features
  ret['vol_x_high'] = ret['vol'] * ret['high']
  ret['vol_x_low'] =  ret['vol'] * ret['low']
  ret['vol_x_open'] =  ret['vol'] * ret['open']
  ret['vol_x_close'] =  ret['vol'] * ret['close']
  ret.dropna(inplace=True)
  ret = ret.map(math.log)
  # Feature names: hloc, v, v crossed h,l.o,c
  return ret.columns, torch.tensor(ret.values, dtype=torch.float32, names=(None, 'features'))
  

class SpyDailyDataset(Dataset):

  def __init__(self, download=False, roc_periods=[5, 15], transform=None):
    file_path = os.path.join(DATA_DIR, 'sp500_daily_prices.pkl')
    self.spy_tickers = spy_components()    
    self.transform = transform

    if download or not os.path.exists(file_path):
      daily_df = yf.download(self.spy_tickers).swaplevel(axis=1).astype('float32')
      # filter data before 1990
      daily_df = daily_df[daily_df.index.year >= 1990]
      # filter no volume or low price (ie. junk stocks)
      daily_df = daily_df[~(daily_df < 1).any(axis=1)]
      daily_df.to_pickle(file_path)
    else:
      daily_df = pd.read_pickle(file_path)

    # split by ticker name and normalize
    self.tensors_by_ticker = [feature_augment(daily_df[ticker], roc_periods)[1] for ticker in self.spy_tickers]
    self.feature_names = list(feature_augment(daily_df[self.spy_tickers[0]])[0])

    cat_df = torch.cat(self.tensors_by_ticker, dim=0)
    std, mean = torch.std_mean(cat_df, dim=0)
    self.std = std
    self.mean = mean
    # normalize inputs, but skip columns to predict
    i = 0
    n_features = len(self.feature_names)
    for name in self.feature_names:
      if name.startswith('leadingROC'):
        mean[i] = 0.0
        std[i] = 1.0
        n_features -= 1
      i += 1
    self.n_features = n_features
    self.tensors_by_ticker = [(t - mean) / std for t in self.tensors_by_ticker]    
    self.set_lookback_periods(45)
    #
    logger.info(f"SPY dataset loaded with {self.total_samples} samples")


  def set_lookback_periods(self, n: int):
    self.n_lookbehind = n
    num_samples = np.array([t.shape[0] - n + 1 for t in self.tensors_by_ticker])
    self.break_points = np.cumsum(num_samples)
    self.total_samples = sum(num_samples)

  def __len__(self):
    return self.total_samples

  def __getitem__(self, idx):
    if idx >= self.total_samples:
      raise RuntimeError(f"index {i} exceeds total number of samples({self.total_samples})")
    df_idx = bisect(self.break_points, idx)
    offset = idx - self.break_points[df_idx]
    logger.info(f"Sample from {self.spy_tickers[df_idx]} historical prices ==>")
    df = self.tensors_by_ticker[df_idx]
    sample = df[offset: offset + self.n_lookbehind]
    sample = torch.transpose(sample, 0, 1).rename(None)
    if self.transform:
      sample = self.transform(sample)
    return sample
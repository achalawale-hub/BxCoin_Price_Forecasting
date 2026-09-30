import yfinance as yf 
import pandas as pd
def fetch_btc_data(ticker: str = "BTC-USD", start_date: str = "2018-01-01", end_date: str = None) -> pd.DataFrame: 
  """Fetches historical Bitcoin price data from Yahoo Finance.""" 
  df = yf.download(ticker, start=start_date, end=end_date) 
  if isinstance(df.columns, pd.MultiIndex): 
    df.columns = df.columns.get_level_values(0)
  df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna() 
  df.index = pd.to_datetime(df.index) 
  return df

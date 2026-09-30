import yfinance as yf 
import pandas as pd
def fetch\_btc\_data(ticker: str = "BTC-USD", start\_date: str = "2018-01-01", end\_date: str = None) -&gt; pd.DataFrame: 
  """Fetches historical Bitcoin price
  data from Yahoo Finance.""" 
  df = yf.download(ticker, start=start\_date, end=end\_date) 
  if isinstance(df.columns, pd.MultiIndex): 
    df.columns = df.columns.get\_level\_values(0)
  df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna() 
  df.index = pd.to\_datetime(df.index) 
  return df

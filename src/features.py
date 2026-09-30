import numpy as np
import pandas as pd 

def create_features(df: pd.DataFrame, target=_horizon: int = 1) -> pd.DataFrame:
  data = df.copy() 
  data['log\_price'] = np.log(data['Close']) 
  data['log\_return'] = data['log\_price'].diff()
  data['target'] = data['log\_return'].shift(-target\_horizon)


  for window in [5, 10, 15, 30]:
   df[f'ret_mean_{window}'] = df['log_return'].rolling(window).mean() 
  df[f'ret_std_{window}'] = df['log_return'].rolling(window).std() 
  df[f'price_sma_{window}'] = df['Close'].rolling(window).mean() / df['Close']  


 for lag in range(1, 8):
   df[f'lag_return_{lag}'] = df['log_return'].shift(lag) 
    df[f'lag_volume_{lag}'] = np.log1p(df['Volume']).diff().shift(lag) 
  delta = data['Close'].diff() 
 gain = (delta.where(delta > 0, 0)).rolling(14).mean() 
loss = (-delta.where(delta < 0, 0)).rolling(14).mean() 
 data['RSI'] = 100 - (100 / (1 + rs)) 
 return data.dropna()

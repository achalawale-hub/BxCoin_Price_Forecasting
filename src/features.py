import numpy as np
import pandas as pd 

def create\_features(df: pd.DataFrame, target\_horizon: int = 1) -&gt; pd.DataFrame:
  data = df.copy() 
  data['log\_price'] = np.log(data['Close']) 
  data['log\_return'] = data['log\_price'].diff()
  data['target'] = data['log\_return'].shift(-target\_horizon)


  for window in [5, 10, 15, 30]:
    data[f'ret\_mean\_{window}'] = data['log\_return'].rolling(window).mean() 
    data[f'ret\_std\_{window}'] = data['log\_return'].rolling(window).std() 
    data[f'price\_sma\_{window}'] = data['Close'].rolling(window).mean() / data['Close'] 


 for lag in range(1, 8):
   data[f'lag\_return\_{lag}'] = data['log\_return'].shift(lag) 
   data[f'lag\_volume\_{lag}'] = np.log1p(data['Volume']).diff().shift(lag) 
 delta = data['Close'].diff() 
 gain = (delta.where(delta &gt; 0, 0)).rolling(14).mean() 
 loss = (-delta.where(delta &lt; 0, 0)).rolling(14).mean() 
 rs = gain / (loss + 1e-8) 
 data['RSI'] = 100 - (100 / (1 + rs)) 
 return data.dropna()

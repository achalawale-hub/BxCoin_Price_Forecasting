import matplotlib.pyplot as plt
from src.data\_loader import fetch\_btc\_data
from src.features import create\_features
from src.backtest import walk\_forward\_backtest

def main():
  print("🚀 Fetching Bitcoin Historical Data...") 
  raw\_data = fetch\_btc\_data(start\_date="2019-01-01")
  
  print("🛠️ Engineering Time-Series Features...")
  featured\_data = create\_features(raw\_data)
  
  print("📈 Running Walk-Forward Backtest with XGBoost Ensemble...")
  results, metrics = walk\_forward\_backtest(featured\_data, initial\_train\_size=1000, step\_size=30)
  print("\n================ BACKTEST RESULTS ================") 
  for k, v in metrics.items(): 
    print(f"{k}: {v}") 
  print("==================================================\n") 
  
  plt.figure(figsize=(12, 6))
  plt.plot(results.index, results['Actual\_Price'], label='Actual BTC Price', color='blue', alpha=0.7)
  plt.plot(results.index, results['Predicted\_Price'], label='XGBoost Predicted Price', color='orange', linestyle='--') 
  plt.title('Bitcoin Price Forecasting - Walk Forward Backtest') 
  plt.xlabel('Date') 
  plt.ylabel('Price (USD)') 
  plt.legend() 
  plt.grid(True) 
  plt.savefig('backtest\_results.png', dpi=300, bbox\_inches='tight') 
  print("📊 Backtest plot saved as 'backtest\_results.png'") 
  
if __name__== "__main__":
  main()

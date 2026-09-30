import matplotlib.pyplot as plt
from src.data_loader import fetch_btc_data
from src.features import create_features
from src.backtest import walk_forward_backtest

def main():
  print("🚀 Fetching Bitcoin Historical Data...") 
  raw_data = fetch_btc_data(start_date="2019-01-01")
  
  print("🛠️ Engineering Time-Series Features...")
  featured\_data = create\_features(raw\_data)
  
  print("📈 Running Walk-Forward Backtest with XGBoost Ensemble...")
  results, metrics = walk_forward_backtest(featured_data, initial_train_size=1000, step_size=30)
  print("\n================ BACKTEST RESULTS ================") 
  for k, v in metrics.items(): 
    print(f"{k}: {v}") 
  print("==================================================\n") 
  
  plt.figure(figsize=(12, 6))
  plt.plot(results.index, results['Actual_Price'], label='Actual BTC Price', color='blue', alpha=0.7)
  plt.plot(results.index, results['Predicted_Price'], label='XGBoost Predicted Price', color='orange', linestyle='--') 
  plt.title('Bitcoin Price Forecasting - Walk Forward Backtest') 
  plt.xlabel('Date') 
  plt.ylabel('Price (USD)') 
  plt.legend() 
  plt.grid(True) 
  plt.savefig('backtest_results.png', dpi=300, bbox_inches='tight') 
  print("📊 Backtest plot saved as 'backtest_results.png'") 
  
if __name__== "__main__":
  main()

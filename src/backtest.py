import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def walk_forward_backtest(df: pd.DataFrame, initial_train_size: int = 1000, step_size: int = 30):
    feature_cols = [c for c in df.columns if c not in ['Open', 'High', 'Low', 'Close', 'Volume', 'log_price', 'target']]
    X = df[feature_cols]
    y = df['target']
    prices = df['Close']
    predictions = []
    actuals = []
    predicted_prices = []
    actual_prices = []
    timestamps = []
    
    n_samples = len(df)
    for start in range(initial_train_size, n_samples - step_size, step_size):
        X_train, y_train = X.iloc[:start], y.iloc[:start]
        X_test, y_test = X.iloc[start:start+step_size], y.iloc[start:start+step_size]
        current_prices = prices.iloc[start:start+step_size]
 
        model = XGBRegressor(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        pred_price = current_prices * np.exp(preds)
        
        predictions.extend(preds)
        actuals.extend(y_test.values)
        predicted_prices.extend(pred_price.values)
        actual_prices.extend(prices.iloc[start+1:start+step_size+1].values)
        timestamps.extend(df.index[start:start+step_size])
        
    results_df = pd.DataFrame({
        'Actual_Price': actual_prices[:len(predicted_prices)],
        'Predicted_Price': predicted_prices
    }, index=timestamps[:len(predicted_prices)])
    
    # Calculate Quantitative Performance Metrics
    rmse = np.sqrt(mean_squared_error(results_df['Actual_Price'], results_df['Predicted_Price']))
    mae = mean_absolute_error(results_df['Actual_Price'], results_df['Predicted_Price'])
    mape = np.mean(np.abs((results_df['Actual_Price'] - results_df['Predicted_Price']) / results_df['Actual_Price'])) * 100
    dir_acc = np.mean(np.sign(predictions[:len(actuals)]) == np.sign(actuals)) * 100
    
    metrics = {
        'RMSE': round(rmse, 2),
        'MAE': round(mae, 2),
        'MAPE (%)': round(mape, 2),
        'Directional Accuracy (%)': round(dir_acc, 2)
    }
    
    return results_df, metrics

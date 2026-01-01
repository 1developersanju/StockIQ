import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
try:
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, LSTM
    HAS_KERAS = True
except ImportError:
    HAS_KERAS = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
import plotly.graph_objects as go
import plotly.utils
import json

# --- UTILITY FUNCTIONS ---

def sanitize_json(obj):
    """
    Recursively replaces NaN and Inf with None for JSON compliance.
    """
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_json(x) for x in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif hasattr(obj, 'tolist'): # Handle Numpy/Pandas types
        return sanitize_json(obj.tolist())
    return obj

def get_dataset(comp_name, start_date, end_date, interval="1d"):
    try:
        comp = yf.Ticker(comp_name)
        df = comp.history(start=start_date, end=end_date, interval=interval)
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def buy_sell(signal):
    """
    Generates Buy/Sell signals based on MACD crossover.
    """
    buy = []
    sell = []
    flag = -1
    
    for i in range(0,len(signal)):
        if signal['MACD'][i] > signal['Signal'][i]:
            sell.append(np.nan)
            if flag != 1:
                buy.append(signal['Close'][i])
                flag = 1
            else:
                buy.append(np.nan)
        elif signal['MACD'][i] < signal['Signal'][i]:
            buy.append(np.nan)
            if flag != 0:
                sell.append(signal['Close'][i])
                flag = 0
            else:
                sell.append(np.nan)
        else:
            buy.append(np.nan)
            sell.append(np.nan)
    return (buy,sell)

def bol_band(newdf):
    """
    Generates Buy/Sell signals based on Bollinger Bands.
    """
    buy = []
    sell = []
    for i in range((len(newdf['Close']))):
        if newdf['Close'][i] > newdf['Upper'][i]:
            buy.append(np.nan)
            sell.append(newdf['Close'][i])
        elif newdf['Close'][i] < newdf['Lower'][i]:
            buy.append(newdf['Close'][i])
            sell.append(np.nan)
        else:
            buy.append(np.nan)
            sell.append(np.nan)
    return buy,sell
        
def ema_buy_sell(df):
    """
    Generates Buy/Sell signals based on Triple EMA Crossover.
    """
    buy = []
    sell = []
    flag_long = False
    flag_short = False
    
    for i in range(len(df)):
        if df['mid'][i] < df['long'][i] and df['short'][i] < df['mid'][i] and flag_long == False and flag_short == False:
            buy.append(df['Close'][i])
            sell.append(np.nan)
            flag_short = True
        elif flag_short == True and df['short'][i] > df['mid'][i]:
            sell.append(df['Close'][i])
            buy.append(np.nan)
            flag_short = False
        elif df['mid'][i] > df['long'][i] and df['short'][i] > df['mid'][i] and flag_long == False and flag_short == False:
            buy.append(df['Close'][i])
            sell.append(np.nan)
            flag_long = True
        elif flag_long == True and df['short'][i] < df['mid'][i]:
            sell.append(df['Close'][i])
            buy.append(np.nan)
            flag_long = False 
        else:
            buy.append(np.nan)
            sell.append(np.nan)
            
    return buy ,sell

def get_mape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return mape

def get_rmse(y_true, y_pred):
    rmse = np.sqrt(np.mean(np.power((y_true - y_pred),2)))
    return rmse

def get_x_y(data, N, offset):
    X, y = [], []
    for i in range(offset, len(data)):
        X.append(data[i-N:i])
        y.append(data[i])
    X = np.array(X)
    y = np.array(y)
    return X, y

# --- SERVICE FUNCTIONS ---

def get_stock_data(ticker, start, end, interval="1d"):
    df = get_dataset(ticker, start, end, interval)
    if df.empty:
        return {"error": "No data found"}
    
    # Prepare data for frontend (JSON serializable)
    # We'll return the raw data so frontend can plot it
    result = {
        "dates": df['Date'].dt.strftime('%Y-%m-%d').tolist(),
        "open": df['Open'].tolist(),
        "high": df['High'].tolist(),
        "low": df['Low'].tolist(),
        "close": df['Close'].tolist(),
        "volume": df['Volume'].tolist()
    }
    return sanitize_json(result)

def get_fundamentals(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info 
        
        # Helper to safely get value or "N/A"
        def get_val(key):
            return info.get(key, 'N/A')

        dividend_yield = get_val('dividendYield')
        if isinstance(dividend_yield, (int, float)):
             dividend_yield = f"{dividend_yield * 100:.2f}%"
        
        data = {
            "pe_ratio": get_val('trailingPE'),
            "roe": get_val('returnOnEquity'),
            "pb_ratio": get_val('priceToBook'),
            "debt_to_equity": get_val('debtToEquity'),
            "current_ratio": get_val('currentRatio'),
            "operating_cash_flow": get_val('operatingCashflow'),
            "return_on_assets": get_val('returnOnAssets'),
            "price_to_sales": get_val('priceToSales'),
            "dividend_yield": dividend_yield,
            "gross_profit_margin": get_val('grossMargins'),
            # These were hardcoded or missing in original but I'll check anyway
            "quick_ratio": 'N/A', 
            "cash_ratio": 'N/A',
            "interest_coverage": 'N/A'
        }
        return sanitize_json(data)
    except Exception as e:
        return {"error": str(e)}

def run_ml_model(ticker, start, end, model_type):
    df = get_dataset(ticker, start, end)
    if df.empty: return {"error": "No data"}
    
    future_days = 90
    
    if model_type == 'Tree Classifier':
        new_df1 = pd.DataFrame()
        new_df1['Close'] = df['Close']
        new_df1['Prediction'] = df[['Close']].shift(-future_days)
        
        X = np.array(new_df1.drop(['Prediction'], axis=1))[:-future_days]
        y = np.array(new_df1['Prediction'])[:-future_days]
        
        if len(X) == 0: return {"error": "Not enough data"}

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        x_future = new_df1.drop(['Prediction'], axis=1)[:-future_days]
        x_future = x_future.tail(future_days)
        x_future = np.array(x_future)
        
        if not HAS_SKLEARN:
            return {"error": "Machine Learning libraries (scikit-learn) are not installed on this system."}
        
        tree = DecisionTreeRegressor().fit(x_train, y_train)
        predictions = tree.predict(x_future)
        
        # Prepare for plotting
        valid = new_df1[X.shape[0]:].copy()
        valid['Predictions'] = predictions
        
        # Return data for plotting
        return sanitize_json({
            "model": "Tree Classifier",
            "train_dates": df.iloc[:len(X)]['Date'].dt.strftime('%Y-%m-%d').tolist(), # approx
            "train_close": new_df1['Close'].head(len(X)).tolist(),
            "valid_dates": df.iloc[X.shape[0]:]['Date'].dt.strftime('%Y-%m-%d').tolist(),
            "valid_close": valid['Close'].tolist(),
            "valid_predictions": valid['Predictions'].tolist(),
            "score": tree.score(x_test, y_test)
        })

    elif model_type == 'LSTM':
        # Prepare data
        test_size = 0.05
        training_size = 1 - test_size
        train_num = int(training_size * len(df))
        
        train = df[:train_num][[ 'Close']]
        test = df[train_num:][[ 'Close']]
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[['Close']])
        scaled_data_train = scaled_data[:train.shape[0]]
        X_train, y_train = get_x_y(scaled_data_train, 60, 60)
        
        if not HAS_KERAS:
            return {"error": "Deep Learning libraries (TensorFlow/Keras) are not installed on this system."}
            
        # Build model
        lstm_units = 50 
        model = Sequential()
        model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=(X_train.shape[1],1)))
        model.add(LSTM(units=lstm_units))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        
        model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=0)
        
        inputs = df['Close'][len(df) - len(test) - 60:].values.reshape(-1,1)
        inputs = scaler.transform(inputs)
        X_test = []
        for i in range(60, inputs.shape[0]):
            X_test.append(inputs[i-60:i,0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
        
        closing_price = model.predict(X_test)
        closing_price = scaler.inverse_transform(closing_price)
        test['Predictions_lstm'] = closing_price
        
        rmse = get_rmse(np.array(test['Close']), np.array(test['Predictions_lstm']))
        mape = get_mape(np.array(test['Close']), np.array(test['Predictions_lstm']))
        
        return sanitize_json({
            "model": "LSTM",
            "train_dates": df.iloc[:train_num]['Date'].dt.strftime('%Y-%m-%d').tolist(),
            "train_close": train['Close'].tolist(),
            "test_dates": df.iloc[train_num:]['Date'].dt.strftime('%Y-%m-%d').tolist(),
            "test_close": test['Close'].tolist(),
            "test_predictions": test['Predictions_lstm'].tolist(),
            "rmse": rmse,
            "mape": mape
        })

    elif model_type == 'Prophet':
        period = 60
        new_df2 = pd.DataFrame()
        new_df2['y'] = df['Close']
        new_df2['ds'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        
        train_len = len(new_df2) - period
        train = new_df2[:train_len]
        valid = new_df2[train_len:].copy()
        
        if not HAS_PROPHET:
            return {"error": "Prophet library is not installed on this system."}
            
        model = Prophet()
        model.fit(train)
        future = model.make_future_dataframe(period)
        forecast = model.predict(future)
        forecast_valid = forecast['yhat'][train_len:]
        
        rms = np.sqrt(np.mean(np.power((np.array(valid['y'])-np.array(forecast_valid)),2)))
        valid['Predictions'] = forecast_valid.values
        
        return sanitize_json({
            "model": "Prophet",
            "train_dates": train['ds'].dt.strftime('%Y-%m-%d').tolist(),
            "train_y": train['y'].tolist(),
            "valid_dates": valid['ds'].dt.strftime('%Y-%m-%d').tolist(),
            "valid_y": valid['y'].tolist(),
            "valid_predictions": valid['Predictions'].tolist(),
            "rms": rms
        })
    
    return sanitize_json({"error": "Invalid model type"})

# Note: We should wrap all returns in sanitize_json or handle inside the blocks.
# Let's handle it at the end of each return for run_ml_model and run_ta.

def run_ta(ticker, start, end, indicator):
    df = get_dataset(ticker, start, end)
    if df.empty: return {"error": "No data"}
    
    dates = df['Date'].dt.strftime('%Y-%m-%d').tolist()
    close = df['Close'].tolist()
    
    if indicator == 'EMA':
        short_ema = df.Close.ewm(span=5, adjust=False).mean()
        mid_ema = df.Close.ewm(span=21, adjust=False).mean()
        long_ema = df.Close.ewm(span=63, adjust=False).mean()
        
        new_df = pd.DataFrame({'Close': df['Close'], 'short': short_ema, 'mid': mid_ema, 'long': long_ema})
        buy, sell = ema_buy_sell(new_df)
        
        return sanitize_json({
            "indicator": "EMA",
            "dates": dates,
            "close": close,
            "short_ema": short_ema.tolist(),
            "mid_ema": mid_ema.tolist(),
            "long_ema": long_ema.tolist(),
            "buy": [x if not pd.isna(x) else None for x in buy],
            "sell": [x if not pd.isna(x) else None for x in sell]
        })
        
    elif indicator == 'RSI':
        delta = df['Close'].diff()[1:]
        up = delta.copy()
        down = delta.copy()
        up[up<0] = 0
        down[down>0] = 0
        
        period = 14
        avg_gain = up.ewm(span=period, adjust=False).mean()
        avg_loss = abs(down.ewm(span=period, adjust=False).mean())
        RS = avg_gain / avg_loss
        RSI = 100.0 - (100.0 / (1.0 + RS))
        
        return sanitize_json({
            "indicator": "RSI",
            "dates": dates[1:], # RSI is one shorter
            "close": close[1:],
            "rsi": RSI.tolist(),
            "rsi_dates": df['Date'][1:].dt.strftime('%Y-%m-%d').tolist()
        })
        
    elif indicator == 'MACD':
        short_ema = df['Close'].ewm(span=12, adjust=False).mean()
        long_ema = df['Close'].ewm(span=26, adjust=False).mean()
        MACD = short_ema - long_ema
        signal = MACD.ewm(span=9, adjust=False).mean()
        
        new_df = pd.DataFrame({'MACD': MACD, 'Signal': signal, 'Close': df['Close']})
        buy, sell = buy_sell(new_df)
        
        return sanitize_json({
            "indicator": "MACD",
            "dates": dates,
            "close": close,
            "macd": MACD.tolist(),
            "signal": signal.tolist(),
            "buy": [x if not pd.isna(x) else None for x in buy],
            "sell": [x if not pd.isna(x) else None for x in sell]
        })
        
    elif indicator == 'Bollinger Band':
        period = 20
        newdf = pd.DataFrame()
        newdf['Close'] = df['Close']
        newdf['SMA'] = newdf['Close'].rolling(window=period).mean()
        newdf['STD'] = newdf['Close'].rolling(window=period).std()
        newdf['Upper'] = newdf['SMA'] + (newdf['STD']*2)
        newdf['Lower'] = newdf['SMA'] - (newdf['STD']*2)
        
        buy, sell = bol_band(newdf)
        
        return sanitize_json({
            "indicator": "Bollinger Band",
            "dates": dates,
            "close": close,
            "sma": newdf['SMA'].tolist(),
            "upper": newdf['Upper'].tolist(),
            "lower": newdf['Lower'].tolist(),
            "buy": buy,
            "sell": sell
        })
        
    return sanitize_json({"error": "Invalid indicator"})

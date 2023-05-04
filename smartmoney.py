# Import libraries
import requests
import pandas as pd
import numpy as np
import talib as ta
import matplotlib.pyplot as plt

# Define trading parameters
symbol = "BTCUSDT" # Trading pair
interval = "1h" # Candlestick interval
lookback = 100 # Number of candles to look back

# Define smart money method indicators
rsi_period = 14 # RSI period
rsi_overbought = 70 # RSI overbought threshold
rsi_oversold = 30 # RSI oversold threshold
ema_fast = 9 # Fast EMA period
ema_slow = 21 # Slow EMA period

# Define a function to get historical data from binance API
def get_data(symbol, interval, lookback):
    # Get klines from binance API
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={lookback}"
    response = requests.get(url)
    klines = response.json()
    # Convert klines to a dataframe
    df = pd.DataFrame(klines, columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_base_volume", "taker_quote_volume", "ignore"])
    # Drop unnecessary columns
    df.drop(["close_time", "quote_volume", "trades", "taker_base_volume", "taker_quote_volume", "ignore"], axis=1, inplace=True)
    # Convert columns to numeric values
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric)
    # Convert open_time to datetime format
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    # Set open_time as the index
    df.set_index("open_time", inplace=True)
    # Return the dataframe
    return df

# Define a function to calculate smart money method indicators and signals
def get_signals(df):
    # Calculate RSI
    df["rsi"] = ta.RSI(df["close"], rsi_period)
    # Calculate fast and slow EMA
    df["ema_fast"] = ta.EMA(df["close"], ema_fast)
    df["ema_slow"] = ta.EMA(df["close"], ema_slow)
    # Calculate EMA crossover signal
    df["ema_crossover"] = np.where(df["ema_fast"] > df["ema_slow"], 1, -1)
    # Calculate EMA crossover direction
    df["ema_direction"] = df["ema_crossover"].diff()
    # Calculate smart money method buy signal
    df["buy_signal"] = np.where((df["rsi"] < rsi_oversold) & (df["ema_direction"] == 2), 1, 0)
    # Calculate smart money method sell signal
    df["sell_signal"] = np.where((df["rsi"] > rsi_overbought) & (df["ema_direction"] == -2), -1, 0)
    # Return the dataframe with signals
    return df

# Define a function to plot the data and signals
def plot_data(df):
    # Create a figure and a subplot
    fig = plt.figure(figsize=(15,10))
    ax1 = plt.subplot2grid((10,1), (0,0), rowspan=7, colspan=1)
    ax2 = plt.subplot2grid((10,1), (7,0), rowspan=3, colspan=1)
    
    # Plot the candlestick chart on ax1
    ax1.plot(df.index, df["close"], color="blue")
    
    # Plot the fast and slow EMA on ax1
    ax1.plot(df.index, df["ema_fast"], color="green")
    ax1.plot(df.index, df["ema_slow"], color="red")
    
    # Plot the buy and sell signals on ax1
    ax1.scatter(df.index[df["buy_signal"] == 1], df[df["buy_signal"] == 1]["close"], marker="^", color="green")
    ax1.scatter(df.index[df["sell_signal"] == -1], df[df["sell_signal"] == -1]["close"], marker="v", color="red")
    
    # Plot the RSI on ax2
    ax2.plot(df.index, df["rsi"], color="black")
    
    # Plot the overbought and oversold levels on ax2
    ax2.axhline(rsi_overbought, color="red")
    ax2.axhline(rsi_oversold, color="green")
    
    # Set the title and labels for the plot
    plt.suptitle(f"{symbol} {interval} Smart Money Method Signals")
    
    # Show the plot 
    plt.show()

# Get the latest data from binance API 
data = get_data(symbol, interval, lookback) 

# Get the signals from the smart money method 
data = get_signals(data) 

# Plot the data and signals 
plot_data(data)

# Print the last row of data 
last_row = data.iloc[-1] 
print(last_row)

# Print the trading advice based on the signals 
if last_row["buy_signal"] == 1:
  print(f"You should buy {symbol} at {last_row['close']} and set your stop loss and take profit levels according to your risk appetite.")
elif last_row["sell_signal"] == -1:
  print(f"You should sell {symbol} at {last_row['close']} and set your stop loss and take profit levels according to your risk appetite.")
else:
  print(f"You should wait for a clear signal before entering a trade.")
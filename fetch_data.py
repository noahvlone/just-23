import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

def create_folders():
    """Create necessary folders for data organization"""
    folders = ["data", "data/stocks", "data/crypto", "lists"]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

def read_symbols(filename):
    """Read symbols from a text file"""
    if not os.path.exists(filename):
        return []
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def fetch_data(symbol, start_date, end_date):
    """Fetch data from Yahoo Finance"""
    try:
        print(f"Fetching data for {symbol}...")
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            print(f"No data found for {symbol}")
            return None
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def save_default_lists():
    """Create default symbol list files if they don't exist"""
    default_stocks = """AAPL
GOOGL
MSFT
TSLA
NVDA
AMD
META
AMZN
NFLX
IBM
ORCL
INTC
CSCO"""

    default_crypto = """BTC-USD
ETH-USD
USDT-USD
BNB-USD
XRP-USD
SOL-USD
ADA-USD
DOGE-USD
MATIC-USD
DOT-USD"""

    if not os.path.exists("lists/stock_symbols.txt"):
        with open("lists/stock_symbols.txt", "w") as f:
            f.write(default_stocks)
    
    if not os.path.exists("lists/crypto_symbols.txt"):
        with open("lists/crypto_symbols.txt", "w") as f:
            f.write(default_crypto)

def main():
    # Create folders and default symbol lists
    create_folders()
    save_default_lists()
    
    # Set time period for maximum available data
    end_date = datetime.now()
    start_date = datetime(2000, 1, 1)  # Going back to year 2000
    
    # Read symbols from files
    stock_symbols = read_symbols("lists/stock_symbols.txt")
    crypto_symbols = read_symbols("lists/crypto_symbols.txt")
    
    # Process stocks
    for symbol in stock_symbols:
        data = fetch_data(symbol, start_date, end_date)
        if data is not None:
            print(f"Saving data for {symbol}")
            data.to_csv(f"data/stocks/{symbol}_data.csv", index=False)
    
    # Process crypto
    for symbol in crypto_symbols:
        data = fetch_data(symbol, start_date, end_date)
        if data is not None:
            print(f"Saving data for {symbol}")
            clean_symbol = symbol.replace("-USD", "")
            data.to_csv(f"data/crypto/{clean_symbol}_data.csv", index=False)

if __name__ == "__main__":
    main()
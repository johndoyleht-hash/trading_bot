import time
import logging
import numpy as np

# Set up logging
logging.basicConfig(filename='trading_algorithm.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Risk management parameters
account_balance = 25000  # Example starting balance
max_drawdown = 0.08  # 8% max drawdown
max_daily_drawdown = 0.039  # 3.9% max daily drawdown
risk_per_trade = 0.01  # 1% risk per trade

# Performance Monitoring Variables
portfolio_value = account_balance
peak_value = account_balance
total_profit = 0
total_loss = 0
win_count = 0
loss_count = 0
trade_count = 0
returns = []  # List to store the returns for calculating Sharpe Ratio

# Example mocked fetch_price_data
def fetch_price_data(pair, timeframe):
    print(f"[MOCK] Fetching price for {pair} on {timeframe}")
    # Return mock price
    return 1.12345

# Example mocked fetch_atr
def fetch_atr(pair, timeframe, period=14):
    print(f"[MOCK] Fetching ATR for {pair} on {timeframe}")
    # Return mock ATR
    return 0.0012

# Example mocked RSI (simple static value for testing)
def fetch_rsi(pair, timeframe, period=14):
    print(f"[MOCK] Fetching RSI for {pair} on {timeframe}")
    # Return static RSI value to simulate buy/sell/neutral
    # Change this value to test different signals
    return 25  # Under 30 triggers buy signal, over 70 triggers sell

# Example trading signal based on RSI
def check_for_trade(pair):
    current_price = fetch_price_data(pair, '1m')
    if current_price is None:
        print(f"No price data available for {pair}. Skipping.")
        return None

    rsi_short = fetch_rsi(pair, '1m')
    if rsi_short is None:
        print(f"No RSI data available for {pair}. Skipping.")
        return None

    if rsi_short < 30:
        return 'buy'
    elif rsi_short > 70:
        return 'sell'
    else:
        return None

# Calculate dynamic stop loss based on ATR
def calculate_stop_loss(current_price, atr, direction, multiplier=1.5):
    if direction == 'buy':
        stop_loss_price = current_price - (atr * multiplier)
    elif direction == 'sell':
        stop_loss_price = current_price + (atr * multiplier)
    else:
        return None
    return stop_loss_price

# Calculate position size based on risk management
def calculate_position_size(account_balance, stop_loss_distance, risk_percentage=risk_per_trade):
    risk_amount = account_balance * risk_percentage
    position_size = risk_amount / abs(stop_loss_distance)
    return position_size

# Place the trade (mock)
def place_trade(pair, direction, price, position_size, stop_loss_price):
    global total_profit, total_loss, win_count, loss_count, trade_count, portfolio_value, peak_value, returns
    
    print(f"[MOCK] Placing {direction} trade on {pair} at {price:.5f}, Position Size: {position_size:.2f}, Stop Loss: {stop_loss_price:.5f}")
    
    trade_count += 1
    
    # Simulate trade outcome alternates win/loss for demonstration
    trade_result = 'win' if trade_count % 2 == 0 else 'loss'
    
    trade_profit_loss = position_size * (price - stop_loss_price)
    portfolio_value += trade_profit_loss
    
    if trade_result == 'win':
        win_count += 1
        total_profit += trade_profit_loss
    else:
        loss_count += 1
        total_loss += trade_profit_loss
    
    if portfolio_value > peak_value:
        peak_value = portfolio_value
    drawdown = (peak_value - portfolio_value) / peak_value
    
    returns.append(trade_profit_loss / portfolio_value)
    
    print(f"Trade {trade_count}: {trade_result} | P/L: {trade_profit_loss:.2f} | Portfolio: {portfolio_value:.2f}")
    
    if trade_count % 5 == 0:
        calculate_metrics()

def calculate_metrics():
    if len(returns) > 0:
        mean_return = np.mean(returns)
        std_dev = np.std(returns)
        sharpe_ratio = mean_return / std_dev if std_dev != 0 else 0
    else:
        sharpe_ratio = 0

    cumulative_return = (portfolio_value - account_balance) / account_balance
    max_drawdown_local = (peak_value - portfolio_value) / peak_value
    win_loss_ratio = win_count / loss_count if loss_count > 0 else win_count

    average_win = total_profit / win_count if win_count > 0 else 0
    average_loss = total_loss / loss_count if loss_count > 0 else 0
    expectancy = ((average_win * win_count) - (average_loss * loss_count)) / trade_count if trade_count > 0 else 0
    
    print(f"Metrics after {trade_count} trades:")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Cumulative Return: {cumulative_return:.2%}")
    print(f"Max Drawdown: {max_drawdown_local:.2%}")
    print(f"Win/Loss Ratio: {win_loss_ratio:.2f}")
    print(f"Expectancy: {expectancy:.2f}")
    print(f"Total Profit: {total_profit:.2f} | Total Loss: {total_loss:.2f}")

def main(account_balance=25000):
    print("✅ main.py is running!")
    print("✅ Calling main()...")
    print("✅ Inside the main() function")

    all_fx_pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY']
    trade_limit = 10  # Limit trades for testing
    global trade_count

    while trade_count < trade_limit:
        for pair in all_fx_pairs:
            print(f"🔍 Checking signal for {pair}...")
            trade_signal = check_for_trade(pair)
            print(f"🟢 Signal for {pair}: {trade_signal}")

            if trade_signal:
                current_price = fetch_price_data(pair, '1m')
                print(f"📈 Current price for {pair}: {current_price}")

                if current_price:
                    atr = fetch_atr(pair, '1m')
                    stop_loss = calculate_stop_loss(current_price, atr, trade_signal)
                    position_size = calculate_position_size(account_balance, stop_loss - current_price)
                    place_trade(pair, trade_signal, current_price, position_size, stop_loss)

                    if trade_count >= trade_limit:
                        break

            time.sleep(1)

    print("✅ Trading loop complete.")

if __name__ == '__main__':
    main()

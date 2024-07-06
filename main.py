import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def calculate_quality(profits, losses):
    """Calculate the Quality metric as described by Mark Douglas"""
    if len(profits) + len(losses) == 0:
        return 0 # No trades
    total_trades = len(profits) + len(losses)
    win_rate = len(profits) / total_trades
    loss_rate = len(losses) / total_trades

    avg_win = np.mean(profits) if profits else 0
    avg_loss = np.mean(np.abs(losses)) if losses else 0
    risk = avg_loss if avg_loss > 0 else 1
    avg_win_r = avg_win / risk
    avg_loss_r = avg_loss / risk

    expectancy = (win_rate * avg_win_r) - (loss_rate * avg_loss_r)

    r_multiples = [profit / risk for profit in profits] + [-loss / risk for loss in losses]
    std_dev_r = np.std(r_multiples) if r_multiples else 1

    quality = expectancy / std_dev_r if std_dev_r > 0 else 0
    return quality

def calculate_metrics(returns, initial_capital, final_capital, profits, losses):
    total_return = (final_capital - initial_capital) / initial_capital
    trading_hours = len(returns)
    years = trading_hours / (252 * 6.5)
    annualized_return = (1 + total_return) ** (1 / years) - 1
    annualized_volatility = returns.std() * np.sqrt(252 * 6.5)
    risk_free_rate = 0.02
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else 0
    cumulative_returns = (1 + returns).cumprod() - 1
    max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()

    total_trades = len(profits) + len(losses)
    win_rate = len(profits) / total_trades if total_trades > 0 else 0
    avg_win_yield = np.mean(profits) / initial_capital if profits else 0
    avg_loss_yield = np.mean(np.abs(losses)) / initial_capital if losses else 0
    quality = calculate_quality(profits, losses)

    return {
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Win Rate": win_rate,
        "Avg Win Yield": avg_win_yield,
        "Avg Loss Yield": avg_loss_yield,
        "Quality": quality
    }

def calculate_commission(shares, price):
    commission = max(1.0, shares * 0.005)
    max_commission = shares * price * 0.01
    return min(commission, max_commission)

def simulate_trading(df, model, feature_indices, scaler, initial_capital=5000):
    current_capital = initial_capital
    position_size = 0.35 # AQUIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
    in_position = False
    entry_price = 0
    shares = 0
    hourly_returns = []
    equity_curve = [initial_capital]
    transactions = []
    profits = []
    losses = []
    total_commission = 0

    df_selected = df[feature_indices]

    for i in range(len(df)):
        if i < 50:
            hourly_returns.append(0)
            equity_curve.append(current_capital)
            continue

        X = df_selected.iloc[i:i+1]
        X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_indices)
        prediction = model.predict(X_scaled)[0]

        if not in_position and prediction:
            in_position = True
            entry_price = df['Close'].iloc[i]
            shares = int((current_capital * position_size) / entry_price)
            commission = calculate_commission(shares, entry_price)
            current_capital -= commission
            total_commission += commission
            transactions.append(("BUY", i, entry_price, shares, current_capital, commission))
        elif in_position and not prediction:
            exit_price = df['Close'].iloc[i]
            commission = calculate_commission(shares, exit_price)
            profit = (exit_price - entry_price) * shares - commission
            current_capital += profit
            total_commission += commission
            transactions.append(("SELL", i, exit_price, shares, current_capital, commission))
            if profit > 0:
                profits.append(profit)
            else:
                losses.append(profit)
            in_position = False
            shares = 0

        if in_position:
            current_capital += (df['Close'].iloc[i] - df['Close'].iloc[i-1]) * shares

        hourly_return = (current_capital - equity_curve[-1]) / equity_curve[-1]
        hourly_returns.append(hourly_return)
        equity_curve.append(current_capital)

    if in_position:
        exit_price = df['Close'].iloc[-1]
        commission = calculate_commission(shares, exit_price)
        profit = (exit_price - entry_price) * shares - commission
        current_capital += profit
        total_commission += commission
        transactions.append(("SELL", len(df)-1, exit_price, shares, current_capital, commission))
        if profit > 0:
            profits.append(profit)
        else:
            losses.append(profit)

    return hourly_returns, equity_curve, transactions, profits, losses, total_commission, initial_capital, current_capital

def calculate_dynamic_quality(prices, window_size=50):
    profits = []
    losses = []
    qualities = []

    for i in range(window_size, len(prices)):
        window_prices = prices[i - window_size:i]
        current_price = prices[i]

        signal = np.random.choice([0, 1])
        if signal == 1:
            entry_price = window_prices[-1]
            profit = current_price - entry_price
            if profit > 0:
                profits.append(profit)
            else:
                losses.append(profit)

        quality = calculate_quality(profits, losses)
        qualities.append(quality)

    return [0] * window_size + qualities

# Fetch and prepare data
ticker = yf.Ticker("QQQ")
df = ticker.history(interval="1h", period="2y")
df = df.between_time('09:30', '16:00')

# Only keep the columns we're interested in
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

# Add derived features
df['Returns'] = df['Close'].pct_change()
df['Range'] = df['High'] - df['Low']
df['PrevClose'] = df['Close'].shift(1)
df['PrevVolume'] = df['Volume'].shift(1)
df['HourOfDay'] = df.index.hour
df['DayOfWeek'] = df.index.dayofweek

# Calculate Quality feature
window_size = 50
df['Quality'] = calculate_dynamic_quality(df['Close'].values, window_size)
df['Quality'] = df['Quality'].shift(1)

features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'Range', 'PrevClose', 'PrevVolume', 'HourOfDay', 'DayOfWeek', 'Quality']
X = df[features]
y = df['Close'].shift(-1) > df['Close']

X = X.iloc[:-1]
y = y.iloc[1:]

X = X.dropna()
y = y.loc[X.index]

common_index = X.index.intersection(y.index)
X = X.loc[common_index]
y = y.loc[common_index]

original_features = X.columns.tolist()

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=original_features)

tscv = TimeSeriesSplit(n_splits=5)

rf_model = RandomForestClassifier(
    n_estimators=25,
    max_depth=5, # AQUIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

cv_scores = []
for train_index, test_index in tscv.split(X_scaled):
    X_train, X_test = X_scaled.iloc[train_index], X_scaled.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    cv_scores.append(accuracy_score(y_test, y_pred))

print(f"Random Forest with Quality - Mean CV Accuracy: {np.mean(cv_scores):.4f}")

rf_model.fit(X_scaled, y)

train_size = int(0.8 * len(X_scaled))
X_train, X_test = X_scaled.iloc[:train_size], X_scaled.iloc[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

df_train = df.iloc[:train_size]
train_returns, train_equity, train_transactions, train_profits, train_losses, train_commission, train_initial_capital, train_final_capital = simulate_trading(df_train, rf_model, X_scaled.columns, scaler)

df_test = df.iloc[train_size:]
test_returns, test_equity, test_transactions, test_profits, test_losses, test_commission, test_initial_capital, test_final_capital = simulate_trading(df_test, rf_model, X_scaled.columns, scaler)

train_metrics = calculate_metrics(pd.Series(train_returns), train_initial_capital, train_final_capital, train_profits, train_losses)
test_metrics = calculate_metrics(pd.Series(test_returns), test_initial_capital, test_final_capital, test_profits, test_losses)

print("Train Set Performance with Quality:")
print(f"Initial Capital: ${train_initial_capital:.2f}")
print(f"Final Capital: ${train_final_capital:.2f}")
for metric, value in train_metrics.items():
    if metric in ["Sharpe Ratio", "Win Rate", "Quality"]:
        print(f"{metric}: {value:.4f}")
    elif metric in ["Avg Win Yield", "Avg Loss Yield"]:
        print(f"{metric}: {value:.4%}")
    else:
        print(f"{metric}: {value:.2%}")
print(f"Total Commission (Train): ${train_commission:.2f}")

print("\nTest Set Performance with Quality:")
print(f"Initial Capital: ${test_initial_capital:.2f}")
print(f"Final Capital: ${test_final_capital:.2f}")
for metric, value in test_metrics.items():
    if metric in ["Sharpe Ratio", "Win Rate", "Quality"]:
        print(f"{metric}: {value:.4f}")
    elif metric in ["Avg Win Yield", "Avg Loss Yield"]:
        print(f"{metric}: {value:.4%}")
    else:
        print(f"{metric}: {value:.2%}")
print(f"Total Commission (Test): ${test_commission:.2f}")

plt.figure(figsize=(12, 6))
plt.plot(range(len(train_equity)), train_equity, label='Train Set with Quality')
plt.plot(range(len(train_equity), len(train_equity) + len(test_equity)), test_equity, label='Test Set with Quality')
plt.title('Equity Curve with Quality Feature')
plt.xlabel('Trading Hours')
plt.ylabel('Capital')
plt.legend()
plt.show()

# Compare with buy-and-hold strategy
buy_and_hold_returns = df['Close'].pct_change().fillna(0)
buy_and_hold_equity = [5000]
for ret in buy_and_hold_returns:
    buy_and_hold_equity.append(buy_and_hold_equity[-1] * (1 + ret))

buy_and_hold_metrics = calculate_metrics(buy_and_hold_returns, buy_and_hold_equity[0], buy_and_hold_equity[-1], [], [])

print("\nBuy-and-Hold Strategy Metrics:")
print(f"Initial Capital: ${buy_and_hold_equity[0]:.2f}")
print(f"Final Capital: ${buy_and_hold_equity[-1]:.2f}")
for metric, value in buy_and_hold_metrics.items():
    if metric in ["Sharpe Ratio", "Win Rate", "Quality"]:
        print(f"{metric}: {value:.4f}")
    elif metric in ["Avg Win Yield", "Avg Loss Yield"]:
        print(f"{metric}: {value:.4%}")
    else:
        print(f"{metric}: {value:.2%}")

# Plot comparison with buy-and-hold
plt.figure(figsize=(12, 6))
plt.plot(range(len(train_equity)), train_equity, label='Model (Train)')
plt.plot(range(len(train_equity), len(train_equity) + len(test_equity)), test_equity, label='Model (Test)')
plt.plot(buy_and_hold_equity, label='Buy-and-Hold')
plt.title('Model vs Buy-and-Hold')
plt.xlabel('Trading Hours')
plt.ylabel('Capital')
plt.legend()
plt.show()

# Hasta aquí todo bien
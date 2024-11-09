from flask import Flask, request, render_template, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.metrics import accuracy_score, precision_score

app = Flask(__name__)

# Step 1: Fetch Data from Yahoo Finance
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="5y")  # Fetch last 5 years of data
    data.reset_index(inplace=True)  # Reset index for easier manipulation
    return data

# Step 2: Prepare the Data
def prepare_data(data):
    data['Date'] = pd.to_datetime(data['Date'])  # Convert 'Date' column to datetime
    data.set_index('Date', inplace=True)
    features = data.index.astype(int) / 10**9  # Convert dates to seconds for regression
    target = data['Close']
    return features.values.reshape(-1, 1), target.values

# Manual Ridge Regression Model (same as before)
class RidgeRegressionModel:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coefficients = None

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        A = np.dot(X_b.T, X_b) + self.alpha * np.eye(X_b.shape[1])
        b = np.dot(X_b.T, y)
        self.coefficients = np.linalg.solve(A, b)

    def predict(self, X):
        if self.coefficients is None:
            raise Exception("Model is not fitted yet.")
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return np.dot(X_b, self.coefficients)

# Plotting and Image Encoding for Web Display
def plot_predictions(data, model, years, features):
    last_date = data.index[-1]
    num_predictions = years * 252
    future_dates = pd.date_range(start=last_date, periods=num_predictions + 1, freq='B')[1:]
    future_X = (future_dates.astype(int) / 10**9).values.reshape(-1, 1)
    predictions = model.predict(future_X)

    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Historical Prices', color='blue', linewidth=2)
    historical_predictions = model.predict(features)
    plt.plot(data.index, historical_predictions, label='Fitted Prices (Training)', color='green', linewidth=3)
    plt.plot(future_dates, predictions, label='Predicted Prices', color='orange', linestyle='--', linewidth=2)
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Prediction using Manual Ridge Regression')
    plt.axvline(x=last_date, color='red', linestyle='--', label='Prediction Start')
    plt.legend()
    plt.grid()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return image_base64

# Function to calculate overall stock nature and growth/decline
def analyze_stock(data, predictions, future_dates):
    # Calculate rate of change over the last year
    last_year_data = data.tail(252)  # Approx 252 trading days in a year
    last_price = last_year_data['Close'].iloc[-1]
    first_price = last_year_data['Close'].iloc[0]
    rate_of_change = (last_price - first_price) / first_price * 100

    # Determine if price is increasing or decreasing in the future
    future_change = predictions[-1] - predictions[0]
    growth_rate = (future_change / predictions[0]) * 100

    # Determine if the stock is investible
    investible = "Investible" if growth_rate > 5 else "Not Investible"

    # Calculate the monthly predicted values
    monthly_predictions = predictions[::22]  # Roughly every 22 trading days in a month
    monthly_dates = future_dates[::22]
    
    # Prepare the output summary
    analysis = {
        'rate_of_change': rate_of_change,
        'growth_rate': growth_rate,
        'investible': investible,
        'monthly_predictions': list(zip(monthly_dates, monthly_predictions))
    }
    return analysis

# Web Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker']
    years = int(request.form['years'])

    data = fetch_stock_data(ticker)
    features, target = prepare_data(data)

    model = RidgeRegressionModel(alpha=1.0)
    model.fit(features, target)
    predictions = model.predict(features)
    
    # Predict future prices
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date, periods=years * 252 + 1, freq='B')[1:]
    future_X = (future_dates.astype(int) / 10**9).values.reshape(-1, 1)
    future_predictions = model.predict(future_X)
    
    accuracy = accuracy_score((target[1:] > target[:-1]).astype(int), (predictions[1:] > predictions[:-1]).astype(int))
    precision = precision_score((target[1:] > target[:-1]).astype(int), (predictions[1:] > predictions[:-1]).astype(int))

    # Get the analysis of the stock
    analysis = analyze_stock(data, future_predictions, future_dates)

    plot_image = plot_predictions(data, model, years, features)

    return render_template('result.html', image=plot_image, accuracy=accuracy * 100, precision=precision * 100,
                           rate_of_change=analysis['rate_of_change'], growth_rate=analysis['growth_rate'],
                           investible=analysis['investible'], monthly_predictions=analysis['monthly_predictions'])

if __name__ == '__main__':
    app.run(debug=True, port=5001)

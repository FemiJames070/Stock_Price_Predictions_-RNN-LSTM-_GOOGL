# 📈 Deep Learning for Algorithmic Trading: Temporal Price Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-3F4F75?style=flat)](https://www.statsmodels.org/)

## 📖 Project Overview
This repository contains a sophisticated time-series forecasting pipeline designed to predict short-term stock price movements (specifically GOOGL). By bridging rigorous statistical analysis with deep learning, this project evaluates the architectural efficiency of various Recurrent Neural Networks (Simple RNN, LSTM, and GRU) in capturing complex, long-term temporal dependencies in highly volatile financial markets.

**Dataset:** GOOGL Historical Daily Market Data via `yfinance` API (Jan 2020 – Nov 2025)

## 🎯 Business Value & Objectives
Financial markets are notoriously chaotic, non-linear, and non-stationary. Traditional forecasting falls short when dealing with dynamic market noise. This project was built to:
1. **Isolate Signal from Noise:** Apply deep learning to filter high-frequency market volatility and capture underlying momentum.
2. **Evaluate Temporal Architectures:** Directly compare how standard RNNs handle the vanishing gradient problem versus gated architectures (LSTM/GRU) when predicting financial data.
3. **Achieve High-Fidelity Forecasting:** Reconstruct scaled network outputs back into actionable USD metrics to determine true commercial viability.

## ⚙️ The Architecture: Engineering Temporal Understanding

The success of a deep learning financial model relies entirely on how the data is forged before it reaches the network. Feeding raw, trending stock prices into a neural network is an anti-pattern. Here is the methodology used to engineer stability:

### 1. Statistical Forging & Stationarity
* **Diagnosing Non-Stationarity:** Exploratory Data Analysis and an Augmented Dickey-Fuller (ADF) test ($p = 0.995$) mathematically proved the raw closing prices were non-stationary with a heavy upward trend. 
* **Taming Heteroscedasticity:** To resolve fluctuating volatility and shifting means, the pipeline utilizes **Differencing**. By training the network to predict *day-to-day price differentials* rather than raw absolute prices, the data was forced into strict stationarity (ADF $p = 0.000$).
* **Temporal Windowing:** The stationary data was scaled and sequenced into 60-day lookback windows, providing the networks with enough historical context to recognize cyclical patterns.

### 2. Network Architectures
Three competing models were constructed, utilizing identical hyperparameter footprints (2 recurrent layers with 50 units, 20% Dropout, and a Dense output) to ensure a strictly fair architectural comparison:
* **Simple RNN:** The baseline model, highly susceptible to the vanishing gradient problem over the 60-day lookback window.
* **LSTM (Long Short-Term Memory):** Utilizing complex cell states and three gates (input, output, forget) to selectively remember critical historical shifts and drop irrelevant noise.
* **GRU (Gated Recurrent Unit):** A streamlined alternative to the LSTM, utilizing only reset and update gates, designed to achieve similar temporal retention with a significantly lighter computational footprint.

## 📈 Model Performance & Results

The models were evaluated on a strictly unseen test set covering the highly volatile trading period of late 2025. The outputs were inverse-transformed from scaled data back into real-world currency (USD) to calculate the final Root Mean Squared Error (RMSE).

| Architecture | Test RMSE (USD) | Clinical / Business Implication |
| :--- | :---: | :--- |
| **LSTM** | **$3.88** | **The Optimal Predictor:** Exhibited the highest stability and tightest accuracy, successfully tracking aggressive price swings. |
| **GRU** | **$3.90** | **The Efficiency King:** Delivered near-identical accuracy to the LSTM but trained significantly faster due to fewer tensor operations. |
| **Simple RNN** | **$3.98** | **The Baseline:** Performed admirably due to the robust data differencing strategy, but struggled slightly with longer-term context. |

*Conclusion: For a stock fluctuating between $140 and $280 during the test window, an average prediction error of ~$3.88 represents a highly robust, functional baseline for short-term algorithmic forecasting.*

## 🛠️ Technology Stack
* **Language:** Python 3.10+
* **Deep Learning:** TensorFlow 2.x, Keras (RNN, LSTM, GRU architectures)
* **Statistical Analysis:** Statsmodels (ADF Testing)
* **Data Engineering:** Pandas, NumPy, Scikit-Learn (MinMaxScaler)
* **Data Sourcing:** `yfinance` API

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/stock-prediction-lstm.git](https://github.com/yourusername/stock-prediction-lstm.git)
   cd stock-prediction-lstm

2. **Install the required dependencies:**
   ```bash
   pip install tensorflow numpy pandas matplotlib scikit-learn yfinance statsmodels

3. **Run the Analysis:**
Execute the Jupyter Notebook to automatically fetch the latest GOOGL data, run the statistical stationarity tests, and train the deep learning models.

⚠️ Financial Disclaimer
For Educational and Research Purposes Only. This artificial intelligence model is designed for portfolio demonstration and algorithmic research. It is not financial advice, and should not be used to execute live trades or make real-world investment decisions.

## ✍️ Author
Femi James
Data & Business Analyst | Integrated AI Specialist

# 📈 Advanced Portfolio Optimization using LSTMs & Mean-Variance Optimization

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F79A3E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)

An end-to-end quantitative finance pipeline that combines **Long Short-Term Memory (LSTM)** neural networks with **Modern Portfolio Theory (MPT)**. The system predicts asset price trajectories using continuous historical time-series sequences and optimizes asset allocation using **Sequential Least Squares Programming (SLSQP)** to maximize the Sharpe ratio(highest return generated for the given level of risk).

---

## 🎯 Executive Summary & Key Results

Traditional Mean-Variance Optimization (MVO) relies heavily on historical moving statistics, which fail to capture dynamic temporal shifts in asset prices. This architecture bridges deep learning with classical portfolio construction by using an LSTM model to forecast trends and feeding the top-performing assets into a constrained numerical optimizer.

### 📊 Portfolio Performance Metrics

| Metric | Optimized Allocation Value | Target Benchmark / Notes |
| :--- | :---: | :--- |
| **Expected Annual Return** | **51.57%** | Derived from model-filtered top assets |
| **Annualized Risk (Volatility)** | **32.02%** | Standard deviation of portfolio daily returns |
| **Sharpe Ratio ($\mathbf{R_f = 4\%}$)** | **1.49** | Maximized via SLSQP numerical optimization |

### 💰 Optimal Asset Weight Distribution
- **GLD (SPDR Gold Shares)**: `39.24%`
- **NVDA (NVIDIA Corp)**: `44.25%`
- **Other Filtered Equities**: Remainder balanced dynamically under $\sum w_i = 1$ constraints.

---

## 🏗️ System Architecture & Workflow

1. **Data Ingestion & Preprocessing**:
   - Automated ingestion of historical daily pricing via `yfinance` across diversified tickers (`AAPL`, `MSFT`, `TSLA`, `AMZN`, `NVDA`, `GOOG`, `SPY`, `XLK`, `QQQ`, `GLD`).
   - Normalization using `MinMaxScaler(0, 1)` and execution of 60-day sliding window sequences for temporal feature learning.

2. **Predictive Modeling (Deep Learning)**:
   - **Architecture**: Sequential Stacked LSTM network (50 units each) with `Dropout(0.2)` layers to prevent overfitting.
   - **Optimization**: Adam optimizer with Mean Squared Error (MSE) loss function.
   - **Validation Metrics**: Model performance validated across evaluation metrics including RMSE, MAE, and MAPE.

3. **Asset Screening & Portfolio Construction**:
   - Converts price forecasts into predicted daily return series.
   - Computes 6-month moving averages across predicted returns to identify top-performing candidate assets.
   - Applies Markowitz Mean-Variance Optimization (SLSQP algorithm) bounded between $[0, 1]$ per asset with fully invested weight constraints.

---

## 🛠️ Tech Stack

- **Deep Learning**: TensorFlow, Keras
- **Quantitative Optimization**: SciPy (`scipy.optimize.minimize`)
- **Financial Data Processing**: `yfinance`, Pandas, NumPy, Scikit-Learn
- **Visualization**: Plotly Interactive Charts , Seaborn, Matplotlib

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/portfolio-optimization-lstm.git](https://github.com/YOUR_USERNAME/portfolio-optimization-lstm.git)
cd portfolio-optimization-lstm

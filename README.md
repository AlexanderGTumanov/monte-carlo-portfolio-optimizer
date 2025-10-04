# monte-carlo-portfolio-optimizer

Mean–variance optimization (MVO) is the classical framework for portfolio construction: given estimates of asset returns and covariances, it produces the portfolio weights that maximize the Sharpe ratio. While elegant in theory, MVO suffers from a critical weakness: its inputs are noisy. Small changes in estimated returns or volatilities can lead to dramatically different portfolios, making the allocations unstable and often unrealistic.

Monte Carlo (MC) resampling provides a way to address this instability. Instead of relying on a single historical estimate, the data is repeatedly resampled, with new parameters are estimated each time. Aggregating the resulting weights by taking their median yields a portfolio that is more robust to noise and less sensitive to individual data points. Compared to MVO, this portfolio places greater emphasis on stability and diversification rather than concentrating heavily in the top historical performers.

The project is organized into two main directories. The `/notebooks` folder contains a Jupyter notebook that demonstrates the construction of portfolios across different market regimes and compares the outputs of MVO and MC. The `/src` folder holds the source code file `utils.py`, which includes functions for data fetching, parameter estimation, optimization, resampling, evaluation, and visualization.

---

## What It Does

- Fetches daily price data from Yahoo Finance for a user-selected set of tickers  
- Computes daily returns and covariance matrices from the historical data  
- Builds MVO portfolios that maximize the Sharpe ratio using historical estimates  
- Runs Monte Carlo resampling to generate multiple resampled datasets  
- Optimizes portfolios on each resample and aggregates weights to form a robust MC portfolio  
- Evaluates and compares MVO and MC portfolios on out-of-sample test data  
- Visualizes results with weight comparisons, confidence bands, and performance metrics

---

## How to Use

1. Clone this repository:
   ```bash
   git clone <https://github.com/your-username/monte-carlo-portfolio-optimizer>
   cd monte-carlo-portfolio-optimizer

---

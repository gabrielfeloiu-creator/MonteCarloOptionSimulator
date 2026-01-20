# MonteCarloOptionSimulator

Interactive Python app for pricing European call options using Monte Carlo simulation, benchmarked against Black–Scholes and driven by real stock data.

## Features
- Monte Carlo pricing of European call options
- Black–Scholes benchmark comparison
- Real market data via yfinance
- Interactive parameter sliders (volatility, strike, maturity, rate)
- Visualizations: distributions, convergence, and price paths

## Key Assumptions
- Stock prices follow Geometric Brownian Motion
- Constant volatility and risk-free interest rate
- No dividends
- European-style options (exercise at maturity only)

## Tech Stack
Python, NumPy, Pandas, SciPy, Matplotlib, Seaborn, Streamlit, yfinance

## Run
```bash
pip install -r requirements.txt
streamlit run app.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import yfinance as yf
from scipy.stats import norm


# --------------------------- Streamlit Page Config ---------------------------
st.set_page_config(
    page_title="Monte Carlo Option Pricing",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------- Global Styling ---------------------------
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#0e1117",
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#eaeaea",
    "xtick.color": "#eaeaea",
    "ytick.color": "#eaeaea",
    "text.color": "#eaeaea",
    "grid.color": "#333333",
    "grid.alpha": 0.35,
    "font.family": "DejaVu Sans",
    "axes.titleweight": "bold",
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})
np.random.seed(42)

#  --------------------------- Sidebar Inputs ---------------------------
st.sidebar.header("Simulation Parameters")

# --------------------------- Single stock input ---------------------------
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper() # default stock is Apple
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2026-01-01"))
N = st.sidebar.number_input("Monte Carlo Simulations", value=10000, min_value=100, step=100)

# --------------------------- Multi-stock testing ---------------------------
multi_test = st.sidebar.checkbox("Run Multi-Stock Accuracy Test (~100 tickers)")

# --------------------------- Core Functions ---------------------------

# --------------------------- Simulate terminal prices ---------------------------
def simulate_terminal_prices(S0, T, r, sigma, N):
    Z = np.random.normal(0, 1, N)
    return S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)

# --------------------------- Find monte carlo call price ---------------------------
def monte_carlo_call_price(S0, K, T, r, sigma, N):
    ST = simulate_terminal_prices(S0, T, r, sigma, N)
    payoff = np.maximum(ST - K, 0)
    return np.exp(-r*T) * np.mean(payoff)

# --------------------------- simulate actual stock paths ---------------------------
def simulate_stock_paths(S0, T, r, sigma, n_steps=252, n_paths=75):
    dt = T / n_steps
    paths = np.zeros((n_steps + 1, n_paths))
    paths[0] = S0
    for t in range(1, n_steps + 1):
        Z = np.random.normal(0, 1, n_paths)
        paths[t] = paths[t-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
    return paths
# --------------------------- black scholes to determine price, and compare it to Monte Carlo  ---------------------------
def black_scholes_call(S0, K, T, r, sigma):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

# --------------------------- Load stock data ---------------------------
data = yf.download(ticker, start=start_date, end=end_date)
if data.empty:
    st.error("No data found for this ticker/date range.")
    st.stop()

S0_default = float(data['Close'].iloc[-1])
S0 = st.sidebar.number_input("Initial Stock Price (S0)", value=S0_default, step=1.0)
K = st.sidebar.number_input("Strike Price (K)", value=S0_default, step=1.0)
T = st.sidebar.number_input("Time to Maturity (Years)", value=1.0, min_value=0.01, step=0.1)
r = st.sidebar.number_input("Risk-free Rate (r)", value=0.02, min_value=0.0, max_value=0.5, step=0.01)
sigma = st.sidebar.number_input("Volatility (sigma)", value=0.5, min_value=0.01, max_value=2.0, step=0.01)

# --------------------------- Header ---------------------------
st.title(f"Monte Carlo Option Pricing for {ticker}")
st.markdown(f"""
This dashboard demonstrates **Monte Carlo option pricing** for {ticker} using **Geometric Brownian Motion (GBM)**.
Adjust the sidebar to see distributions, convergence, and simulated stock paths in real time.
""")

# --------------------------- Tabs ---------------------------
tab1, tab2, tab3 = st.tabs(["Distributions", "Convergence", "Stock Price Paths"])

# --------------------------- TAB 1: Distributions ---------------------------
with tab1:
    ST = simulate_terminal_prices(S0, T, r, sigma, N)
    payoffs = np.maximum(ST - K, 0)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.histplot(ST, bins=60, kde=True, stat="density", ax=ax, color="#4cc9f0", alpha=0.85)
        ax.axvline(K, linestyle="--", linewidth=2, color="#f72585", label="Strike Price")
        ax.set_title("Terminal Stock Price Distribution")
        ax.legend()
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.histplot(payoffs, bins=60, kde=True, stat="density", ax=ax, color="#80ffdb", alpha=0.85)
        ax.set_title("Call Option Payoff Distribution")
        st.pyplot(fig)

# --------------------------- TAB 2: Convergence ---------------------------
with tab2:
    N_values = [100, 500, 1000, 2500, 5000, 10000, 20000]
    mc_prices = [monte_carlo_call_price(S0, K, T, r, sigma, n) for n in N_values]

    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(N_values, mc_prices, marker="o", linewidth=2, color="#4cc9f0", label="Monte Carlo Estimate")
    ax.axhline(black_scholes_call(S0, K, T, r, sigma), linestyle="--", linewidth=2, color="#f72585", label="Black–Scholes Price")
    ax.set_xscale("log")
    ax.set_title("Monte Carlo Convergence to Black–Scholes Price")
    ax.set_xlabel("Number of Simulations (log scale)")
    ax.set_ylabel("Option Price")
    ax.legend()
    st.pyplot(fig)

# --------------------------- TAB 3: Stock Price Paths ---------------------------
with tab3:
    n_steps = 252
    n_paths = 75
    paths = simulate_stock_paths(S0, T, r, sigma, n_steps, n_paths)
    time_grid = np.linspace(0, T, n_steps+1)

    fig, ax = plt.subplots(figsize=(11,5))
    for i in range(n_paths):
        ax.plot(time_grid, paths[:,i], color="#4cc9f0", alpha=0.25, linewidth=1)
    ax.axhline(S0, linestyle="--", linewidth=2, color="white", alpha=0.6, label="Initial Price")
    ax.axhline(K, linestyle="--", linewidth=2, color="#f72585", label="Strike Price")
    ax.set_title("Simulated Stock Price Paths Over Time (GBM)")
    ax.set_xlabel("Time (Years)")
    ax.set_ylabel("Stock Price")
    ax.legend()
    st.pyplot(fig)

# --------------------------- Metrics for single stock ---------------------------
st.markdown("---")
col1, col2 = st.columns(2)
bs_price = black_scholes_call(S0, K, T, r, sigma)
mc_price = monte_carlo_call_price(S0, K, T, r, sigma, N)
col1.metric("Black–Scholes Price", f"{bs_price:.2f}")
col2.metric("Monte Carlo Price", f"{mc_price:.2f}")

# --------------------------- Multi-stock Accuracy Test (~100 tickers) ---------------------------
if multi_test:
    st.markdown("### Multi-Stock Monte Carlo Accuracy Test")
    # Example: use S&P 100 tickers
    tickers_list = [
        "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","BRK-B","JPM","V","UNH",
        "HD","PG","MA","DIS","PYPL","NFLX","KO","PFE","PEP","CSCO","VZ","ADBE",
        "INTC","T","CRM","CMCSA","ABT","NKE","ORCL","XOM","CVX","WMT","COST","ACN",
        "C","QCOM","LLY","AVGO","TXN","HON","IBM","MDT","UPS","NEE","LIN","LOW",
        "PM","BA","SBUX","GS","MS","RTX","AMGN","ISRG","BLK","CAT","SPGI","EL","NOW",
        "GILD","BKNG","TMO","SYK","MMM","ANTM","GE","ADI","DE","LMT","MDLZ","COP",
        "FIS","PLD","ZTS","REGN","MO","SCHW","CI","CI","CCI","DUK","CCI","SO","EQIX",
        "PNC","USB","MMC","HCA","TJX","FISV","ICE","BDX","WM","BSX","ADP","APD","ECL",
        "ITW","CSX","NSC","CL","SHW"
    ]
    results = []

    for sym in tickers_list:
        try:
            hist = yf.download(sym, start=start_date, end=end_date)
            if hist.empty:
                continue
            S0_hist = hist['Close'].iloc[-1]
            K_hist = S0_hist
            returns = hist['Close'].pct_change().dropna()
            sigma_hist = returns.std() * np.sqrt(252)
            # Monte Carlo price
            mc_price = monte_carlo_call_price(float(S0_hist), float(K_hist), T, r, float(sigma_hist), N)
            # Black-Scholes price
            bs_price_hist = black_scholes_call(float(S0_hist), float(K_hist), T, r, float(sigma_hist))
            error_pct = abs(mc_price - bs_price_hist) / bs_price_hist * 100
            results.append({
                "Ticker": sym,
                "MC Price": mc_price,
                "BS Price": bs_price_hist,
                "Error %": error_pct
            })
        except Exception as e:
            continue

    if results:
        df_results = pd.DataFrame(results).sort_values("Error %")
        st.dataframe(df_results)
        st.markdown(f"**Average Error %:** {df_results['Error %'].mean():.2f}%")
    else:
        st.warning("No valid stock data found for the multi-stock test.")

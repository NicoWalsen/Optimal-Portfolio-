#IMPORTING LIBRARIES AND MODULES
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

#SET A LIST OF THE TICKERS WE WILL INCLUDE IN OUR PORTFOLIO
tickers = ["SPY", "BND", "GLD", "QQQ", "VTI"]

#Set the end date to today
end_date = datetime.today()

#SET THE START DATE TO 2' YEARS AGO
start_date = end_date - timedelta(days = 20*365) # we will use 20 years from today to the past.
print (start_date)
print (end_date)

##DOWNLOAD ADJUSTED CLOSE PRICES##

#Create an empty DataFrame to store the adjusted close prices
adj_close_df = pd.DataFrame()

#Downloading the close prices for each ticker, creating a dataframe called "adj_close_df"
for ticker in tickers: 
    data = yf.download(ticker,start=start_date,end=end_date)
    adj_close_df[ticker] = data['Adj Close']

#Display the DataFrame
adj_close_df

#Calculate the lognormal returns for each sticker
log_returns = np.log(adj_close_df/adj_close_df.shift(1)) 

#Drop any missing values
log_returns = log_returns.dropna()

##CALCULATE COVARIANCE MATRIX##
cov_matrix = log_returns.cov()*252  #We assume 252 working days in a year.
cov_matrix

## DEFINING PORTFOLIO PERFORMANCE METRICS ##

#Calculate the portfolio standars deviation
def standard_deviation (weights, cov_matrix):
    variance = weights.T @ cov_matrix @weights
    return np.sqrt(variance)

#Calculate the expected return* Key assumption: Expected returns are based on historical returns
def expected_return (weights,log_returns):
    return np.sum(log_returns.mean()*weights)*252


#Calculate the Sharpe Ratio
def sharpe_ratio (weights, log_returns, cov_matrix, risk_free_rate):
    return (expected_return (weights,log_returns)-risk_free_rate)/standard_deviation (weights, cov_matrix)

#Set the risk-free rate to 2% by agreement.
risk_free_rate = 0.02

##DEFINE THE FUNCTION TO MINIMIZE (NEGATIVE SHARPE RATIO)
def neg_sharpe_ratio(weights, log_retunrs, cov_matrix, risk_free_rate):
    return -sharpe_ratio (weights, log_returns, cov_matrix, risk_free_rate)

##SET THE CONSTRAINS AND BOUNDS
constrains = {'type': 'eq', 'fun': lambda weights: np.sum(weights)-1}  
bounds = [(0, 0.9) for _ in range(len(tickers))]  #We will restrict maximum weight for 1 ticker as less than 90%.

#SET THE INITIAL WEIGHTS
initial_weights = np.array([1/len(tickers)]*len(tickers)) #Creating fixed initial weights as 20% for each ticker.

#Optimize the weights to maximize Sharpe Ratio
optimized_results = minimize(neg_sharpe_ratio, initial_weights, args=(log_returns, cov_matrix, risk_free_rate), method = 'SLSQP', constraints = constrains, bounds = bounds)

#GET THE OPTIMAL WEIGHTS
optimal_weights = optimized_results.x


print("Optimal_Weights:")
for ticker, weight in zip(tickers, optimal_weights):
    print (f"{ticker}: {weight:.4f}")
    
    print()
    
    optimal_portfolio_return = expected_return(optimal_weights, log_returns)
    optimal_portfolio_volatility = standard_deviation(optimal_weights, cov_matrix)
    optimal_sharpe_ratio = sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate)
    
    print(f"Expected Annual Return: {optimal_portfolio_return:.4f}")
    print(f"Expected Volatility: {optimal_portfolio_volatility:.4f}")
    print(f"Sharpe Ratio:{optimal_sharpe_ratio:.4f}")

#DISPLAY THE FINAL PORTFOLIO IN A PLOT#
plt.figure(figsize = (10,6))
plt.bar (tickers, optimal_weights)
plt.xlabel('Assets')
plt.ylabel('Optimal weights')
plt.title('Optimal Portfolio Weights')
plt.grid()

plt.show()

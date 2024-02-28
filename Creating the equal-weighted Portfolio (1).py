#Importing modules and libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.display.float_format = '{:.2f}'.format

#We download the ticker data from Yahoo Finance with the method yf.download().
#Then we save the stock data into a csv file "stocks".
stocks = yf.download(["AMZN","BA","DIS","IBM","KO","MSFT"], start = "2013-12-31", end = "2018-12-31") #download the  price info for those 6 tickers from Yahoo Finance!
stocks = stocks.to_csv("stocks.csv") #we create a csv file called "stocks" with the ticker information.
#Reading the csv file "stocks". including the date column as index.
stocks = pd.read_csv("stocks.csv",header = [0,1], index_col = [0], parse_dates = [0])
stocks.head()
stocks.tail()
#Creating a dataframe including only the column "Adj Close" and saving this as a csv file "port_stocks.csv"
stocks = stocks['Adj Close'].copy() 
stocks.head()
stocks.to_csv("port_stocks.csv") 
#We read the new csv file that contains only the Adj Close info.
stocks = pd.read_csv("port_stocks.csv", parse_dates = ["Date"], index_col = "Date")
stocks.head()
#We utilize the panda method pct_change() to obtain the returns, also we delete any possible empty cell. #We utilize the panda method pct_change() to obtain the returns, also we delete any possible empty cell.
ret = stocks.pct_change().dropna()
ret.head()
ret.shape
ret.mean(axis=1) # we calculate the mean of the returns for each stock

#We create a list with 6 times 1/6, that represents the equal weights.
no_assets = len(stocks.columns)
weights = [1/no_assets for i in range(no_assets)] 
weights
#We multiply the matrix "ret" with "weights" obtaining daily returns for each equal-weighted portfolio.
ret.dot(weights) 
#Adding the column "EWP" to our dataframe "ret". That column includes daily returns for the equal weighted portfolio.
ret["EWP"] = ret.dot(weights) 
ret.head()

summary = ret.agg(["mean","std"]).T #Method .agg() allows you to apply a function or a list of function names to be executed along one of the axis of the DataFrame, default 0, which is the index (row) axis. Además, trasponemos la matriz para que se vea invertida.
summary
#Then we change the names of the columns to "Return" and "Risk" and we anualize both the Return and the Risk
summary.columns = ["Return","Risk"] 
summary.Return = summary.Return*252 
summary.Risk = summary.Risk*np.sqrt(252) 
summary

#We plot each of our stocks and equal weighted portfolio
summary.plot(kind = "scatter", x= "Risk", y = "Return", figsize = (13,9), s= 50, fontsize = 15)
for i in summary.index:
    plt.annotate(i, xy=(summary.loc[i, "Risk"]+0.002, summary.loc[i,"Return"]+0.002), size=15)
plt.xlabel("ann, Risk(std)",fontsize=15)
plt.ylabel("ann, Return", fontsize=15)
plt.title ("Risk/Return", fontsize=20)
plt.grid()
plt.show()
#We define the function ann_risk_return() to add that we make before more efficiently.
def ann_risk_return(returns_df):
    summary = returns_df.agg(["mean","std"]).T
    summary.columns =["Return","Risk"]
    summary.Return = summary.Return*252
    summary.Risk = summary.Risk*np.sqrt(252)
    return summary

summary = ann_risk_return(ret) 
summary

noa = len(stocks.columns)
noa #number of assets
nop = 100000
nop #number of random porfolios 
#We will create random list with numbers with 10 files and 6 columns.
np.random.random(10*6).reshape(10,6) 
#If we run the previous cell once again, we will get different numbers each time. To avoid this we must include np.random.seed() method.
np.random.seed(123) # If we run the previous code again, we will get always the same numbers [if we pass the number "123" to the method np.random.seed()]
#we run again the same code that before but using "noa" and "nop"
matrix = np.random.random(noa*nop).reshape(nop,noa)
matrix
#We generate the sum for each file of the previous matrix (axis=1). Keepdims = True means that we will mantain the same dimentions that the matrix of origin.
matrix.sum(axis=1, keepdims=True) 

#Now, we are going to normalize the values of the matrix "matrix". In order to do it, we will divide each of the values for file for the value calculated en each file of the matrix of 1 columns and 10 files.
#Moreover, we obtain that the sum of each file is equal to 1 and we acomplish the first restriction for our portfolios.
weights = matrix / matrix.sum(axis=1, keepdims=True)
weights 
weights.sum(axis=1, keepdims=True) #demostrating that the sum of each file is equal to 1.

port_ret = ret.dot(weights.T) #We multiply the matrix returns with transposed weights in order to obtain the daily returns for each of our 10 portfolios. 
port_ret
port_summary = ann_risk_return(port_ret) #our function will provide both the anualized return and risk for each of our 100.000 portfolios.
port_summary 

# We graph the stocks and equal weighted portfolio and we also add our 100.000 random portfolios. 
plt.figure(figsize = (15,9))
plt.scatter(port_summary.loc[:,"Risk"], port_summary.loc[:,"Return"], s=20, color = "r")
plt.scatter(summary.loc[:,"Risk"], summary.loc[:,"Return"], s=50, color ="black", marker ="D")
plt.xlabel("ann, Risk(std)",fontsize=15)
plt.ylabel("ann, Return", fontsize=15)
plt.title ("Risk/Return", fontsize=20)
plt.grid()
plt.show()

#We will get the Sharpe ratios for each of our 100.000 random portfolios.
risk_free_return= 0.017 #as a postulation.
risk_free_risk = 0
#Creating a list with the risk free return and risk free risk called "rf".
rf = [risk_free_return,risk_free_risk]
rf

#Including Sharpe Ratio for our 6 stocks.
summary["Sharpe"]= (summary["Return"]-rf[0])/summary["Risk"] #We calculate the Sharpe Ratio by definition.
summary #Best performed stock is Microsoft! as has the higher Sharpe Ratio.
#Including Sharpe Ratio for our 100.000 random portfolios.
port_summary["Sharpe"] = (port_summary["Return"]-rf[0])/port_summary["Risk"]
port_summary
port_summary.describe()

#We plot our 100.000 stocks and their Sharpe Ratio, higher sharpe ratios are strong red and lower are strong blue.
#We also graph our 6 initial stocks.
plt.figure(figsize = (15,8))
plt.scatter(port_summary.loc[:,"Risk"], port_summary.loc[:,"Return"], s=20,
            c=port_summary.loc[:,"Sharpe"], cmap = "coolwarm", vmin = 0.75, vmax = 1.18, alpha = 0.8)
plt.colorbar()
plt.scatter(summary.loc[:,"Risk"], summary.loc[:,"Return"], s=50, color ="black", marker ="D")
plt.xlabel("ann, Risk(std)",fontsize=15)
plt.ylabel("ann, Return", fontsize=15)
plt.title ("Sharpe Ratio", fontsize=20)
plt.grid()
plt.show()

#Looking forward to the higher Sharpe ratio random portfolio.
port_summary.head()
port_summary.describe()
weights
msrp = port_summary.Sharpe.idxmax() #We will get the number of file with the higher Sharpe Ratio with the method .idmax()
msrp
msrp_p = port_summary.iloc[msrp] #We found that the higher Sharpe ratio is in the file 76879, also, we get the descriptive statistics.
msrp_p
msrp_w = weights[msrp,:] # With this we obtain the weights for this portfolio.
msrp_w
#Finally we get the weights for each assets that represents our random portfolio with the maximum Sharpe ratio.
pd.Series(index=stocks.columns, data = msrp_w) 

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import yfinance as yf


# In[2]:


stocks = yf.download(["AMZN","BA","DIS","IBM","KO","MSFT"], start = "2013-12-31", end = "2018-12-31")


# In[3]:


stocks = stocks.to_csv("stocks.csv")


# In[4]:


stocks = pd.read_csv("stocks.csv",header = [0,1], index_col = [0], parse_dates = [0])


# In[5]:


stocks.head()


# In[6]:


stocks.tail()


# In[7]:


stocks = stocks['Adj Close'].copy()


# In[8]:


stocks.head()


# In[9]:


stocks.to_csv("port_stocks.csv")


# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.display.float_format = '{:.2f}'.format


# In[11]:


stocks = pd.read_csv("port_stocks.csv", parse_dates = ["Date"], index_col = "Date")


# In[12]:


stocks.head()


# In[40]:


ret = stocks.pct_change().dropna()
ret.head()
ret.shape


# In[14]:


ret.mean(axis=1)


# In[15]:


no_assets = len(stocks.columns)
no_assets


# In[16]:


weights = [1/no_assets for i in range(no_assets)]
weights


# In[17]:


ret.mul(weights, axis = "columns").sum(axis=1) #forma larga de hacerlo


# In[18]:


ret.dot(weights) #multiplicamos matrices: en este caso la de retornos con la de pesos . Esto nos da los retornos diarios del portafolio equal-weighted


# In[19]:


ret["EWP"] = ret.dot(weights) #Agregamos la columna EWP "Equaly-weighted portfolio" a nuestro dataframa "ret"


# In[20]:


ret.head()


# In[21]:


summary = ret.agg(["mean","std"]).T # el método .agg() llows you to apply a function or a list of function names to be executed along one of the axis of the DataFrame, default 0, which is the index (row) axis. Además, trasponemos la matriz para que se vea invertida.
summary


# In[22]:


summary.columns = ["Return","Risk"] #Cambiamos los nombres de las columnas a "Return" y "Risk"


# In[23]:


summary.Return = summary.Return*252 #Anualizamos el retorno multiplicandolo por 252 días hábiles
summary.Risk = summary.Risk*np.sqrt(252) # Anualizamos la desviación estándar multiplicándola por la raíz de 252.


# In[24]:


summary


# In[25]:


summary.plot(kind = "scatter", x= "Risk", y = "Return", figsize = (13,9), s= 50, fontsize = 15)
for i in summary.index:
    plt.annotate(i, xy=(summary.loc[i, "Risk"]+0.002, summary.loc[i,"Return"]+0.002), size=15)
plt.xlabel("ann, Risk(std)",fontsize=15)
plt.ylabel("ann, Return", fontsize=15)
plt.title ("Risk/Return", fontsize=20)
plt.grid()
plt.show()


# In[45]:


def ann_risk_return(returns_df):
    summary = returns_df.agg(["mean","std"]).T
    summary.columns =["Return","Risk"]
    summary.Return = summary.Return*252
    summary.Risk = summary.Risk*np.sqrt(252)
    return summary


# In[46]:


#Creating many Random Portfolios

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.display.float_format = '{:.2f}'.format


# In[47]:


summary = ann_risk_return(ret) #Usamos la función para que arroje el summary en un paso


# In[48]:


summary


# In[49]:


noa = len(stocks.columns)
noa #number of assets


# In[50]:


nop = 100000
nop #number of random porfolios that we will simulate


# In[32]:


np.random.random(10*6).reshape(10,6) #arroja números aleatorios de 10 filas y 6 columnas


# In[51]:


np.random.seed(123) # si volvemos a correr la celda anterior, obtendremos números diferentes cada vez.
#Para evitar esto, haremos que sea reproducible y no varien los números (siempre que pasemos el número "123") y así se muestren siempre los mismos números.
matrix = np.random.random(noa*nop).reshape(nop,noa)


# In[52]:


matrix #ahora si volvemos a correr el código de arriba, obtendremos exactamente  los mismos números, ya que usamos el método mencionado .seed(123)


# In[53]:


matrix.sum(axis=1, keepdims=True) #Generamos la suma por cada fila de la matriz anterior (axis=1), keepdims = True es para que mantenga las dimenciones del número de filas.


# In[54]:


#ahora queremos normalizar los valores, de la matriz "matrix", para hacer esto, dividimos cada unod e los valores por fila de la matrix por el valor calculado en cada fila de la matriz de 1 columna y 10 filas anterior. 
#De esta manera obtendremos que la suma de todos los valores por fila será igual a 1, y cumpliremos la primera restriccion para nuestros porfatfolios
weights = matrix / matrix.sum(axis=1, keepdims=True)


# In[55]:


weights #ahora los valores están normalizados por filas (deberían sumar uno).
#De esta manera podemos crear 10 portafolios simulados.


# In[56]:


#comprobemos si suman 1

weights.sum(axis=1, keepdims=True) #suman 1! Vamos por el camino correcto


# In[57]:


port_ret = ret.dot(weights.T) #multiplicamos la matriz de returns y weights transpuesta, para obtener retornos diarios de nuestros 10 portfolios
port_ret


# In[58]:


port_summary = ann_risk_return(port_ret)


# In[59]:


port_summary #entrega el returno y riesgo anualizado para cada uno de nuestros 100000 portfolios


# In[60]:


plt.figure(figsize = (15,9))
plt.scatter(port_summary.loc[:,"Risk"], port_summary.loc[:,"Return"], s=20, color = "r")
plt.scatter(summary.loc[:,"Risk"], summary.loc[:,"Return"], s=50, color ="black", marker ="D")
plt.xlabel("ann, Risk(std)",fontsize=15)
plt.ylabel("ann, Return", fontsize=15)
plt.title ("Risk/Return", fontsize=20)
plt.grid()
plt.show()


# In[63]:


risk_free_return= 0.017 #supuesto
risk_free_risk = 0


# In[65]:


rf = [risk_free_return,risk_free_risk]
rf


# In[66]:


summary


# In[68]:


port_summary.head()


# In[70]:


summary["Sharpe"]= (summary["Return"]-rf[0])/summary["Risk"]
summary #Best performance is MSFT, porque tiene el mayor Sharpe Ratio


# In[73]:


port_summary["Sharpe"] = (port_summary["Return"]-rf[0])/port_summary["Risk"]
port_summary


# In[74]:


port_summary.describe()


# In[75]:


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


# In[76]:


#Vamos a identificar el portfolio con el mayor Sharpe Ratio


# In[77]:


port_summary.head()


# In[78]:


port_summary.describe()


# In[79]:


weights


# In[82]:


msrp = port_summary.Sharpe.idxmax() #Entrega el número de fila en que se encuentra el portfolio con mayor Sharpe Ratio
msrp


# In[83]:


msrp_p = port_summary.iloc[msrp] #encontramos el porfolio de la fila 76879, junto a sus estadísticas descriptibas 
msrp_p


# In[84]:


msrp_w = weights[msrp,:] # obtenemos los pesos del portfolio con mayr Sharpe Ratio
msrp_w


# In[85]:


pd.Series(index=stocks.columns, data = msrp_w) #obtenemos los weights de las distintas acciones que constituyen nuestro mejor portafolio random (entre 100000)


# In[ ]:





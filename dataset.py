from ucimlrepo import fetch_ucirepo 
import pandas as pd
  
# fetch dataset 
online_retail = fetch_ucirepo(id=352) 
  
# data (as pandas dataframes) 
X = online_retail.data.features 
y = online_retail.data.targets 
  
# metadata 
df = pd.concat([X, y], axis=1)
df.to_csv("online_retail.csv", index=False)

 
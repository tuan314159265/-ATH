import integration.intergration as it
import cleaning.cleaning as cls
import merge_data.merge as merge
import tranformation_data.tranformation as tf
import pandas as pd


df1 = pd.read_csv("raw_data/kaggle_retail_sales.csv")
df2 = pd.read_csv("raw_data/ord.csv")
df3 = pd.read_csv("raw_data/uci_online_retail.csv")

print('kaggle',df1.columns.tolist(),'\n')
print('ord',df2.columns.tolist(),'\n')
print('uci',df3.columns.tolist(),'\n')

tf.transform()







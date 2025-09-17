import src

df1['CustomerID'] = df1['CustomerID'].dropna().astype(int).astype(str)
df1['CustomerID'] = "CUST" + df1['CustomerID']

df2['Customer ID'] = df2['Customer ID'].astype(str)
for i in df2.index:
    df2.loc[i, 'Customer ID'] = df2.loc[i, 'Customer ID'][:4] + "00" + df2.loc[i, 'Customer ID'][5:]

print(df1['CustomerID'].head())
print(df2['Customer ID'].head())
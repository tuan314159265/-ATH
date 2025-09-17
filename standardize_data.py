import src

src.df1['CustomerID'] = src.df1['CustomerID'].dropna().astype(int).astype(str)
src.df1['CustomerID'] = "CUST" + src.df1['CustomerID']

src.src.df2['Customer ID'] = src.df2['Customer ID'].astype(str)
for i in src.df2.index:
    src.df2.loc[i, 'Customer ID'] = src.df2.loc[i, 'Customer ID'][:4] + "00" + src.df2.loc[i, 'Customer ID'][5:]

print(src.df1['CustomerID'].head())
print(src.df2['Customer ID'].head())
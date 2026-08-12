import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("data/sales_eda_raw.csv")
print(df.shape); print(df.isna().sum()); print("Duplicates:",df.duplicated().sum())
df=df.drop_duplicates().reset_index(drop=True)
for c in ["Unit_Price","Discount","Profit"]: df[c]=df[c].fillna(df[c].median())
df["Region"]=df["Region"].fillna(df["Region"].mode()[0])
q1,q3=df.Quantity.quantile([.25,.75]); iqr=q3-q1
df["Quantity"]=df["Quantity"].clip(q1-1.5*iqr,q3+1.5*iqr)
df["Date"]=pd.to_datetime(df.Date); df["Sales"]=df.Quantity*df.Unit_Price*(1-df.Discount)
df.to_csv("data/sales_eda_cleaned.csv",index=False)
print("Total Sales:",df.Sales.sum()); print("Total Profit:",df.Profit.sum())
print("Top Product:",df.groupby("Product").Sales.sum().idxmax())

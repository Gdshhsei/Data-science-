import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

df=pd.read_csv("data/retail_sales_raw.csv")
print(df.shape, df.isna().sum(), "duplicates:", df.duplicated().sum())
df=df.drop_duplicates().reset_index(drop=True)
for c in ["Unit_Price","Discount"]: df[c]=df[c].fillna(df[c].median())
for c in ["Region","Channel"]: df[c]=df[c].fillna(df[c].mode()[0])
q1,q3=df.Units_Sold.quantile([.25,.75]); iqr=q3-q1
df.Units_Sold=df.Units_Sold.clip(q1-1.5*iqr,q3+1.5*iqr)
df.Date=pd.to_datetime(df.Date); df["Year"]=df.Date.dt.year; df["Month"]=df.Date.dt.month; df["DayOfWeek"]=df.Date.dt.dayofweek
df.Revenue=df.Units_Sold*df.Unit_Price*(1-df.Discount); df.to_csv("data/retail_sales_cleaned.csv",index=False)
features=["Product","Category","Region","Channel","Promotion","Unit_Price","Discount","Year","Month","DayOfWeek"]
X=df[features]; y=df.Units_Sold
cat=["Product","Category","Region","Channel","Promotion"]; num=["Unit_Price","Discount","Year","Month","DayOfWeek"]
pre=ColumnTransformer([("num",SimpleImputer(strategy="median"),num),("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),cat)])
model=Pipeline([("pre",pre),("rf",RandomForestRegressor(n_estimators=180,max_depth=10,random_state=42))])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42); model.fit(Xtr,ytr); p=model.predict(Xte)
print("MAE:",mean_absolute_error(yte,p)); print("RMSE:",mean_squared_error(yte,p)**.5); print("R2:",r2_score(yte,p))

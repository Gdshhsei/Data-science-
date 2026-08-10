import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

df=pd.read_csv("data/customer_churn_raw.csv")
print("Initial shape:",df.shape)
print("Missing values:\n",df.isna().sum())
print("Duplicates:",df.duplicated().sum())
df=df.drop_duplicates().reset_index(drop=True)
q1,q3=df.Monthly_Charges.quantile([.25,.75]); iqr=q3-q1
df.Monthly_Charges=df.Monthly_Charges.clip(q1-1.5*iqr,q3+1.5*iqr)
df.to_csv("data/customer_churn_cleaned.csv",index=False)
X=df.drop(columns=["Churn","Customer_ID"]); y=df.Churn.map({"No":0,"Yes":1})
cat=X.select_dtypes("object").columns.tolist(); num=X.select_dtypes(exclude="object").columns.tolist()
pre=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),num),("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),cat)])
Xt,Xv,yt,yv=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
model=Pipeline([("pre",pre),("rf",RandomForestClassifier(n_estimators=250,max_depth=8,class_weight="balanced",random_state=42))])
model.fit(Xt,yt); pred=model.predict(Xv); proba=model.predict_proba(Xv)[:,1]
print("Accuracy:",accuracy_score(yv,pred)); print("Precision:",precision_score(yv,pred)); print("Recall:",recall_score(yv,pred)); print("F1:",f1_score(yv,pred)); print("Confusion Matrix:\n",confusion_matrix(yv,pred)); fpr,tpr,_=roc_curve(yv,proba); print("ROC-AUC:",auc(fpr,tpr))

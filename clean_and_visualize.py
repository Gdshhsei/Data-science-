import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/sales_raw.csv")
print("Initial shape:", df.shape)
print("Missing values before cleaning:\n", df.isna().sum())
print("Duplicate rows:", df.duplicated().sum())

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
for c in ["Unit_Price","Discount","Profit","Quantity","Sales"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["Region"] = df["Region"].fillna(df["Region"].mode()[0])
for c in ["Unit_Price","Discount","Profit"]:
    df[c] = df[c].fillna(df[c].median())

df = df.drop_duplicates().reset_index(drop=True)

q1, q3 = df["Quantity"].quantile([0.25, 0.75])
iqr = q3-q1
lower, upper = q1-1.5*iqr, q3+1.5*iqr
outliers = ((df["Quantity"] < lower) | (df["Quantity"] > upper)).sum()
print("Quantity outliers detected:", int(outliers))
df["Quantity"] = df["Quantity"].clip(lower=lower, upper=upper)

df["Sales"] = df["Quantity"] * df["Unit_Price"] * (1-df["Discount"])
df["Profit"] = df["Profit"].clip(lower=0)
df.to_csv("data/sales_cleaned.csv", index=False)

# Visualizations
monthly = df.groupby(df["Date"].dt.month)["Sales"].sum()
monthly.plot(marker="o", title="Monthly Sales Trend")
plt.xlabel("Month"); plt.ylabel("Sales"); plt.grid(alpha=.25); plt.tight_layout()
plt.savefig("outputs/monthly_sales_trend.png", dpi=180); plt.close()

df.groupby("Category")["Sales"].sum().sort_values(ascending=False).plot(kind="bar", title="Sales by Category")
plt.ylabel("Sales"); plt.xticks(rotation=20); plt.tight_layout()
plt.savefig("outputs/sales_by_category.png", dpi=180); plt.close()

df.groupby("Region")["Profit"].sum().sort_values(ascending=False).plot(kind="bar", title="Profit by Region")
plt.ylabel("Profit"); plt.tight_layout()
plt.savefig("outputs/profit_by_region.png", dpi=180); plt.close()

print("\nTotal Sales:", round(df["Sales"].sum(),2))
print("Total Profit:", round(df["Profit"].sum(),2))
print("Top Product:", df.groupby("Product")["Sales"].sum().idxmax())
print("Top Category:", df.groupby("Category")["Sales"].sum().idxmax())
print("Top Profit Region:", df.groupby("Region")["Profit"].sum().idxmax())

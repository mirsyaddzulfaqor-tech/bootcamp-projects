"""Builds and executes the notebooks in this repo (run: python build_notebooks.py)."""
import sys
sys.path.insert(0, ".")
from nbrun import Notebook

SETUP = """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
"""

# ---------------------------------------------------------------- salaries
nb = Notebook(".")
nb.md("""
# Data Science Salaries: What Pays and Where?

**Questions:** how do experience, job title, company size and remote work affect data-science pay, and how does
Indonesia compare? Rebuilt from my Digital Skola *Wrangler Project* (data wrangling with pandas).

**Data:** 3,755 salary records from 2020–2023, in USD (`data/ds_salaries.csv`, public dataset).
""")
nb.code(SETUP)
nb.code("""
df = pd.read_csv("data/ds_salaries.csv")
LEVEL = {"EN": "Entry", "MI": "Mid", "SE": "Senior", "EX": "Executive"}
SIZE = {"S": "Small", "M": "Medium", "L": "Large"}
REMOTE = {0: "On-site", 50: "Hybrid", 100: "Fully remote"}
df["level"] = df["experience_level"].map(LEVEL)
df["company"] = df["company_size"].map(SIZE)
df["work_mode"] = df["remote_ratio"].map(REMOTE)
print(df.shape, "| missing:", df.isna().sum().sum(), "| duplicate rows:", df.duplicated().sum())
df["work_year"].value_counts().sort_index()
""")
nb.md("""
Exact duplicate rows are kept: in salary surveys, two people with the same title and pay are plausible.
Most records come from 2022–2023, so yearly trends are compared with that in mind.
""")
nb.code("""
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
order = ["Entry", "Mid", "Senior", "Executive"]
sns.boxplot(data=df, x="level", y="salary_in_usd", order=order, ax=axes[0], showfliers=False, color="#4C72B0")
axes[0].set(title="Salary by experience level", xlabel="", ylabel="USD per year")
sns.barplot(data=df, x="work_mode", y="salary_in_usd", estimator=np.median, errorbar=None,
            order=["On-site", "Hybrid", "Fully remote"], ax=axes[1], color="#4C72B0")
axes[1].set(title="Median salary by work mode", xlabel="", ylabel="")
sns.barplot(data=df, x="company", y="salary_in_usd", estimator=np.median, errorbar=None,
            order=["Small", "Medium", "Large"], ax=axes[2], color="#4C72B0")
axes[2].set(title="Median salary by company size", xlabel="", ylabel="")
plt.tight_layout()
summary = pd.concat({
    "by level": df.groupby("level")["salary_in_usd"].median().reindex(order),
    "by work mode": df.groupby("work_mode")["salary_in_usd"].median(),
    "by company size": df.groupby("company")["salary_in_usd"].median(),
})
summary.rename("median_usd").to_frame()
""")
nb.code("""
roles = (df.groupby("job_title")["salary_in_usd"].agg(["count", "median"])
           .query("count >= 30").sort_values("median", ascending=False))
ax = roles["median"].sort_values().plot(kind="barh", figsize=(8, 5), color="#4C72B0")
ax.set(title="Median salary by job title (titles with 30+ records)", xlabel="USD per year", ylabel="")
roles
""")
nb.code("""
fully_remote_share = (df.groupby("work_year")["remote_ratio"].apply(lambda s: (s == 100).mean()) * 100).round(1)
median_by_year = df.groupby("work_year")["salary_in_usd"].median()
print("Share of fully remote roles by year (%):")
print(fully_remote_share.to_string())
print("\\nMedian salary by year (USD):")
print(median_by_year.to_string())
""")
nb.code("""
idn = df[df["company_location"] == "ID"]
asia = ["ID", "IN", "SG", "MY", "PH", "TH", "VN", "JP"]
asia_med = (df[df["company_location"].isin(asia)].groupby("company_location")["salary_in_usd"]
              .agg(["count", "median"]).sort_values("median", ascending=False))
print(f"Indonesia-based companies: {len(idn)} records, median USD {idn['salary_in_usd'].median():,.0f}")
remote_from_id = df[(df["employee_residence"] == "ID")]
print(f"Employees living in Indonesia: {len(remote_from_id)} records")
asia_med
""")
nb.md("""
**Findings**

* **Experience is the biggest pay lever.** The median roughly doubles from entry level to senior.
* **Fully remote roles pay about the same as on-site roles at the median** (USD 135k vs 140k), while hybrid roles pay less, so remote work does not mean a lower salary.
* **The share of fully remote roles fell from about 53% in 2022 to 33% in 2023**, so remote jobs are more competitive and a strong portfolio matters more.
* **Medium-sized companies pay the highest median** in this sample. These are mostly US tech firms.
* **Indonesia has very few records**, so the local median is only indicative. Asian markets in this sample pay far below US and European companies, which is why remote roles with foreign employers are attractive.
* *Caveat:* the data is self-reported and dominated by US companies, so treat country comparisons as directional.
""")
nb.save("data_science_salaries.ipynb")

# ---------------------------------------------------------------- e-commerce
nb = Notebook(".")
nb.md("""
# E-commerce Sales and Profit Analysis (2023)

**Questions:** where does revenue and profit come from, how does it move through the year, and are shipping and
discounts helping or hurting? Built on the e-commerce dataset from my Digital Skola *Alchemist Project*.

**Data:** 51,282 order lines from 2023 across 4 product categories and 14 regions (`data/ecommerce_orders.csv.gz`).
""")
nb.code(SETUP)
nb.code("""
df = pd.read_csv("data/ecommerce_orders.csv.gz", parse_dates=["Order Date", "Ship Date"])
print(df.shape, "| period:", df["Order Date"].min().date(), "to", df["Order Date"].max().date())
print("Missing values:\\n", df.isna().sum()[df.isna().sum() > 0].to_string())
df = df.dropna(subset=["Customer ID", "Region"])
print("Region values with a typo:", df.loc[df["Region"] == "4orth", "Region"].count())
df["Region"] = df["Region"].replace({"4orth": "North"})   # data-entry typo
df["month"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
kpi = pd.Series({
    "Revenue": df["Sales"].sum(),
    "Profit": df["Profit"].sum(),
    "Profit margin %": df["Profit"].sum() / df["Sales"].sum() * 100,
    "Orders": df["Order Id"].nunique(),
    "Customers": df["Customer ID"].nunique(),
    "Avg delivery days": df["Aging"].mean(),
}).round(1)
kpi
""")
nb.code("""
monthly = df.groupby("month")[["Sales", "Profit"]].sum()
ax = monthly.plot(figsize=(11, 4), marker="o")
ax.set(title="Monthly revenue and profit, 2023", xlabel="", ylabel="USD")
best, worst = monthly["Sales"].idxmax(), monthly["Sales"].idxmin()
print(f"Best month: {best:%B} (${monthly['Sales'].max():,.0f}) | weakest: {worst:%B} (${monthly['Sales'].min():,.0f})")
print(f"Best month is {monthly['Sales'].max() / monthly['Sales'].min():.1f}x the weakest")
""")
nb.code("""
cat = (df.groupby("Product Category")
         .agg(revenue=("Sales", "sum"), profit=("Profit", "sum"), orders=("Order Id", "nunique")))
cat["margin_%"] = cat["profit"] / cat["revenue"] * 100
cat["revenue_share_%"] = cat["revenue"] / cat["revenue"].sum() * 100
top_products = (df.groupby("Product")["Profit"].sum().sort_values(ascending=False).head(10))
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
cat["revenue"].sort_values().plot(kind="barh", ax=axes[0], color="#4C72B0")
axes[0].set(title="Revenue by category", xlabel="USD", ylabel="")
top_products.sort_values().plot(kind="barh", ax=axes[1], color="#55A868")
axes[1].set(title="Top 10 products by profit", xlabel="USD", ylabel="")
plt.tight_layout()
cat.sort_values("revenue", ascending=False).round(1)
""")
nb.code("""
region = (df.groupby("Region").agg(revenue=("Sales", "sum"), profit=("Profit", "sum"))
            .assign(margin_pct=lambda x: x["profit"] / x["revenue"] * 100)
            .sort_values("revenue", ascending=False))
ax = region["revenue"].sort_values().plot(kind="barh", figsize=(8, 5), color="#4C72B0")
ax.set(title="Revenue by region", xlabel="USD", ylabel="")
region.round(1)
""")
nb.code("""
ship = df.groupby("Ship Mode").agg(lines=("Order Id", "size"), avg_delivery_days=("Aging", "mean"),
                                   avg_shipping_cost=("Shipping Cost", "mean"),
                                   margin_pct=("Profit", lambda s: s.sum() / df.loc[s.index, "Sales"].sum() * 100))
df["discount_band"] = pd.cut(df["Discount"], [0, 0.02, 0.03, 0.04, 0.05], labels=["1-2%", "3%", "4%", "5%"])
disc = df.groupby("discount_band", observed=True).apply(
    lambda g: pd.Series({"lines": len(g), "avg_quantity": g["Quantity"].mean(),
                         "margin_pct": g["Profit"].sum() / g["Sales"].sum() * 100}), include_groups=False)
print("Shipping mode:")
print(ship.round(2).to_string())
print("\\nDiscount level:")
print(disc.round(2).to_string())
""")
nb.md("""
**Findings and recommendations**

* **Fashion is the business.** It brings 65% of revenue at the best margin (47.7%), so protect stock and marketing spend there first.
* **Revenue is steady through the year.** December, the best month, is only about 14% above February, the weakest. Growth has to come from more customers and orders, not from seasonal peaks.
* **Discounts give away margin without adding volume.** The average quantity per order is about 3.0 at every discount level, while the margin falls from 50% at 1–2% discounts to 41% at 5%. Recommendation: cap routine discounts at 2% and test targeted offers instead.
* **Same Day delivery arrives in 1 day versus about 5.5 days for other modes, at a similar shipping cost**, which makes it a strong candidate for premium positioning.
* **Data quality:** one region label had a typo ("4orth") and was corrected to North. One row with a missing customer and region was removed.

*Note:* the dataset is a teaching dataset with simulated orders, so the patterns illustrate the method rather than a real company.
""")
nb.save("ecommerce_sales_analysis.ipynb")
print("saved both")
